# Migration auf GPT-Realtime-2 / GPT-Realtime-2.1

> Implementierungsplan für den Umstieg auf die neuen OpenAI-Realtime-Modelle
> (`gpt-realtime-2`, `gpt-realtime-2.1`) und die dazugehörigen nativen Features
> **Parallel Tool Calls**, **Preambles** und **Async Function Calling**.
>
> Zielgruppe: der implementierende Agent. Dieses Dokument ist bewusst
> ausführlich, damit die Umsetzung ohne weitere Recherche möglich ist.

---

## 0. TL;DR / Was ist zu tun

1. **Modell-Enum aktualisieren**: Default auf `gpt-realtime-2.1`, Mini-Variante ergänzen,
   deprecatete Modelle markieren.
2. **Parallel Tool Calls unterstützen**: `ToolCallHandler` von „ein `response.create`
   pro Tool" auf „ein `response.create` pro Response-Batch" umbauen. Das ist der
   größte und wichtigste Umbau.
3. **Preambles nativ nutzen statt selbst zu emulieren**: Das Konstrukt
   `holding_instruction` / `_send_holding_message` wird durch das native
   Preamble-Verhalten des Modells ersetzt (via Instructions gesteuert).
4. **Async Function Calling nativ nutzen**: Die `SupervisorCoordinator`-Maschinerie,
   die die Konversation während langer Tool-Calls „am Leben hält", wird verschlankt.
   Das Modell hält die Konversation jetzt selbst flüssig und sagt bei Nachfragen
   „ich arbeite noch daran".
5. **Legacy-Beta-Reste entfernen**: Beta-only Events/Enums (`response.text.delta`,
   `transcription_session.*`) aus `schemas.py` entfernen.
6. **`result_instruction`-Semantik anpassen**: Bei parallelen Calls darf nur **ein**
   `response.create` gesendet werden — pro-Tool-Instructions müssen zusammengeführt
   oder fallengelassen werden.

Der Repo ist bereits weitgehend auf dem **GA-Schema** (Commit „Adjust api to
realtime-2 model"): `session.type = "realtime"`, `output_modalities`,
`audio.output`-Nesting und `reasoning.effort` sind schon korrekt. Es geht also
**nicht** um die Beta→GA-Schema-Migration selbst, sondern um das saubere Nutzen
der neuen Modell-Fähigkeiten.

---

## 1. Rechercheergebnisse (Stand Juli 2026)

### 1.1 Modelle und Deprecations

OpenAI hat die Realtime-API im Mai 2026 auf GA gehoben und eine neue
Modellgeneration eingeführt. Relevante Fakten:

| Modell | Status | Ersatz / Hinweis | Shutdown |
|---|---|---|---|
| `gpt-realtime-2.1` | **aktuell empfohlen** | Produktions-Default | – |
| `gpt-realtime-2.1-mini` | aktuell | günstigere Variante | – |
| `gpt-realtime-2` | aktuell | Vorgänger von 2.1, weiterhin gültig | – |
| `gpt-realtime-1.5` | aktuell | Nachfolger der Preview-Modelle | – |
| `gpt-realtime` | **deprecated** | → `gpt-realtime-2.1` | 2027-01-20 |
| `gpt-realtime-mini` | **deprecated** | → `gpt-realtime-2.1-mini` | 2027-01-20 |
| `gpt-4o-realtime` / `gpt-4o-realtime-preview` | deprecated | → `gpt-realtime-2.1` | 2026-05-07 / 2027-01-20 |

Zusätzlich neu (nicht Teil dieses Repos, aber gut zu wissen):
- `gpt-realtime-translate` — Streaming-Sprachübersetzung (70+ → 13 Sprachen).
- `gpt-realtime-whisper` — Streaming-STT (bessere Alternative zu `whisper-1` für
  Input-Transkription).

**`gpt-realtime-2` / `2.1` Eckdaten** (offizielle Model-Page):
- Kontextfenster: **128k** Token (vorher 32k), max. Output **32k** Token.
- Reasoning-Effort-Stufen: `minimal`, `low`, `medium`, `high`, `xhigh` (Default `low`).
- Preise (pro 1M Token): Text In $4.00 / Out $24.00 / Cached In $0.40;
  Audio In $32.00 / Out $64.00.
- Time-to-first-audio: ~1.12 s (minimal) bis ~2.33 s (high).

### 1.2 Beta → GA (bereits erledigt, nur zur Kontrolle)

Diese Änderungen sind im Repo **schon umgesetzt**, hier nur als Checkliste zum
Verifizieren, dass keine Beta-Reste übrig sind:

- `OpenAI-Beta: realtime=v1` Header **entfernt** → siehe `providers/openai.py`,
  dort wird nur `Authorization` gesetzt. ✅
- `session.type = "realtime"` gesetzt. ✅ (`RealtimeSessionSettings`)
- Audio-Config unter `session.audio.input` / `session.audio.output`. ✅
- `output_modalities` statt `modalities`. ✅
- Content-Typen `input_text` / `output_text` / `input_audio` / `output_audio`. ✅
- Event-Umbenennungen `response.text.* → response.output_text.*`,
  `response.audio.* → response.output_audio.*`,
  `response.audio_transcript.* → response.output_audio_transcript.*`. ✅ (GA-Namen
  vorhanden) — **aber**: die Beta-Namen `response.text.delta` / `response.text.done`
  und `transcription_session.*` sind noch als Enum-Werte vorhanden und sollten
  entfernt werden (siehe §4.6).

### 1.3 Die drei neuen Fähigkeiten im Detail

**(a) Preambles** — Das Modell spricht von sich aus kurze Überbrückungsfloskeln,
bevor bzw. während es ein Tool aufruft: „let me check that", „one moment while I
look into it", „checking your calendar". Es gibt **keinen** dedizierten
Config-Flag; das Verhalten wird über die **System-Instructions** gesteuert
(z. B. „Before calling a tool that may take a moment, briefly tell the user what
you are about to do."). Das Modell macht die Tool-Aktivität hörbar, statt in
Stille zu verfallen.

**(b) Parallel Tool Calls** — Das Modell kann in **einer** Response mehrere
Funktionen gleichzeitig aufrufen. Auf Event-Ebene bedeutet das: mehrere
`response.function_call_arguments.done`-Events (jeweils eigenes `call_id` /
`item_id`) innerhalb **einer** Response, gefolgt von **einem** `response.done`.
Der Client muss für **jeden** Call ein `conversation.item.create`
(`function_call_output`) mit passendem `call_id` senden und **danach genau ein**
`response.create`, um die Folge-Response auszulösen. Das ist der kritische Punkt:
Ein `response.create` **pro** Tool-Output (wie aktuell im Code) ist bei parallelen
Calls falsch und führt zu Race Conditions / vorzeitigen oder doppelten Responses.

**(c) Async Function Calling** — Lang laufende Tool-Calls blockieren die Session
nicht mehr. Das Modell führt die Konversation flüssig weiter, während es auf ein
Ergebnis wartet, und antwortet auf Nachfragen mit „ich warte noch darauf" o. Ä.
**Nativ aktiviert** für die neuen Modelle, kein Code-Flag nötig. Das
`function_call_output` kann jederzeit später gesendet werden; das Modell fügt das
Ergebnis dann in den laufenden Dialog ein.

### 1.4 Quellen

- [Advancing voice intelligence with new models in the API — OpenAI](https://openai.com/index/advancing-voice-intelligence-with-new-models-in-the-api/)
- [GPT-Realtime-2 Model — OpenAI API Docs](https://developers.openai.com/api/docs/models/gpt-realtime-2)
- [Realtime Guide (Beta→GA Migration) — OpenAI API Docs](https://developers.openai.com/api/docs/guides/realtime)
- [Deprecations — OpenAI API Docs](https://developers.openai.com/api/docs/deprecations)
- [Developer notes on the Realtime API — OpenAI](https://developers.openai.com/blog/realtime-api)
- [Introducing gpt-realtime — OpenAI](https://openai.com/index/introducing-gpt-realtime/) (Async Function Calling: „available natively … developers do not need to update their code")
- [AINews: GPT-Realtime-2, -Translate, -Whisper — latent.space](https://www.latent.space/p/ainews-gpt-realtime-2-translate-and)

---

## 2. Ist-Zustand im Repo (relevante Dateien)

| Datei | Rolle | Betroffen? |
|---|---|---|
| `rtvoice/agent/views.py` | `RealtimeModel`, `ReasoningEffort`, `TranscriptionModel` Enums | **Ja** — Modell-Enum |
| `rtvoice/realtime/schemas.py` | Alle Pydantic-Events + Session-Settings | **Ja** — Legacy-Cleanup, Response-Parsing |
| `rtvoice/handler/tool_call_handler.py` | Reguläre Tool-Ausführung | **Ja** — Parallel-Batching |
| `rtvoice/handler/tool_call_helpers.py` | `send_function_call_output`, `send_response_event` | **Ja** — Batch-fähige Helfer |
| `rtvoice/handler/supervisor_coordinator.py` | Langläufer-Tool „Supervisor" | **Ja** — Holding-Message entfernen, verschlanken |
| `rtvoice/tools/views.py` | `Tool` (mit `holding_instruction`, `result_instruction`, `status`) | **Ja** — `holding_instruction` deprecaten |
| `rtvoice/tools/tools.py` | `Tools`-Registry / `action`-Decorator | **Ja** — `holding_instruction`-Param |
| `rtvoice/realtime/session.py` | Verdrahtung aller Handler + Session-Update | **Ja** — Default-Instructions für Preambles |
| `rtvoice/realtime/websocket.py` | WS-Transport, Event-Parsing | Nein (nur ggf. neue Event-Typen im Union) |

**Wichtige Eigenschaft des Event-Busts**: In `session.py::_forward_events` werden
Events **sequenziell** verarbeitet (`async for event ... await dispatch(event)`).
D. h. alle `FunctionCallItem`-Handler eines Response-Batches laufen **vor** dem
`ResponseDoneEvent`-Handler. Das ist die Grundlage für das Batching-Design unten:
Wenn `ResponseDoneEvent` eintrifft, sind alle Tool-Calls dieser Response bereits
registriert. **Voraussetzung**: Die Handler dürfen die eigentliche Tool-Ausführung
**nicht inline awaiten** (das würde die Forward-Loop blockieren), sondern müssen
sie als Task starten.

---

## 3. Ziel-Architektur / Designentscheidungen

### Entscheidung 1 — Ein `response.create` pro Response, nicht pro Tool

Der aktuelle Ablauf in `ToolCallHandler._handle_tool_call`:

```
FunctionCallItem → execute → send function_call_output → send response.create
```

wird ersetzt durch einen **Response-scoped Batch**:

```
FunctionCallItem (call A) → Task A starten, in Batch[response_id] registrieren
FunctionCallItem (call B) → Task B starten, in Batch[response_id] registrieren
ResponseDoneEvent(response_id) → await A,B → send output A → send output B → EIN response.create
```

Begründung: Genau ein `response.create` nach dem Absenden **aller** Outputs ist
das von OpenAI vorgeschriebene Muster für parallele Calls. Das `ResponseDoneEvent`
ist der zuverlässige „alle Calls dieser Response sind bekannt"-Trigger, weil der
Bus sequenziell dispatcht (siehe §2).

### Entscheidung 2 — Preambles nativ, `holding_instruction` raus

Das Modell spricht Überbrückungsfloskeln jetzt selbst. Das explizite Senden einer
zweiten `response.create` mit `tool_choice=NONE` (`_send_holding_message`) würde
mit dem nativen Preamble **kollidieren** (Doppel-Sprache, Timing-Konflikte).
→ `holding_instruction` wird **deprecated** (Parameter bleibt vorerst als No-op für
API-Kompatibilität, Nutzung wird entfernt), das Preamble-Verhalten wird stattdessen
über die **System-Instructions** gesteuert (siehe §4.5).

### Entscheidung 3 — Supervisor verschlanken, aber nicht abschaffen

Async Function Calling macht die „Konversation-am-Leben-halten"-Mechanik nativ.
Der `SupervisorCoordinator` behält aber echten Mehrwert:
`cancel_supervisor`, `update_supervisor` und den Clarification-Flow. Diese bleiben.
Entfernt/vereinfacht wird nur:
- `_send_holding_message` (Preamble ist nativ),
- die Annahme, dass ohne Holding-Message „Stille" entsteht.

Der Supervisor läuft weiter als Background-Task (er ist echt langlaufend und braucht
Cancel/Update) und liefert sein Ergebnis via `function_call_output` + `response.create`
nach — das passt bereits zum Async-Modell.

### Entscheidung 4 — `result_instruction` bei parallelen Calls

Bei N parallelen Tools mit je eigener `result_instruction` kann nur **eine**
Folge-Response erzeugt werden. Regel:
- Genau ein Tool im Batch mit `result_instruction` → diese als Instruction der
  einen `response.create` verwenden.
- Mehrere Tools mit `result_instruction` → Instructions **konkatenieren** (durch
  Zeilenumbruch) und als eine Instruction senden; alternativ ganz weglassen und
  dem Modell die (native) Formulierung überlassen. **Empfehlung: konkatenieren**,
  damit bestehende Tool-Semantik erhalten bleibt.
- Kein Tool mit `result_instruction` → plain `response.create()`.

---

## 4. Konkrete Umbauten (Schritt für Schritt)

### 4.1 `rtvoice/agent/views.py` — Modell-Enum

```python
class RealtimeModel(StrEnum):
    # Aktuell empfohlen
    GPT_REALTIME_2_1 = "gpt-realtime-2.1"
    GPT_REALTIME_2_1_MINI = "gpt-realtime-2.1-mini"
    GPT_REALTIME_2 = "gpt-realtime-2"
    GPT_REALTIME_1_5 = "gpt-realtime-1.5"
    # Deprecated (Shutdown 2027-01-20) — nur aus Kompatibilität behalten
    GPT_REALTIME = "gpt-realtime"            # → GPT_REALTIME_2_1
    GPT_REALTIME_MINI = "gpt-realtime-mini"  # → GPT_REALTIME_2_1_MINI
```

- **Default anpassen**: In `schemas.py::RealtimeSessionSettings.model` ist der
  Default aktuell `RealtimeModel.GPT_REALTIME_2`. Auf `GPT_REALTIME_2_1` setzen.
- Prüfen, wo `RealtimeModel` als Default in der öffentlichen API/Builder gesetzt
  wird (z. B. `RealtimeAgent`-Konstruktion, `examples/`), und dort ebenfalls auf
  `2.1` heben. `grep -rn "GPT_REALTIME_2\b\|gpt-realtime-2\b\|RealtimeModel\." rtvoice examples`.
- Optional: `TranscriptionModel` um `gpt-realtime-whisper` erweitern (bessere
  Streaming-STT als `whisper-1`), aber **nicht** als Default erzwingen — das ist
  ein separates, risikoarmes Follow-up. `whisper-1` bleibt gültig.

### 4.2 `rtvoice/realtime/schemas.py` — Response-Output parsen

Damit das Batching robust ist (und optional die Zahl der Function-Calls pro
Response verifiziert werden kann), sollte `RealtimeResponseObject` die
Output-Items lesbar machen. Minimal-invasiv:

```python
class RealtimeResponseObject(BaseModel):
    id: str
    status: str | None = None
    usage: TokenUsage | None = None
    # NEU: rohe Output-Items, um Function-Calls einer Response zu zählen
    output: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def function_call_ids(self) -> list[str]:
        return [
            item["call_id"]
            for item in self.output
            if item.get("type") == "function_call" and "call_id" in item
        ]
```

> Hinweis: Das Batching in §4.3 funktioniert **auch ohne** dieses Feld, weil der
> Bus sequenziell dispatcht und beim `ResponseDoneEvent` alle `FunctionCallItem`s
> bereits registriert sind. `function_call_ids` ist eine **Verifikations-/Robustheits-
> Reserve** (z. B. um zu prüfen, ob wirklich alle erwarteten Outputs vorliegen).
> Wenn man es minimal halten will, kann dieser Schritt entfallen.

### 4.3 `rtvoice/handler/tool_call_handler.py` — Parallel-Batching (Kernstück)

Neues Verhalten:

```python
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rtvoice.events import EventBus
from rtvoice.handler.tool_call_helpers import (
    send_batched_response,          # NEU (siehe 4.4)
    send_function_call_output,
    serialize_tool_result,
)
from rtvoice.realtime.schemas import FunctionCallItem, ResponseDoneEvent
from rtvoice.realtime.websocket import RealtimeWebSocket

if TYPE_CHECKING:
    from rtvoice.tools import Tools
    from rtvoice.tools.views import Tool

logger = logging.getLogger(__name__)


@dataclass
class _PendingCall:
    call_id: str
    tool: Tool
    task: asyncio.Task


@dataclass
class _ResponseBatch:
    response_id: str
    calls: list[_PendingCall] = field(default_factory=list)


class ToolCallHandler:
    def __init__(
        self,
        event_bus: EventBus,
        tools: Tools,
        websocket: RealtimeWebSocket,
        supervisor_tool_name: str | None = None,
    ) -> None:
        self._tools = tools
        self._websocket = websocket
        self._supervisor_tool_name = supervisor_tool_name
        self._batches: dict[str, _ResponseBatch] = {}

        event_bus.subscribe(FunctionCallItem, self._on_function_call)
        event_bus.subscribe(ResponseDoneEvent, self._on_response_done)

    async def _on_function_call(self, event: FunctionCallItem) -> None:
        # Supervisor-Tool wird vom SupervisorCoordinator behandelt.
        if event.name == self._supervisor_tool_name:
            return

        tool = self._tools.get(event.name)
        if not tool:
            logger.error("Tool '%s' not found", event.name)
            return

        logger.info(
            "Tool call: '%s' [args=%s]",
            event.name,
            json.dumps(event.arguments or {}, ensure_ascii=False),
        )

        # Ausführung SOFORT und PARALLEL starten (nicht inline awaiten!)
        task = asyncio.create_task(
            self._tools.execute(event.name, event.arguments or {})
        )
        batch = self._batches.setdefault(
            event.response_id, _ResponseBatch(response_id=event.response_id)
        )
        batch.calls.append(_PendingCall(call_id=event.call_id, tool=tool, task=task))

    async def _on_response_done(self, event: ResponseDoneEvent) -> None:
        batch = self._batches.pop(event.response_id, None)
        if not batch or not batch.calls:
            return  # Response ohne (reguläre) Tool-Calls

        # Auf alle parallelen Ausführungen warten
        results = await asyncio.gather(
            *(c.task for c in batch.calls), return_exceptions=True
        )

        # Alle function_call_output-Items senden (noch KEIN response.create)
        result_instructions: list[str] = []
        for call, result in zip(batch.calls, results):
            if isinstance(result, Exception):
                logger.exception("Tool '%s' failed", call.tool.name, exc_info=result)
                serialized = f"Tool execution failed: {result}"
            else:
                serialized = serialize_tool_result(result)
                logger.info("Tool result: '%s' [result=%s]", call.tool.name, serialized)

            await send_function_call_output(self._websocket, call.call_id, serialized)

            if call.tool.result_instruction:
                result_instructions.append(call.tool.result_instruction)

        # GENAU EIN response.create für den gesamten Batch
        await send_batched_response(self._websocket, result_instructions)
```

Wichtige Punkte:
- **Kein inline-`await` der Tool-Ausführung** im `FunctionCallItem`-Handler
  (sonst blockiert die Forward-Loop und Parallelität geht verloren).
- `return_exceptions=True` in `gather`, damit ein fehlgeschlagenes Tool die
  anderen Outputs nicht verschluckt; Fehler werden als Text-Output ans Modell
  zurückgegeben (analog zum bisherigen impliziten Verhalten).
- Der **einzelne** `response.create` wird über `send_batched_response` erzeugt
  (Instruction-Merging, siehe §4.4).
- Der `supervisor_tool_name`-Filter bleibt; der Supervisor läuft weiter über den
  `SupervisorCoordinator`.

### 4.4 `rtvoice/handler/tool_call_helpers.py` — Batch-fähiger Response-Helfer

`send_response_event(ws, tool)` (pro-Tool) wird durch `send_batched_response`
(pro-Batch) ersetzt. Alt lassen für den Supervisor-Pfad (der genau ein Tool
liefert) oder auf den neuen Helfer umstellen.

```python
async def send_batched_response(
    ws: RealtimeWebSocket, result_instructions: list[str]
) -> None:
    """Sendet GENAU EIN response.create für einen (ggf. parallelen) Tool-Batch."""
    if not result_instructions:
        await ws.send(ConversationResponseCreateEvent())
        return

    merged = "\n".join(result_instructions)
    await ws.send(ConversationResponseCreateEvent.from_instructions(merged))
```

Der bestehende `send_response_event(ws, tool)` kann als dünner Wrapper bleiben:

```python
async def send_response_event(ws: RealtimeWebSocket, tool: Tool) -> None:
    instructions = [tool.result_instruction] if tool.result_instruction else []
    await send_batched_response(ws, instructions)
```

### 4.5 `rtvoice/realtime/session.py` — Preambles über Instructions aktivieren

Da Preambles instruction-gesteuert sind, sollte die Session-Instruction einen
Standard-Hinweis enthalten (oder ein optionales Flag `enable_preambles: bool`
am Agent/Builder, das diesen Textbaustein anhängt):

```python
_PREAMBLE_GUIDANCE = (
    "When you are about to call a tool that may take a moment, first say a short, "
    "natural acknowledgment of what you are doing (e.g. 'let me check that', "
    "'one moment while I look that up'). If the user asks for a result that is not "
    "ready yet, tell them you are still working on it."
)
```

Diesen Baustein in `_build_session_settings()` an die `instructions` anhängen
(nur wenn Tools registriert sind bzw. wenn `enable_preambles` gesetzt ist). So
wird das native Preamble-/Async-Verhalten zuverlässig ausgelöst, ohne eigene
Holding-Responses zu senden.

> Optional/empfehlenswert: `enable_preambles` als Konstruktor-Parameter der
> `RealtimeSession` bzw. des öffentlichen Agent-Builders durchreichen, Default `True`
> für die neuen Modelle. So bleibt das Verhalten testbar und abschaltbar.

### 4.6 `rtvoice/handler/supervisor_coordinator.py` — Holding-Message entfernen

- `_send_holding_message` und der Aufruf in `_handle_tool_call` **entfernen**.
  Begründung: Native Preambles übernehmen die Überbrückung; die explizite
  `response.create` mit `tool_choice=NONE` kollidiert damit.
- `PendingSupervisorCall` / Cancel / Update / Clarification **unverändert lassen**.
- Der Ergebnis-Pfad (`_deliver_supervisor_result` → `send_function_call_output`
  + `send_response_event`) bleibt; er passt bereits zum Async-Modell (Output wird
  nachgeliefert, dann eine `response.create`).
- **Parallel-Edge-Case dokumentieren**: Falls das Modell den Supervisor-Handoff
  zusammen mit einem regulären Tool in **derselben** Response aufruft, erzeugt der
  `ToolCallHandler`-Batch **ein** `response.create` (nach den regulären Outputs),
  und der Supervisor liefert später **ein weiteres** `response.create` nach.
  Mit Async Function Calling ist das unkritisch (die zweite Response integriert das
  Supervisor-Ergebnis). In der Praxis parallelisiert das Modell einen Handoff
  selten mit anderen Tools; kein Sonderhandling nötig, aber als bekannter Fall
  vermerken.

### 4.7 `rtvoice/tools/views.py` + `tools.py` — `holding_instruction` deprecaten

- `holding_instruction` als Parameter **belassen** (API-Kompatibilität), aber:
  - Docstring/Kommentar ergänzen: „Deprecated — natives Preamble-Verhalten des
    Modells nutzen; wird nicht mehr aktiv gesendet."
  - Alle internen Verwendungen entfernen (nur noch in `supervisor_coordinator.py`
    referenziert → dort entfällt sie mit §4.6).
- `result_instruction` und `status` bleiben unverändert.

### 4.8 `rtvoice/realtime/schemas.py` — Legacy-Beta-Reste entfernen

Nach Verifikation (grep, Tests) entfernen:
- `RealtimeClientEvent.TRANSCRIPTION_SESSION_UPDATE`
- `RealtimeServerEvent.TRANSCRIPTION_SESSION_CREATED` / `TRANSCRIPTION_SESSION_UPDATED`
- `RealtimeServerEvent.RESPONSE_TEXT_DELTA` / `RESPONSE_TEXT_DONE`
  (Beta-Namen `response.text.delta` / `response.text.done`; GA nutzt
  `response.output_text.*`, die bereits vorhanden sind)
- Die Pydantic-Modelle `ResponseTextDelta` / `ResponseTextDone` und ihre Einträge
  im `ServerEvent`-Union.

> **Vorsicht**: Erst per `grep -rn "ResponseTextDelta\|ResponseTextDone\|RESPONSE_TEXT_\|TRANSCRIPTION_SESSION" rtvoice tests`
> prüfen, ob Tests oder Handler darauf hören. `TranscriptionAccumulator` nutzt
> vermutlich die `output_audio_transcript.*`- und `input_audio_transcription.*`-
> Events — das sind die GA-Namen und bleiben. Nur die reinen Beta-Duplikate raus.

---

## 5. Umgang mit „Tools, die vorher nativ emulierte Features gemacht haben"

Das ist die zentrale konzeptionelle Frage. Zuordnung alt → neu:

| Bisheriges Konstrukt | Was es emuliert hat | Neue Behandlung |
|---|---|---|
| `holding_instruction` + `_send_holding_message` | **Preamble** („one moment…") | **Entfernen.** Nativ via Instructions (§4.5). |
| `SupervisorCoordinator` Keep-alive-Maschinerie | **Async Function Calling** (Konversation läuft weiter während langem Tool) | **Verschlanken.** Nativ; Coordinator behält nur Cancel/Update/Clarification. |
| `result_instruction` + pro-Tool `response.create` | Follow-up-Response nach Tool-Output | **Behalten, aber batchen** (§4.3/4.4): ein `response.create` pro Response, Instructions gemergt. |
| `status` (Callable/Template) | Sichtbarer Status-Text an UI | **Behalten** — reine Client-UX, kein API-Feature; unabhängig von Modell-Änderungen. |

Grundprinzip: **Nicht mehr selbst emulieren, was das Modell nativ kann.** Alles,
was reine Client-seitige UX ist (`status`, Recording, Transkript-Anzeige), bleibt.
Alles, was nur existierte, um API-Lücken (kein Preamble, kein Async) zu
kompensieren, wird auf das native Verhalten zurückgebaut.

---

## 6. Tests

Bestehende Tests, die anzupassen sind:
- `tests/watchdogs/test_tool_calling_watchdog.py` — Erwartung „ein `response.create`
  pro Tool" auf „ein `response.create` pro Response-Batch" umstellen.
- `tests/supervisor/test_supervisor_agent_loop.py` /
  `tests/watchdogs/test_supervisor_interaction_watchdog.py` — Assertions auf das
  Senden der Holding-Message entfernen.
- Modell-Default-Assertions (falls `gpt-realtime-2` irgendwo hart erwartet wird).

Neue Tests:
1. **Parallel Tool Calls**: Zwei `FunctionCallItem` mit demselben `response_id`,
   danach ein `ResponseDoneEvent`. Erwartung: zwei `function_call_output`-Sends,
   **genau ein** `response.create` danach.
2. **Fehlertoleranz im Batch**: Ein Tool wirft, das andere liefert. Erwartung:
   beide Outputs gesendet (Fehler als Text), ein `response.create`.
3. **`result_instruction`-Merging**: Zwei Tools mit je eigener `result_instruction`.
   Erwartung: ein `response.create` mit konkatenierter Instruction.
4. **Single-Call-Regression**: Ein Tool + `ResponseDoneEvent` → verhält sich wie
   bisher (ein Output, ein `response.create`).
5. **Reihenfolge-Garantie**: Test, dass `FunctionCallItem` vor `ResponseDoneEvent`
   dispatcht wird (dokumentiert die Bus-Invariante, auf der das Batching beruht).

Hinweis: Der Event-Bus dispatcht sequenziell — in Tests die Events in dieser
Reihenfolge einspeisen und `asyncio`-Tasks vor der Assertion durchlaufen lassen
(`await asyncio.sleep(0)` bzw. auf die Tasks warten).

---

## 7. Reihenfolge der Umsetzung (empfohlen)

1. **§4.1** Modell-Enum + Default (risikoarm, sofort nutzbar).
2. **§4.8** Legacy-Cleanup (isoliert, per grep abgesichert).
3. **§4.3 + §4.4** Parallel-Batching im `ToolCallHandler` + Helfer (Kernstück) inkl.
   neuer Tests (§6.1–6.5).
4. **§4.6 + §4.7** Supervisor entschlacken, `holding_instruction` deprecaten.
5. **§4.5** Preamble-Instructions verdrahten (+ optionales `enable_preambles`-Flag).
6. **§4.2** (optional) `RealtimeResponseObject.output` für Robustheit/Verifikation.
7. Volllauf `pytest`, manueller Smoke-Test mit `examples/showcase.py` gegen
   `gpt-realtime-2.1` (echte Parallel-Tool-Szene: z. B. „Wetter UND Kalender").

---

## 8. Offene Fragen / Risiken

- **Verfügbarkeit `gpt-realtime-2.1-mini`**: Als Deprecation-Ersatz dokumentiert;
  vor Default-Nutzung gegen die eigene Org-Quota/Verfügbarkeit prüfen.
- **Preamble-Zuverlässigkeit**: Rein instruction-gesteuert → nicht 100 % garantiert.
  Wenn deterministisches Überbrücken nötig ist, kann ein sehr kurzer Prompt-Baustein
  helfen; ein hartes „Holding-Response senden" sollte aber **nicht** reaktiviert
  werden (Kollision mit nativem Preamble).
- **`response.create`-Timing bei Mischung Supervisor + reguläres Tool** in einer
  Response (siehe §4.6) — als bekannter, unkritischer Edge-Case dokumentiert.
- **`whisper-1` → `gpt-realtime-whisper`**: separates Follow-up, kein Blocker.
- **Client-Secrets/WebRTC** (`POST /v1/realtime/client_secrets`,
  `/v1/realtime/calls`): Dieses Repo nutzt die **WebSocket**-Server-Verbindung mit
  API-Key — davon **nicht** betroffen. Nur relevant, falls künftig Browser-/
  WebRTC-Clients angebunden werden.
```
