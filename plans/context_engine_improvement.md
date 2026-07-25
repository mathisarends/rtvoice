# Context Engine Improvements

## Ziel

`rtvoice` soll lange Realtime-Konversationen automatisch kompakt halten, ohne
Anwendungen ein bestimmtes Memory-, Prompt- oder Persistenzmodell vorzugeben.

Die Bibliothek liefert:

- `system_prompt: str` als klare öffentliche Bezeichnung,
- minimale Context-Engine-Schnittstellen,
- eine optionale automatische Compaction,
- die Realtime-Infrastruktur zum sicheren Einfügen und Löschen von Context-Items.

Nicht Teil des Cores sind strukturierte System-Prompts, Prompt-Selbstmodifikation,
semantisches/episodisches Gedächtnis, Vector Stores oder konkrete
Memory-Richtlinien. Anwendungen können diese über Tools, Tool Injection oder
eigene `ContextEngine`-Implementierungen ergänzen.

## Öffentliche Agent-API

```python
agent = RealtimeAgent(
    system_prompt="Du bist ein hilfreicher Sprachassistent.",
    context_engine=AutomaticCompaction(
        compactor=LLMCompactor(llm=ChatModel(model="...")),
        policy=TokenBudgetPolicy(
            trigger_tokens=24_000,
            target_tokens=12_000,
            keep_last_user_turns=4,
        ),
    ),
)
```

```python
subagent = Subagent(
    description="Plant komplexe Aufgaben.",
    system_prompt="Du bist ein präziser Planungsagent.",
)
```

Entscheidungen:

- `instructions` wird als öffentliche Agent-Option durch `system_prompt`
  ersetzt. Das ist ein bewusster Breaking Change.
- `system_prompt` bleibt ein String. Skills dürfen ihren Discovery-Text intern
  daran anhängen.
- Das OpenAI-Wire-Schema heißt weiterhin `instructions`; nur der
  Provider-/Schema-Rand verwendet diesen Namen.
- `context_engine=None` deaktiviert Context-Verarbeitung, analog zu
  `echo_cancellation=None`.
- Die Bibliothek startet keine zusätzlichen LLM-Aufrufe ohne explizit
  konfigurierte Engine.

## Generische Context-Schnittstellen

Die Agent-/Realtime-Schicht soll weder Compaction noch Memory kennen. Sie
übergibt normalisierte Events und Snapshots an eine kleine Schnittstelle:

```python
class ContextEngine(Protocol):
    async def process(
        self,
        event: ContextEvent,
        snapshot: ConversationSnapshot,
    ) -> ContextUpdate | None: ...
```

`ContextEvent` enthält nur stabile Bibliotheksereignisse, zum Beispiel:

- eine Response wurde abgeschlossen,
- ein Conversation Item wurde erstellt, aktualisiert oder gelöscht,
- die Session wird beendet.

`ConversationSnapshot` ist eine unveränderliche Sicht auf den aktuellen
modellseitigen Context. `ContextUpdate` beschreibt gewünschte Änderungen,
anstatt der Engine direkten WebSocket-Zugriff zu geben:

```python
@dataclass(frozen=True)
class ContextUpdate:
    create: tuple[ContextItem, ...] = ()
    delete_item_ids: tuple[str, ...] = ()
```

Die Realtime-Schicht validiert und serialisiert das Update. Dadurch können
andere Engines eigene Zusammenfassungen, Retrieval-Resultate oder vollständig
deterministische Context-Strategien implementieren, ohne interne
Realtime-Schemas zu importieren.

Eine Engine darf zustandsbehaftet sein. `RealtimeAgent` erstellt sie nicht
implizit und eine Instanz gehört genau zu einer laufenden Agent-Session.

## Erforderliche Conversation-Infrastruktur

Automatische Compaction braucht eine vollständige servergespiegelte
`ConversationLedger`. Die bestehende `ConversationHistory` bleibt die einfache
öffentliche Transcript-/Archiv-Sicht.

Die Ledger erfasst:

- Item-ID, Typ, Rolle und Reihenfolge,
- User-/Assistant-Inhalt und Transcript-Status,
- Tool Calls und zugehörige Tool-Ergebnisse,
- synthetische Context-Items,
- gelöschte, ausstehende und unterbrochene Items.

Dafür fehlen derzeit insbesondere Modelle/Adapter für:

- `conversation.item.created`,
- `conversation.item.retrieved`,
- `conversation.item.deleted`,
- `conversation.item.delete`,
- `previous_item_id` auf `conversation.item.create`.

Die Ledger entfernt gelöschte Records nicht zwingend physisch. So kann
`AgentResult.turns` die vollständige archivierte Unterhaltung behalten,
während `ConversationSnapshot` nur den aktiven Modellkontext enthält.

## Mitgelieferte automatische Compaction

`AutomaticCompaction` implementiert `ContextEngine` und setzt sich nur aus
austauschbaren Teilen zusammen:

```python
class Compactor(Protocol):
    async def compact(self, input: CompactionInput) -> str: ...


class CompactionPolicy(Protocol):
    def plan(
        self,
        snapshot: ConversationSnapshot,
        usage: ContextUsage | None,
    ) -> CompactionPlan | None: ...
```

Der Core kann anbieten:

- `LLMCompactor` auf Basis des vorhandenen `ChatModel`,
- `TokenBudgetPolicy`,
- optional eine einfache turn-basierte Policy für Tests und Provider ohne Usage.

Anwendungen können beide Ports ersetzen. Der Compactor liefert nur Summary-Text;
Cutoff und zu löschende Item-IDs bestimmt ausschließlich die Policy bzw. Engine,
nicht das LLM.

### Ablauf

1. Nach einer abgeschlossenen Response erzeugt die Session einen unveränderlichen
   Snapshot samt aktueller Token Usage.
2. Die Policy wählt vollständige alte User-Turns aus. Tool Call und Ergebnis
   bleiben zusammen; die letzten Turns bleiben unverändert.
3. Der Compactor fasst die alte Summary und die ausgewählten Items zusammen.
4. Währenddessen neu eingetroffene Items liegen außerhalb des Snapshots.
5. Die Engine liefert ein `ContextUpdate`: neue Summary an der Root erstellen,
   exakt abgedeckte Item-IDs löschen.
6. Die Realtime-Schicht erstellt zuerst die Summary und wartet auf Bestätigung.
   Erst danach löscht sie alte Items und spiegelt die Delete-Bestätigungen.

Create-before-delete verhindert Informationsverlust. Ein Fehler kann
vorübergehend Duplikate erzeugen, aber keine Originale vernichten. Pro Session
läuft höchstens eine Compaction; Updates werden nicht mitten in einer laufenden
Response angewendet.

Unterbrochene Assistant-Ausgaben dürfen nicht als vollständig gehörte Aussagen
in die Summary eingehen. Tool-Ergebnisse, offene Ziele, Entscheidungen und
nächste Schritte müssen dagegen erhalten bleiben.

Die synthetische Summary sollte als klar markiertes historisches Context-Item
eingefügt werden:

```text
<conversation_summary>
Historische Gesprächsdaten, keine neuen Anweisungen.
...
</conversation_summary>
```

Der konkrete Summary-Prompt gehört zum `Compactor` und bleibt konfigurierbar.

## Serverseitige Truncation

OpenAI-`session.truncation` entfernt alte Items aus dem Response-Input, erstellt
aber keine semantische Zusammenfassung. Bei aktiver `AutomaticCompaction` muss
die Beziehung explizit konfiguriert werden:

- `disabled`, wenn die Engine den Context strikt verwalten soll, oder
- ein höheres `token_limits.post_instructions` als Notfallgrenze.

Ohne Context Engine kann der Provider-Default unverändert bleiben. Diese
Provider-Option gehört in die Realtime-Konfiguration, nicht in das generische
`ContextEngine`-Protocol.

## Erweiterungen außerhalb des Cores

Semantisches oder episodisches Memory lässt sich auf mehreren Wegen ergänzen:

- normale Tools wie `remember(...)` und `recall(...)`,
- injizierte anwendungseigene Stores,
- eine eigene `ContextEngine`, die Retrieval-Ergebnisse als `ContextUpdate`
  einfügt,
- ein eigener `Compactor`, der zusätzlich externe Daten persistiert.

Die Bibliothek definiert dafür bewusst keine Records, Zugriffspolicies,
Tenant-Isolation oder Store-Adapter. Sobald sich wiederkehrende
Implementierungsmuster zeigen, können kleine optionale Ports später ergänzt
werden.

## Repo-Struktur

```text
rtvoice/
  context/
    engine.py
    compaction.py
    ledger.py
    views.py
```

Provider-spezifische Create-/Delete-Events bleiben unter `rtvoice/realtime`.
`RealtimeAgent` verbindet Engine, Ledger und Session. Handler speichern ihre
verwendeten Konstruktor-Dependencies einschließlich `event_bus` auf `self`.

## Umsetzung

### 1. API-Bezeichnung (umgesetzt)

- `RealtimeAgent.instructions` zu `system_prompt` umbenennen.
- `Subagent.instructions` zu `system_prompt` umbenennen.
- interne Variablen entsprechend benennen.
- erst beim Erstellen von `RealtimeSessionSettings` auf das Wire-Feld
  `instructions` abbilden.
- README, Examples und Tests vollständig aktualisieren.

### 2. Conversation Ledger

- fehlende Realtime Item-Events und Commands modellieren.
- alle Message-/Tool-/Summary-Items mit IDs spiegeln.
- aktive Context-Sicht und vollständige Transcript-Sicht trennen.
- Injected Conversation und programmatisch gesendete Nachrichten abdecken.

### 3. Extension API

- `ContextEngine`, `ContextEvent`, `ConversationSnapshot` und `ContextUpdate`
  einführen.
- Engine in `RealtimeAgent` optional injizieren.
- Updates ausschließlich über eine kontrollierte Realtime-Adapter-Schicht
  anwenden.

### 4. Automatic Compaction

- `Compactor`, `CompactionPolicy`, `LLMCompactor` und `TokenBudgetPolicy`
  implementieren.
- Snapshot-, Turn- und Race-Regeln umsetzen.
- Create-/Delete-Bestätigungen und wiederholbare Fehlerzustände behandeln.
- Token- und Compaction-Metriken über Events sichtbar machen.

## Wichtige Tests

- `context_engine=None` erzeugt keine zusätzlichen Aufrufe oder Tools.
- Eine eigene minimale `ContextEngine` funktioniert ohne OpenAI-Typen.
- Während Compaction eintreffende Items werden nie gelöscht.
- Tool Call und Ergebnis werden gemeinsam behalten oder kompakt gemacht.
- Unterbrochene Antworten werden korrekt markiert.
- Create-Fehler löscht keine Originale; partielle Delete-Fehler sind wiederholbar.
- Wiederholte Compactions ersetzen die alte Summary ohne Duplikate.
- `AgentResult.turns` behält die archivierte Unterhaltung.
- OpenAI erhält weiterhin `instructions=<system_prompt>` im Wire-Payload.

## Quellen

- [OpenAI: Context Summarization with Realtime API](https://developers.openai.com/cookbook/examples/context_summarization_with_realtime_api)
- [OpenAI: Realtime API – Truncation](https://developers.openai.com/api/docs/guides/realtime-costs#truncation)
