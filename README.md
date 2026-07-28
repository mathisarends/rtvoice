# rtvoice

![rtvoice banner](static/banner.jpg)

[![PyPI version](https://badge.fury.io/py/rtvoice.svg)](https://badge.fury.io/py/rtvoice)
[![Python Version](https://img.shields.io/badge/python-3.13%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)

A Python library for building real-time voice agents powered by the OpenAI Realtime API. It handles the full session lifecycle — microphone input, WebSocket streaming, turn detection, tool calling, and audio playback — so you can focus on what your agent does, not how it talks.

---

## Installation

```bash
pip install rtvoice[audio]
```

Requires Python 3.13+ and an `OPENAI_API_KEY` environment variable (or pass `api_key=` directly).

---

## Quickstart

```python
import asyncio
from rtvoice import RealtimeAgent

async def main():
    agent = RealtimeAgent(
        system_prompt="You are Jarvis, a concise and helpful voice assistant.",
    )
    await agent.run()

asyncio.run(main())
```

Run it, speak into your microphone, and the agent responds through your speakers. Press `Ctrl+C` to end the session.

`system_prompt` is a plain string that defines the agent's behavior.

---

## Table of Contents

- [Tool calling](#tool-calling)
  - [Basic tools](#basic-tools)
    - [Pydantic model tools](#pydantic-model-tools)
  - [Long-running tools](#long-running-tools)
  - [Status templates](#status-templates)
  - [Context injection](#context-injection)
  - [Custom application context](#custom-application-context)
- [Agent Skills](#agent-skills)
- [Subagent](#subagent)
- [Injected conversation](#injected-conversation)
- [Lifecycle listener](#lifecycle-listener)
- [Custom audio devices](#custom-audio-devices)
- [Echo cancellation](#echo-cancellation)
- [Turn detection](#turn-detection)
- [Voice and model](#voice-and-model)
- [Recording](#recording)
- [Token tracking](#token-tracking)
- [Inactivity timeout](#inactivity-timeout)
- [Stopping and interrupting](#stopping-and-interrupting)
- [Azure OpenAI](#azure-openai)

---

## Tool calling

### Basic tools

Create a `Tools` instance, decorate functions with `@tools.action(description)`, then pass the instance to `RealtimeAgent`. Both `async` and regular `def` functions are supported.

```python
import asyncio
from rtvoice import RealtimeAgent, Tools

tools = Tools()

@tools.action("Get the current weather for a given city")
async def get_weather(city: str) -> str:
    return f"It's 18°C and partly cloudy in {city}."

async def main():
    agent = RealtimeAgent(
        system_prompt="Answer weather questions using get_weather.",
        tools=tools,
    )
    await agent.run()

asyncio.run(main())
```

Parameter types are inferred from the function signature and included in the schema sent to the model. All parameters without a default value are marked required.

### Pydantic model tools

For richer schemas, register a Pydantic model with `param_model=`. The model fields become the tool parameters, and the function receives a validated model instance.

```python
from typing import Literal

from pydantic import BaseModel, Field
from rtvoice import Tools

tools = Tools()

class CalendarSearchParams(BaseModel):
    query: str = Field(description="What to search for")
    date: str | None = Field(default=None, description="Optional ISO date filter")
    limit: int = Field(default=5, description="Maximum number of matches")
    source: Literal["work", "personal"] = "work"

@tools.action(
    "Search calendar events",
    param_model=CalendarSearchParams,
)
async def search_calendar(params: CalendarSearchParams) -> str:
    return await calendar.search(
        query=params.query,
        date=params.date,
        limit=params.limit,
        source=params.source,
    )
```

Nested Pydantic models, typed lists, enums, literals, defaults, and `Field(description=...)` are included in the generated tool schema.

### Long-running tools

```python
@tools.action(
    "Search the web for a query",
)
async def search_web(query: str) -> str:
    result = await do_search(query)
    return result
```

Optionally add `result_instruction` to tell the model how to present the result once the tool returns:

```python
@tools.action(
    "Fetch the latest headlines",
    result_instruction="Summarise the headlines in two sentences.",
)
async def get_headlines() -> str: ...
```

Set `respond=False` when a successful action needs no assistant follow-up:

```python
@tools.action("Turn on the lights", respond=False)
async def turn_on_lights() -> None:
    await lights.turn_on()
```

An action can override that default per call. This lets the model choose through
a tool parameter whether a spoken response is useful:

```python
from rtvoice.tools import ActionResult

class LightParams(BaseModel):
    room: str
    respond: bool = Field(description="Whether to confirm this action aloud")

@tools.action("Turn on the lights", params=LightParams)
async def turn_on_lights(params: LightParams) -> ActionResult:
    await lights.turn_on(params.room)
    return ActionResult.success(respond=params.respond)
```

Tool output is still added to the conversation when no response is created.
Failures respond by default.

### Status templates

`status` is a spoken update for tools registered with `param_model=`. Use `{field_name}` placeholders from the Pydantic model — rtvoice validates them at registration time.

```python
class PlaySongParams(BaseModel):
    song: str = Field(description="Song title")

@tools.action(
    "Play a song by name",
    param_model=PlaySongParams,
    status="Playing {song} now.",
)
async def play_song(params: PlaySongParams) -> str:
    await music_player.play(params.song)
    return f"Now playing: {params.song}"
```

`status` can also be a callable that receives the validated Pydantic model and returns a string dynamically.

```python
@tools.action(
    "Play a song by name",
    param_model=PlaySongParams,
    status=lambda params: f"Playing {params.song} now.",
)
async def play_song(params: PlaySongParams) -> str:
    await music_player.play(params.song)
    return f"Now playing: {params.song}"
```

### Context injection

Any tool parameter typed as `Inject[T]` is filled automatically by the framework — the model never sees it and does not need to supply a value. Three types are injectable:

| Type                          | What it provides              |
| ----------------------------- | ----------------------------- |
| `Inject[EventBus]`            | Transit Bus event bus         |
| `Inject[ConversationHistory]` | Full conversation so far      |
| `Inject[YourContextType]`     | Your custom `tool_injection_context=` object |

```python
from rtvoice import Tools, Inject
from rtvoice.tools import ToolContext
from rtvoice.conversation import ConversationHistory

tools = Tools()

@tools.action("Summarise the conversation so far")
async def summarise(
    history: Inject[ConversationHistory],
) -> str:
    text = history.format()
    return await llm.summarise(text)
```

### Custom application context

Pass any object as `tool_injection_context=` on `RealtimeAgent`. It is then injectable in every tool via `Inject[YourType]`.

```python
from dataclasses import dataclass
from rtvoice import RealtimeAgent, Tools, Inject

@dataclass
class AppState:
    user_name: str
    premium: bool

tools = Tools()

@tools.action("Greet the user by name")
async def greet(state: Inject[AppState]) -> str:
    tier = "premium" if state.premium else "free"
    return f"Hello {state.user_name}, you are on the {tier} plan."

agent = RealtimeAgent(
    system_prompt="Greet the user when asked.",
    tools=tools,
    tool_injection_context=AppState(user_name="Alice", premium=True),
)
```

---

## Agent Skills

`rtvoice` supports the [Agent Skills](https://agentskills.io/specification) folder
format with progressive disclosure. A skills directory contains one subdirectory
per skill:

```text
skills/
└── internet-research/
    ├── SKILL.md
    ├── scripts/
    ├── references/
    └── assets/
```

`SKILL.md` starts with the standard YAML frontmatter:

```markdown
---
name: internet-research
description: Research current sources. Use when up-to-date facts are required.
---

# Internet Research

Follow the workflow in `references/workflow.md`.
```

Pass a `Skills` source to either `RealtimeAgent` or `Subagent`:

```python
from rtvoice import RealtimeAgent, Skills, Subagent

skills = Skills.from_local_dir("./skills")

subagent = Subagent(
    description="Handles research tasks.",
    system_prompt="Use the relevant skill before researching.",
    skills=skills,
)

agent = RealtimeAgent(
    system_prompt="Help the user and load relevant skills before using them.",
    skills=skills,
    subagent=subagent,
)
```

At startup, only each skill's `name` and `description` are added to the agent
prompt. Three tools are registered automatically as soon as at least one skill
is available:

| Tool                                     | Purpose                                                        |
| ---------------------------------------- | -------------------------------------------------------------- |
| `load_skill(name)`                       | Full instructions plus the list of the skill's bundled files   |
| `read_skill_resource(name, path)`        | Contents of one bundled file                                   |
| `run_skill_script(name, path, args)`     | Runs one bundled script in the skill's directory               |

Both `path` arguments are relative to the skill directory and cannot escape it.
Scripts run through `subprocess` with an explicit argument list — no shell, so
no command substitution or shell control flow. `.py` runs on the current
interpreter, `.sh` and `.bash` on `bash`; any other suffix must be directly
executable. Only skills you configure yourself are exposed, so treat a skill
directory as trusted code.

---

## Subagent

Delegate complex, multi-step tasks to an LLM-driven subagent. The voice agent
hands off the task and presents the result when done.

```python
from rtvoice import RealtimeAgent, Subagent, Tools
from rtvoice.llm import ChatOpenAI

tools = Tools()

@tools.action("Book a restaurant table")
async def book_table(
    restaurant: str,
    date: str,
    time: str,
    party_size: int,
) -> str:
    return f"Booked for {party_size} at {restaurant} on {date} at {time}."

subagent = Subagent(
    description="Books restaurant tables on behalf of the user.",
    system_prompt="Use book_table to complete booking requests.",
    tools=tools,
    llm=ChatOpenAI(model="gpt-4o-mini"),
)

agent = RealtimeAgent(
    system_prompt="Delegate restaurant bookings to the subagent.",
    subagent=subagent,
)
```

**How it works:** the realtime agent provides the `Subagent` through tool
dependency injection and registers it as a regular `subagent` tool. When
invoked, the subagent runs its own tool-calling loop and returns the final LLM
completion as the tool result.

**`Subagent` parameters:**

| Parameter              | Description                                               |
| ---------------------- | --------------------------------------------------------- |
| `description`          | Shown to the realtime model to decide when to delegate    |
| `system_prompt`        | System prompt for the subagent's own LLM loop             |
| `llm`                  | `ChatOpenAI(model=...)` or any `ChatModel` implementation |
| `tools`                | `Tools` instance with the actions the subagent may call   |
| `skills`               | Local Agent Skills exposed through progressive disclosure |
| `result_instructions`  | Tells the realtime model how to present the result        |
| `handoff_instructions` | Extra guidance appended to the tool description           |
| `max_iterations`       | Loop iteration cap (default: 10)                          |
| `tool_injection_context` | Arbitrary object injectable inside subagent tools        |

---

## Injected conversation

Pre-fill the session with synthetic conversation history before the microphone opens. The model will behave as if those exchanges already happened.

```python
from rtvoice import (
    RealtimeAgent,
    InjectedConversation,
    InjectedUserMessage,
    InjectedAssistantMessage,
)

agent = RealtimeAgent(
    system_prompt="You are a helpful assistant.",
    injected_conversation=InjectedConversation(
        messages=[
            InjectedUserMessage(text="My name is Alice and I prefer short answers."),
            InjectedAssistantMessage(text="Got it, Alice. I'll keep things brief."),
        ]
    ),
)
```

---

## Lifecycle listener

Subclass `AgentListener` and pass it to `RealtimeAgent` to hook into session events. Override only the methods you care about — all are async no-ops by default.

```python
from rtvoice import RealtimeAgent, AgentListener

class MyListener(AgentListener):
    async def on_agent_starting(self) -> None:
        print("Agent is starting up...")

    async def on_agent_session_connected(self) -> None:
        print("WebSocket connected, ready to talk.")

    async def on_user_transcript(self, transcript: str) -> None:
        print(f"User said: {transcript}")

    async def on_assistant_transcript(self, transcript: str) -> None:
        print(f"Assistant replied: {transcript}")

    async def on_agent_stopped(self) -> None:
        print("Session ended.")

agent = RealtimeAgent(
    system_prompt="You are a helpful assistant.",
    listener=MyListener(),
)
```

**All available callbacks:**

| Method                                            | When it fires                                                               |
| ------------------------------------------------- | --------------------------------------------------------------------------- |
| `on_agent_starting()`                             | Before any I/O or WebSocket setup                                           |
| `on_agent_session_connected()`                    | WebSocket session established                                               |
| `on_agent_stopped()`                              | Agent fully shut down                                                       |
| `on_user_started_speaking()`                      | VAD detected speech start                                                   |
| `on_user_stopped_speaking()`                      | VAD detected speech end                                                     |
| `on_user_transcript(transcript)`                  | Finalised user transcript (requires `transcription_model`)                  |
| `on_assistant_started_responding()`               | Assistant began streaming audio                                             |
| `on_assistant_stopped_responding()`               | Assistant finished streaming audio                                          |
| `on_assistant_transcript(transcript)`             | Full assistant response text                                                |
| `on_assistant_transcript_delta(delta)`            | Incremental assistant text chunk (requires `"text"` in `output_modalities`) |
| `on_agent_interrupted()`                          | User interrupted the assistant mid-response                                 |
| `on_agent_error(error)`                           | Session or API error                                                        |
| `on_user_inactivity_countdown(remaining_seconds)` | Fires each second before inactivity timeout                                 |
| `on_tool_executed(name, action_kind, silent)`     | Tool completed with its kind and effective silent state                      |

```python
from rtvoice import ActionKind, AgentListener

class MyListener(AgentListener):
    async def on_tool_executed(
        self, name: str, action_kind: ActionKind, silent: bool
    ) -> None:
        print(name, action_kind, silent)
```

---

## Custom audio devices

Implement `AudioInputDevice` or `AudioOutputDevice` from `rtvoice.audio` to replace the default microphone or speaker — useful for telephony, file playback, testing, or embedded hardware.

### Custom input

```python
from collections.abc import AsyncIterator
from rtvoice.audio import AudioInputDevice

class CustomMicrophone(AudioInputDevice):
    def __init__(self):
        self._active = False

    async def start(self) -> None:
        self._active = True
        # open your audio source here

    async def stop(self) -> None:
        self._active = False
        # release resources here

    async def stream_chunks(self) -> AsyncIterator[bytes]:
        while self._active:
            chunk = await self._read_pcm_chunk()  # raw 16-bit PCM, 24 kHz mono
            yield chunk

    @property
    def is_active(self) -> bool:
        return self._active

agent = RealtimeAgent(
    system_prompt="...",
    audio_input=CustomMicrophone(),
)
```

### Custom output

```python
from rtvoice.audio import AudioOutputDevice

class CustomSpeaker(AudioOutputDevice):
    def __init__(self):
        self._playing = False

    async def start(self) -> None:
        self._playing = True

    async def stop(self) -> None:
        self._playing = False

    async def play_chunk(self, chunk: bytes) -> None:
        # write raw 16-bit PCM audio to your sink
        await self._write_to_device(chunk)

    async def clear_buffer(self) -> None:
        # discard buffered audio (called on user interruption)
        await self._flush()

    @property
    def is_playing(self) -> bool:
        return self._playing

agent = RealtimeAgent(
    system_prompt="...",
    audio_output=CustomSpeaker(),
)
```

Audio format: **16-bit PCM, 24 kHz, mono** in both directions.

---

## Echo cancellation

When speaker and microphone are separate devices (e.g. a laptop's built-in
hardware without headphones), the microphone picks up the assistant's own
playback and the agent interrupts itself. Pass an `EchoCancellation` instance
to subtract the assistant's speech from the captured audio before it reaches
turn detection:

```python
from rtvoice import EchoCancellation, RealtimeAgent

agent = RealtimeAgent(
    system_prompt="...",
    echo_cancellation=EchoCancellation(),
)
```

This is a core feature of rtvoice: an adaptive NLMS filter learns the
loudspeaker-to-microphone path and removes it in real time, leaving barge-in
intact — unlike simple gating, the user can still interrupt the assistant
while it speaks. It works with any `AudioInputDevice`/`AudioOutputDevice`
pair, including custom ones.

---

## Turn detection

Control when the model decides the user has finished speaking.

### Semantic VAD (default)

Waits for a semantically complete thought. Less likely to cut off mid-sentence.

```python
from rtvoice import RealtimeAgent, SemanticVAD, SemanticEagerness

agent = RealtimeAgent(
    system_prompt="...",
    turn_detection=SemanticVAD(eagerness=SemanticEagerness.LOW),
)
```

`SemanticEagerness` values: `LOW`, `MEDIUM`, `HIGH`, `AUTO` (default).

### Server VAD

Energy-based: triggers on silence duration. More predictable latency.

```python
from rtvoice import RealtimeAgent, ServerVAD

agent = RealtimeAgent(
    system_prompt="...",
    turn_detection=ServerVAD(
        threshold=0.5,           # energy threshold 0–1
        prefix_padding_ms=300,   # audio kept before speech onset
        silence_duration_ms=500, # silence needed to commit end-of-turn
    ),
)
```

---

## Voice and model

```python
from rtvoice import AssistantVoice, RealtimeAgent, RealtimeModel, ReasoningEffort

agent = RealtimeAgent(
    model=RealtimeModel.GPT_REALTIME_2_1_MINI,   # default; or GPT_REALTIME_2_1, GPT_REALTIME_2
    reasoning_effort=ReasoningEffort.LOW,        # default for gpt-realtime-2.1-mini
    voice=AssistantVoice.CORAL,
    speech_speed=1.2,                            # 0.25–1.5, default 1.0
    system_prompt="...",
)
```

`gpt-realtime-2.1-mini` is the default model. It supports reasoning controls; start with `ReasoningEffort.LOW` for most voice agents and increase only for workflows that need deeper planning. Set `reasoning_effort=None` if you need to omit the `reasoning` session setting.

`GPT_REALTIME` and `GPT_REALTIME_MINI` remain available for compatibility but
emit `DeprecationWarning`; migrate to `GPT_REALTIME_2_1` and
`GPT_REALTIME_2_1_MINI`, respectively.

Available voices: `ALLOY`, `ASH`, `BALLAD`, `CORAL`, `ECHO`, `FABLE`, `ONYX`, `NOVA`, `SAGE`, `SHIMMER`, `VERSE`, `CEDAR`, `MARIN`.

---

## Recording

Save the raw session audio to a file:

```python
agent = RealtimeAgent(
    system_prompt="...",
    recording_path="session.pcm",
)

result = await agent.run()
print(result.recording_path)   # Path to the saved file
```

The returned `AgentResult` also contains `result.turns` — a list of `ConversationTurn` objects with role and text for every exchange.

---

## Token usage and cost estimate

Every `response.done` and input-transcription completion event is tracked for
the lifetime of an agent. The final result contains modality-level token totals,
cached input tokens, and an estimated USD cost:

```python
result = await agent.start()

print(result.usage.tokens.realtime.input_audio_tokens)
print(result.usage.tokens.realtime.output_audio_tokens)
print(result.usage.tokens.realtime.cached_input_tokens)
print(result.usage.cost.total, result.usage.cost.currency)
```

The built-in USD prices come from Tokenary's generated catalog and update with
the installed Tokenary version. `agent.usage_report()` returns the current
report while a session is running. Currency conversion is deliberately left to
the caller because exchange rates fluctuate.

Pass a custom `PricingCatalog` to `RealtimeAgent` when using provider-specific
or negotiated rates.

`estimate.cost.is_complete` is false when an event lacks the modality details
needed for exact pricing. Whisper is billed per minute; if the event only
contains audio tokens, its duration is estimated at 100 ms per audio token and
recorded in `estimate.cost.notes`.

---

## Inactivity timeout

Automatically stop the agent after a period of user silence:

```python
agent = RealtimeAgent(
    system_prompt="...",
    inactivity_timeout_seconds=30.0,
    listener=MyListener(),   # on_user_inactivity_countdown fires each second 5→1
)
```

The countdown fires through `AgentListener.on_user_inactivity_countdown(remaining_seconds)` — useful for playing a "still there?" prompt before the session closes.

---

## Stopping and interrupting

Every agent gets a built-in `stop` tool, so the model can end the conversation
itself when the user says goodbye. The teardown waits until the assistant's
farewell finished playing.

Both actions are also available programmatically:

```python
await agent.interrupt()  # cut the assistant off mid-sentence, session stays open
await agent.stop()       # end the session
```

`interrupt()` runs the same path as a user barge-in: the in-flight response is
cancelled, buffered audio is dropped, and the conversation item is truncated to
what was actually played, so the model knows how much the user heard.
`AgentListener.on_agent_interrupted` fires either way.

---

## Azure OpenAI

Pass an `AzureOpenAIProvider` instead of the default OpenAI provider:

```python
from rtvoice import RealtimeAgent
from rtvoice import AzureOpenAIProvider

agent = RealtimeAgent(
    system_prompt="...",
    provider=AzureOpenAIProvider(
        azure_endpoint="https://your-resource.openai.azure.com",
        azure_deployment="gpt-4o-realtime-preview",
        api_version="2024-12-17",
        api_key="...",          # or omit to use AZURE_OPENAI_API_KEY
    ),
)
```
