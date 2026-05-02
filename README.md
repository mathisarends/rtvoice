# rtvoice

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
        instructions="You are Jarvis, a concise and helpful voice assistant.",
    )
    await agent.run()

asyncio.run(main())
```

Run it, speak into your microphone, and the agent responds through your speakers. Press `Ctrl+C` to end the session.

---

## Table of Contents

- [Tool calling](#tool-calling)
  - [Basic tools](#basic-tools)
  - [Long-running tools](#long-running-tools)
  - [Status templates](#status-templates)
  - [Tool steering](#tool-steering)
  - [Context injection](#context-injection)
  - [Custom application context](#custom-application-context)
- [Subagents](#subagents)
- [MCP servers](#mcp-servers)
- [Conversation seeds](#conversation-seeds)
- [Lifecycle listener](#lifecycle-listener)
- [Custom audio devices](#custom-audio-devices)
- [Turn detection](#turn-detection)
- [Voice and model](#voice-and-model)
- [Recording](#recording)
- [Inactivity timeout](#inactivity-timeout)
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
        instructions="Answer weather questions using get_weather.",
        tools=tools,
    )
    await agent.run()

asyncio.run(main())
```

Parameter types are inferred from the function signature and included in the schema sent to the model. All parameters without a default value are marked required. For richer per-parameter descriptions, pass a Pydantic model via `param_model=` on `@tools.action(...)` and use `Field(description=...)` on its fields.

### Long-running tools

Set `holding_instruction` to have the assistant speak a phrase while the tool runs. The agent will say it immediately after calling the tool, before the result arrives.

```python
@tools.action(
    "Search the web for a query",
    holding_instruction="Let me search that for you, give me a moment.",
)
async def search_web(query: str) -> str:
    result = await do_search(query)
    return result
```

Optionally add `result_instruction` to tell the model how to present the result once the tool returns:

```python
@tools.action(
    "Fetch the latest headlines",
    holding_instruction="Fetching the news...",
    result_instruction="Summarise the headlines in two sentences.",
)
async def get_headlines() -> str: ...
```

### Status templates

`status` is a spoken update that interpolates tool arguments. Use `{param_name}` placeholders — rtvoice validates them at registration time.

```python
@tools.action(
    "Play a song by name",
    status="Playing {song} now.",
)
async def play_song(song: str) -> str:
    await music_player.play(song)
    return f"Now playing: {song}"
```

`status` can also be an `async def` that receives the same arguments and returns a string dynamically.

### Tool steering

`steering` appends hidden guidance to the tool result without exposing it to the user transcript. Useful for nudging the model without cluttering the response.

```python
@tools.action(
    "Look up a contact",
    steering="If the contact has a preferred name, use it in your response.",
)
async def lookup_contact(name: str) -> str: ...
```

### Context injection

Any tool parameter typed as `Inject[T]` is filled automatically by the framework — the model never sees it and does not need to supply a value. Three types are injectable:

| Type | What it provides |
|---|---|
| `Inject[EventBus]` | Internal event bus |
| `Inject[ConversationHistory]` | Full conversation so far |
| `Inject[YourContextType]` | Your custom `context=` object |

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

Pass any object as `context=` on `RealtimeAgent`. It is then injectable in every tool via `Inject[YourType]`.

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
    instructions="Greet the user when asked.",
    tools=tools,
    context=AppState(user_name="Alice", premium=True),
)
```

---

## Subagents

Delegate complex, multi-step tasks to a dedicated LLM-driven subagent. The voice agent hands off, speaks a holding phrase, and presents the result when done.

```python
from rtvoice import RealtimeAgent, SubAgent, Tools
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

booking_agent = SubAgent(
    name="Booking Assistant",
    description="Books restaurant tables on behalf of the user.",
    holding_instruction="I'm checking availability, just a moment.",
    instructions="Use book_table to complete booking requests. Call done() when finished.",
    tools=tools,
    llm=ChatOpenAI(model="gpt-4o-mini"),
)

agent = RealtimeAgent(
    instructions="Delegate restaurant bookings to the Booking Assistant.",
    subagents=[booking_agent],
)
```

**How it works:** the realtime agent registers each `SubAgent` as a callable tool. When invoked, the subagent runs its own agentic loop (tool calls → LLM → tool calls …) until it either calls `done()` or needs a clarification from the user via `clarify()`. Clarifications are automatically routed back through the voice agent and the loop resumes.

**`SubAgent` parameters:**

| Parameter | Description |
|---|---|
| `name` | Unique name; becomes the tool name the realtime model calls |
| `description` | Shown to the realtime model to decide when to delegate |
| `instructions` | System prompt for the subagent's own LLM loop |
| `llm` | `ChatOpenAI(model=...)` or any `ChatModel` implementation |
| `tools` | `Tools` instance with the actions the subagent may call |
| `mcp_servers` | MCP servers to connect to during prewarm |
| `holding_instruction` | Spoken while the subagent works |
| `result_instructions` | Tells the realtime model how to present the result |
| `handoff_instructions` | Extra guidance appended to the tool description |
| `max_iterations` | Loop iteration cap (default: 10) |
| `context` | Arbitrary object injectable inside subagent tools |

---

## MCP servers

Connect any MCP-compatible tool server via `MCPServerStdio`. Tools are discovered automatically during startup.

```python
from rtvoice import RealtimeAgent
from rtvoice import MCPServerStdio

agent = RealtimeAgent(
    instructions="You can read and write files in /tmp.",
    mcp_servers=[
        MCPServerStdio(
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
        )
    ],
)
```

For heavy tool sets, attach the MCP server to a `SubAgent` instead. This keeps the realtime model's tool list short and avoids latency on every turn:

```python
research_agent = SubAgent(
    name="Researcher",
    description="Searches the web and reads URLs.",
    instructions="Use the available tools to answer research questions.",
    llm=ChatOpenAI(model="gpt-4o"),
    mcp_servers=[
        MCPServerStdio(command="uvx", args=["mcp-server-fetch"]),
    ],
)

agent = RealtimeAgent(
    instructions="Delegate research tasks to the Researcher.",
    subagents=[research_agent],
)
```

---

## Conversation seeds

Pre-fill the session with synthetic conversation history before the microphone opens. The model will behave as if those exchanges already happened.

```python
from rtvoice import RealtimeAgent, ConversationSeed, SeedMessage

agent = RealtimeAgent(
    instructions="You are a helpful assistant.",
    conversation_seed=ConversationSeed(
        messages=[
            SeedMessage.user("My name is Alice and I prefer short answers."),
            SeedMessage.assistant("Got it, Alice. I'll keep things brief."),
        ]
    ),
)
```

Use `ConversationSeed.from_pairs()` for a more concise form when you have multiple user/assistant exchanges:

```python
seed = ConversationSeed.from_pairs(
    ("My name is Alice.", "Nice to meet you, Alice."),
    ("I prefer short answers.", "Understood, I'll be brief."),
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
    instructions="You are a helpful assistant.",
    listener=MyListener(),
)
```

**All available callbacks:**

| Method | When it fires |
|---|---|
| `on_agent_starting()` | Before any I/O or WebSocket setup |
| `on_agent_session_connected()` | WebSocket session established |
| `on_agent_stopped()` | Agent fully shut down |
| `on_user_started_speaking()` | VAD detected speech start |
| `on_user_stopped_speaking()` | VAD detected speech end |
| `on_user_transcript(transcript)` | Finalised user transcript (requires `transcription_model`) |
| `on_assistant_started_responding()` | Assistant began streaming audio |
| `on_assistant_stopped_responding()` | Assistant finished streaming audio |
| `on_assistant_transcript(transcript)` | Full assistant response text |
| `on_assistant_transcript_delta(delta)` | Incremental assistant text chunk (requires `"text"` in `output_modalities`) |
| `on_agent_interrupted()` | User interrupted the assistant mid-response |
| `on_agent_error(error)` | Session or API error |
| `on_subagent_started(agent_name)` | A subagent began running |
| `on_subagent_finished(agent_name)` | A subagent finished |
| `on_user_inactivity_countdown(remaining_seconds)` | Fires each second before inactivity timeout |

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
    instructions="...",
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
    instructions="...",
    audio_output=CustomSpeaker(),
)
```

Audio format: **16-bit PCM, 24 kHz, mono** in both directions.

---

## Turn detection

Control when the model decides the user has finished speaking.

### Semantic VAD (default)

Waits for a semantically complete thought. Less likely to cut off mid-sentence.

```python
from rtvoice import RealtimeAgent, SemanticVAD, SemanticEagerness

agent = RealtimeAgent(
    instructions="...",
    turn_detection=SemanticVAD(eagerness=SemanticEagerness.LOW),
)
```

`SemanticEagerness` values: `LOW`, `MEDIUM`, `HIGH`, `AUTO` (default).

### Server VAD

Energy-based: triggers on silence duration. More predictable latency.

```python
from rtvoice import RealtimeAgent, ServerVAD

agent = RealtimeAgent(
    instructions="...",
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
from rtvoice import RealtimeAgent, AssistantVoice, RealtimeModel

agent = RealtimeAgent(
    model=RealtimeModel.GPT_REALTIME,       # or GPT_REALTIME_MINI, GPT_REALTIME_1_5
    voice=AssistantVoice.CORAL,
    speech_speed=1.2,                       # 0.25–1.5, default 1.0
    instructions="...",
)
```

Available voices: `ALLOY`, `ASH`, `BALLAD`, `CORAL`, `ECHO`, `FABLE`, `ONYX`, `NOVA`, `SAGE`, `SHIMMER`, `VERSE`, `CEDAR`, `MARIN`.

---

## Recording

Save the raw session audio to a file:

```python
agent = RealtimeAgent(
    instructions="...",
    recording_path="session.pcm",
)

result = await agent.run()
print(result.recording_path)   # Path to the saved file
```

The returned `AgentResult` also contains `result.turns` — a list of `ConversationTurn` objects with role and text for every exchange.

---

## Inactivity timeout

Automatically stop the agent after a period of user silence:

```python
agent = RealtimeAgent(
    instructions="...",
    inactivity_timeout_enabled=True,
    inactivity_timeout_seconds=30.0,
    listener=MyListener(),   # on_user_inactivity_countdown fires each second 5→1
)
```

The countdown fires through `AgentListener.on_user_inactivity_countdown(remaining_seconds)` — useful for playing a "still there?" prompt before the session closes.

---

## Azure OpenAI

Pass an `AzureOpenAIProvider` instead of the default OpenAI provider:

```python
from rtvoice import RealtimeAgent
from rtvoice import AzureOpenAIProvider

agent = RealtimeAgent(
    instructions="...",
    provider=AzureOpenAIProvider(
        azure_endpoint="https://your-resource.openai.azure.com",
        azure_deployment="gpt-4o-realtime-preview",
        api_version="2024-12-17",
        api_key="...",          # or omit to use AZURE_OPENAI_API_KEY
    ),
)
```
