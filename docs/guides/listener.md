# Listener

`AgentListener` is a callback interface that lets you react to session lifecycle events — transcript completions, speaking state changes, errors, and session boundaries. It is the primary integration point for building UIs, logging pipelines, or any code that needs to observe what the agent is doing.

---

## Overview

Subclass `AgentListener`, override the methods you need, and pass an instance to `RealtimeAgent`:

```python
from rtvoice import RealtimeAgent, AgentListener

class MyListener(AgentListener):
    async def on_user_transcript(self, transcript: str) -> None:
        print(f"User: {transcript}")

    async def on_assistant_transcript(self, transcript: str) -> None:
        print(f"Assistant: {transcript}")

agent = RealtimeAgent(
    instructions="...",
    listener=MyListener(),
)
await agent.run()
```

All methods are async no-ops by default — override only the ones you care about.

---

## Available callbacks

### Session lifecycle

| Method                         | When it fires                                                 |
| ------------------------------ | ------------------------------------------------------------- |
| `on_agent_session_connected()` | The WebSocket session is open and ready.                      |
| `on_agent_stopped()`           | The agent has fully shut down and `run()` is about to return. |

### Speaking state

| Method                              | When it fires                                    |
| ----------------------------------- | ------------------------------------------------ |
| `on_user_started_speaking()`        | VAD detects that the user has started speaking.  |
| `on_user_stopped_speaking()`        | VAD detects that the user has stopped speaking.  |
| `on_assistant_started_responding()` | The assistant begins streaming audio.            |
| `on_assistant_stopped_responding()` | The assistant finishes streaming audio.          |
| `on_agent_interrupted()`            | The user interrupted the assistant mid-response. |

### Transcripts

| Method                                | When it fires                                                           |
| ------------------------------------- | ----------------------------------------------------------------------- |
| `on_user_transcript(transcript)`      | The user's turn is fully transcribed. Requires a `transcription_model`. |
| `on_assistant_transcript(transcript)` | The assistant's response transcript is complete.                        |

### Errors

| Method                  | When it fires                                                       |
| ----------------------- | ------------------------------------------------------------------- |
| `on_agent_error(error)` | An error was received from the Realtime API or the agent internals. |

---

## Logging transcripts to the console

```python
from rtvoice import AgentListener

class ConsolePrinter(AgentListener):
    async def on_user_transcript(self, transcript: str) -> None:
        print(f"\033[36mYou: {transcript}\033[0m")

    async def on_assistant_transcript(self, transcript: str) -> None:
        print(f"Jarvis: {transcript}")
```

---

## Tracking speaking state for UI

Use speaking state callbacks to drive UI indicators (e.g. a mic activity light or a "thinking" spinner):

```python
from rtvoice import AgentListener

class UIBridge(AgentListener):
    async def on_user_started_speaking(self) -> None:
        ui.set_mic_active(True)

    async def on_user_stopped_speaking(self) -> None:
        ui.set_mic_active(False)

    async def on_assistant_started_responding(self) -> None:
        ui.show_spinner(True)

    async def on_assistant_stopped_responding(self) -> None:
        ui.show_spinner(False)

    async def on_agent_interrupted(self) -> None:
        ui.show_spinner(False)
```

---

## Handling errors

```python
from rtvoice import AgentListener
from rtvoice.views import AgentError

class ErrorHandler(AgentListener):
    async def on_agent_error(self, error: AgentError) -> None:
        print(f"[{error.type}] {error.message}")
        # e.g. log to Sentry, display an error banner, etc.
```

`AgentError` has two fields:

- `type` — OpenAI error type identifier (e.g. `invalid_request_error`)
- `message` — Human-readable description

---

## Reacting to session end

```python
class SessionLogger(AgentListener):
    async def on_agent_session_connected(self) -> None:
        self._start = time.time()

    async def on_agent_stopped(self) -> None:
        duration = time.time() - self._start
        print(f"Session ended after {duration:.0f}s")
```

---

## Full example

```python
import asyncio
from rtvoice import RealtimeAgent, AgentListener
from rtvoice.views import AgentError

class VerboseListener(AgentListener):
    async def on_agent_session_connected(self) -> None:
        print("[connected]")

    async def on_user_started_speaking(self) -> None:
        print("[user speaking…]")

    async def on_user_transcript(self, transcript: str) -> None:
        print(f"You: {transcript}")

    async def on_assistant_started_responding(self) -> None:
        print("[assistant responding…]")

    async def on_assistant_transcript(self, transcript: str) -> None:
        print(f"Assistant: {transcript}")

    async def on_agent_interrupted(self) -> None:
        print("[interrupted]")

    async def on_agent_error(self, error: AgentError) -> None:
        print(f"Error: {error}")

    async def on_agent_stopped(self) -> None:
        print("[session ended]")


async def main():
    agent = RealtimeAgent(
        instructions="You are a concise voice assistant.",
        listener=VerboseListener(),
    )
    await agent.run()

asyncio.run(main())
```

---

## API reference

See [`AgentListener`](../api/lifecycle_events.md) and [`AgentError`](../api/lifecycle_events.md) for full method signatures.
