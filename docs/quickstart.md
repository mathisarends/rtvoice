# Quickstart

Get a working voice agent running in under a minute.

## Prerequisites

- Python 3.13+
- `OPENAI_API_KEY` set in your environment
- PortAudio installed (required for `pyaudio`)

```bash
pip install rtvoice[audio]
```

---

## Minimal example

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

Run it, speak into your microphone, and the agent will respond through your speakers.
Press `Ctrl+C` to stop the session.

---

## What happens when you call `run()`

1. `prepare()` is called automatically — MCP servers connect, supervisor warms up.
2. A WebSocket session opens to the OpenAI Realtime API.
3. The microphone starts streaming audio to the API.
4. The API detects when you finish speaking (semantic VAD by default) and generates a response.
5. Audio is streamed back and played through the speaker in real time.
6. The session runs until you call `stop()`, an inactivity timeout fires, or the process is interrupted.

---

## Adding your first tool

```python
import asyncio
from typing import Annotated
from rtvoice import RealtimeAgent, Tools

tools = Tools()

@tools.action("Get the current time in a given city")
async def get_time(city: Annotated[str, "The city name"]) -> str:
    return f"It's 14:32 in {city}."  # replace with real logic

async def main():
    agent = RealtimeAgent(
        instructions="You are a helpful assistant. Answer time questions with get_time.",
        tools=tools,
    )
    await agent.run()

asyncio.run(main())
```

See the [Tools guide](guides/tools.md) for the full decorator API including long-running tools and auto-injected parameters.

---

## Printing transcripts

```python
import asyncio
from rtvoice import RealtimeAgent, AgentListener

class ConsolePrinter(AgentListener):
    async def on_user_transcript(self, transcript: str) -> None:
        print(f"You: {transcript}")

    async def on_assistant_transcript(self, transcript: str) -> None:
        print(f"Assistant: {transcript}")

async def main():
    agent = RealtimeAgent(
        instructions="You are a concise voice assistant.",
        listener=ConsolePrinter(),
    )
    await agent.run()

asyncio.run(main())
```

See the [Listener guide](guides/listener.md) for all available callbacks.

---

## Auto-stop after silence

```python
agent = RealtimeAgent(
    instructions="...",
    inactivity_timeout_enabled=True,
    inactivity_timeout_seconds=30,
)
```

The session ends automatically after 30 seconds without the user speaking. Useful for kiosk or embedded applications.

---

## Next steps

- [Tools](guides/tools.md) — register functions the model can call
- [Supervisor Agent](guides/supervisor.md) — delegate complex tasks to an LLM-driven sub-agent
- [MCP Servers](guides/mcp.md) — connect stdio-based tool servers
- [Listener](guides/listener.md) — react to session lifecycle events
- [API Reference](api/agent.md) — full parameter list for `RealtimeAgent`
