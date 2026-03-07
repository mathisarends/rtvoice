# rtvoice

**rtvoice** is a Python library for building real-time voice agents powered by the [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime). It handles the full session lifecycle — microphone input, WebSocket streaming, turn detection, tool calling, and audio playback — so you can focus on what your agent does, not how it talks.

```python
import asyncio
from rtvoice import RealtimeAgent

async def main():
    agent = RealtimeAgent(instructions="You are Jarvis, a helpful assistant.")
    await agent.run()

asyncio.run(main())
```

---

## Features

- **One-class API** — `RealtimeAgent` manages the full voice loop out of the box
- **Tool calling** — register async functions with `@tools.action(...)` in seconds
- **Supervisor agents** — delegate complex tasks to an LLM-driven sub-agent with automatic handoff
- **MCP integration** — connect any Model Context Protocol server via `MCPServerStdio`
- **Listener hooks** — receive transcripts, speaking state, and errors through `AgentListener`
- **VAD options** — semantic (default) or energy-based voice-activity detection
- **Inactivity timeout** — automatically stop the session after a configurable silence window
- **Session recording** — optionally save the full audio session to disk

---

## Installation

```bash
pip install rtvoice
```

For microphone and speaker support (requires [PortAudio](https://www.portaudio.com/)):

```bash
pip install rtvoice[audio]
```

Set your OpenAI API key before running:

```bash
export OPENAI_API_KEY=sk-...
```

Or pass it directly:

```python
agent = RealtimeAgent(api_key="sk-...")
```

---

## Next steps

- [Quickstart](quickstart.md) — a minimal working agent in 10 lines
- [Tools guide](guides/tools.md) — register functions the model can call
- [Supervisor guide](guides/supervisor.md) — delegate complex tasks to a sub-agent
- [MCP guide](guides/mcp.md) — connect external tool servers
- [Listener guide](guides/listener.md) — hook into session events for UI integration
