# rtvoice

[![Documentation](https://img.shields.io/badge/docs-rtvoice-blue?style=flat&logo=readthedocs)](https://mathisarends.github.io/rtvoice/)
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

## Tool calling

Register any async (or sync) function with `@tools.action(...)` and the model will call it when appropriate:

```python
import asyncio
from typing import Annotated
from rtvoice import RealtimeAgent, Tools

tools = Tools()

@tools.action("Get the current weather for a given city")
async def get_weather(city: Annotated[str, "The city name"]) -> str:
    return f"It's 18°C and partly cloudy in {city}."

async def main():
    agent = RealtimeAgent(
        instructions="Answer weather questions using get_weather.",
        tools=tools,
    )
    await agent.run()

asyncio.run(main())
```

For long-running tools, set `is_long_running=True` and provide a `holding_instruction` so the assistant keeps the user informed while it works. → [Tools guide](https://mathisarends.github.io/rtvoice/guides/tools/)

---

## Subagents

Delegate complex, multi-step tasks to a dedicated LLM-driven sub-agent. The voice agent hands off the task, speaks a holding phrase, and presents the result when done:

```python
from llmify import ChatOpenAI
from rtvoice import RealtimeAgent, SubAgent, Tools

tools = Tools()

@tools.action("Book a restaurant table.")
async def book_table(restaurant: str, date: str, time: str, party_size: int) -> str:
    return f"Booked table for {party_size} at {restaurant} on {date} at {time}."

booking_agent = SubAgent(
    name="Booking Assistant",
    description="Books restaurant tables for the user.",
    holding_instruction="I'm checking availability, just a moment.",
    instructions="Use book_table to complete booking requests.",
    tools=tools,
    llm=ChatOpenAI(model="gpt-4o-mini"),
)

agent = RealtimeAgent(
    instructions="Delegate restaurant bookings to the Booking Assistant.",
    subagents=[booking_agent],
)
```

If a subagent needs information from the user (e.g. party size), it asks a clarifying question through the voice agent automatically. → [Subagents guide](https://mathisarends.github.io/rtvoice/guides/subagent/)

---

## MCP servers

Connect any MCP-compatible tool server via `MCPServerStdio`. Tools are discovered and registered automatically during `prepare()`:

```python
from rtvoice import RealtimeAgent
from rtvoice.mcp import MCPServerStdio

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

Prefer attaching MCP servers to a `SubAgent` rather than `RealtimeAgent` directly to keep the realtime model's tool list short. → [MCP guide](https://mathisarends.github.io/rtvoice/guides/mcp/)

---

## Custom audio devices

Implement `AudioInputDevice` or `AudioOutputDevice` to use any audio source or sink — useful for testing, telephony, or embedded hardware:

```python
from collections.abc import AsyncIterator
from rtvoice.audio import AudioInputDevice

class CustomMicrophone(AudioInputDevice):
    async def start(self) -> None: ...
    async def stop(self) -> None: ...

    async def stream_chunks(self) -> AsyncIterator[bytes]:
        while self.is_active:
            yield await self._read_audio_chunk()

    @property
    def is_active(self) -> bool:
        return self._active

agent = RealtimeAgent(
    instructions="...",
    audio_input=CustomMicrophone(),
)
```

→ [Audio API reference](https://mathisarends.github.io/rtvoice/api/audio/)

---

## Documentation

Full documentation including guides and API reference: **[mathisarends.github.io/rtvoice](https://mathisarends.github.io/rtvoice/)**

- [Quickstart](https://mathisarends.github.io/rtvoice/quickstart/)
- [Tools](https://mathisarends.github.io/rtvoice/guides/tools/)
- [Subagents](https://mathisarends.github.io/rtvoice/guides/subagent/)
- [MCP Servers](https://mathisarends.github.io/rtvoice/guides/mcp/)
- [Listener](https://mathisarends.github.io/rtvoice/guides/listener/)
- [API Reference](https://mathisarends.github.io/rtvoice/api/agent/)
