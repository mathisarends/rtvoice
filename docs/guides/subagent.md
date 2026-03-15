# Subagents

A `SubAgent` is an LLM-driven helper that handles complex, multi-step tasks on behalf of the voice agent. When the user asks something that requires tool calling, research, or a series of decisions, the voice agent hands the task off to a subagent and speaks a holding phrase while it works.

---

## When to use a subagent

Use a subagent when a task:

- Requires multiple tool calls in sequence (e.g. look up a contact, then book a meeting)
- Involves decision-making that shouldn't block the voice session
- Belongs to a distinct domain (calendar, weather, home automation, …)
- Is better served by a more capable LLM than the realtime model

---

## Creating a subagent

```python
from llmify import ChatOpenAI
from rtvoice import SubAgent, Tools
from typing import Annotated

tools = Tools()

@tools.action("Book a restaurant table for the given date, time and party size.")
async def book_table(
    restaurant: Annotated[str, "Restaurant name"],
    date: Annotated[str, "Date in YYYY-MM-DD format"],
    time: Annotated[str, "Time in HH:MM (24 h)"],
    party_size: Annotated[int, "Number of guests"],
) -> str:
    return f"Booked table for {party_size} at {restaurant} on {date} at {time}."


booking_agent = SubAgent(
    name="Booking Assistant",
    description="Books restaurant tables for the user.",
    instructions=(
        "You are a restaurant booking assistant. "
        "Use the book_table tool to complete booking requests. "
        "Confirm details before booking."
    ),
    tools=tools,
    llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.2),
)
```

---

## Attaching to the voice agent

Pass subagents via `subagents=[...]`. `RealtimeAgent` automatically registers a handoff tool for each subagent that the realtime model can call:

```python
from rtvoice import RealtimeAgent

agent = RealtimeAgent(
    instructions=(
        "You are a friendly assistant. "
        "Delegate restaurant bookings to the Booking Assistant."
    ),
    subagents=[booking_agent],
)
await agent.run()
```

The tool name the realtime model sees is derived from the subagent's `name` (spaces replaced with underscores): `Booking_Assistant`.

---

## Handoff parameters

These parameters on `SubAgent` shape how the handoff appears to the realtime model:

| Parameter              | Description                                                                            |
| ---------------------- | -------------------------------------------------------------------------------------- |
| `description`          | Tells the realtime model what this agent does and when to delegate.                    |
| `handoff_instructions` | Extra guidance appended to the tool description (e.g. "always include the city name"). |
| `holding_instruction`  | Phrase the realtime agent speaks while the subagent is working.                        |
| `result_instructions`  | How the realtime agent should present the subagent's result to the user.               |

```python
booking_agent = SubAgent(
    name="Booking Assistant",
    description="Books restaurant tables.",
    handoff_instructions="Always include restaurant name, date, time, and party size.",
    holding_instruction="I'm checking availability, just a moment.",
    result_instructions="Confirm the booking details warmly and ask if there's anything else.",
    instructions="...",
    llm=ChatOpenAI(model="gpt-4o-mini"),
)
```

---

## Clarification questions

If the subagent cannot proceed without missing information, it calls the built-in `clarify` tool. The realtime agent asks the user the question out loud, waits for a voice reply, and feeds the answer back so the subagent can continue.

```python
# The subagent will call clarify() automatically if party_size is missing.
# No extra code required on your side.
```

!!! note
Transcription must be enabled (the default) for clarification to work — the user's spoken reply is transcribed and returned to the subagent.

---

## Max iterations

The subagent runs a tool-calling loop and stops when it calls `done`, when the LLM responds without a tool call, or after `max_iterations` steps (default: `10`). Set this lower for simpler agents or higher for complex multi-step workflows:

```python
subagent = SubAgent(
    ...,
    max_iterations=5,
)
```

---

## Prewarming

Call `prepare()` on the agent (which also prewarms subagents) to connect MCP servers and warm up the LLM before the session starts:

```python
await agent.prepare()
await agent.run()
```

Or chain it:

```python
result = await (await agent.prepare()).run()
```

---

## Full example

```python
import asyncio
from typing import Annotated
from llmify import ChatOpenAI
from rtvoice import RealtimeAgent, SubAgent, Tools

tools = Tools()

@tools.action("Fetch the current weather for a given city.")
def get_weather(city: Annotated[str, "City name"]) -> str:
    return f"12°C and cloudy in {city}."


async def main():
    weather_agent = SubAgent(
        name="Weather Assistant",
        description="Looks up current weather conditions for any city.",
        handoff_instructions="Always include the city name in the task.",
        holding_instruction="Let me check the weather for you.",
        instructions="Use get_weather to answer weather questions. Be concise.",
        tools=tools,
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.2),
    )

    agent = RealtimeAgent(
        instructions=(
            "You are a friendly assistant. "
            "Delegate weather questions to the Weather Assistant."
        ),
        subagents=[weather_agent],
    )
    await agent.run()

asyncio.run(main())
```

---

## API reference

See [`SubAgent`](../api/subagent.md) for the complete parameter list.
