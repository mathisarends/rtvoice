# Tools

Tools let the model call your Python functions during a voice session. Register an async (or sync) function with `@tools.action(...)`, pass the `Tools` instance to `RealtimeAgent`, and the model will invoke it when appropriate.

---

## Basic registration

```python
from typing import Annotated
from rtvoice import Tools

tools = Tools()

@tools.action("Get the current weather for a given city")
async def get_weather(city: Annotated[str, "The city name"]) -> str:
    return f"It is sunny and 22°C in {city}."
```

Pass the description as the first argument — write it as a natural-language instruction so the model knows when to call the tool.

Parameter descriptions are read from `Annotated` type hints and included in the schema sent to the model.

---

## Passing tools to the agent

```python
from rtvoice import RealtimeAgent

agent = RealtimeAgent(
    instructions="Answer weather questions using get_weather.",
    tools=tools,
)
await agent.run()
```

---

## Decorator options

```python
@tools.action(
    "Search the knowledge base for relevant articles",
    name="search_kb",                          # override the tool name (default: function name)
    result_instruction="Summarise the top result in one sentence.",
    is_long_running=True,                      # enables a holding phrase while the tool runs
    holding_instruction="Let me search for that…",
)
async def search_knowledge_base(query: str) -> str: ...
```

| Parameter             | Description                                                                                         |
| --------------------- | --------------------------------------------------------------------------------------------------- |
| `description`         | Natural-language instruction shown to the model (required).                                         |
| `name`                | Override the tool name. Defaults to the function name.                                              |
| `result_instruction`  | Appended to the tool result to guide how the model presents it.                                     |
| `is_long_running`     | Set `True` for tools that take more than ~1 s. The assistant speaks a holding phrase while waiting. |
| `holding_instruction` | What the assistant says while the tool is running. Requires `is_long_running=True`.                 |

---

## Auto-injected parameters

Use `Inject[T]` to mark parameters that should be injected from the `ToolContext` at runtime. These are excluded from the model-facing schema automatically.

```python
from rtvoice import Inject, Tools
from rtvoice.events.bus import EventBus

tools = Tools()

@tools.action("Emit a custom event")
async def emit(event_bus: Inject[EventBus]) -> str:
    # event_bus is injected, not exposed to the model
    return "done"
```

Available injectable types:

| Type                  | Injected value                                       |
| --------------------- | ---------------------------------------------------- |
| `EventBus`            | The session-scoped event bus                         |
| `ConversationHistory` | Accumulated transcript turns                         |
| your custom type      | Whatever you passed as `context=` to `RealtimeAgent` |

### Example: shared context object

```python
from dataclasses import dataclass
from typing import Annotated
from rtvoice import Inject, RealtimeAgent, Tools
from rtvoice.conversation import ConversationHistory

@dataclass
class AppContext:
    user_id: str
    preferences: dict

tools = Tools()

@tools.action("Save the user's preferred language")
async def save_language(
    language: Annotated[str, "The language code, e.g. 'de'"],
    ctx: Inject[AppContext],
) -> str:
    ctx.preferences["language"] = language
    return f"Language set to {language}."

app_ctx = AppContext(user_id="u-42", preferences={})

agent = RealtimeAgent(
    instructions="Help the user configure their preferences.",
    tools=tools,
    context=app_ctx,
)
await agent.run()
```

### Example: reading conversation history

```python
from rtvoice import Inject
from rtvoice.conversation import ConversationHistory

@tools.action("Summarise the conversation so far")
async def summarise(history: Inject[ConversationHistory]) -> str:
    return history.format()
```

---

## Synchronous tools

Sync functions work too — the framework detects `async` automatically:

```python
@tools.action("Convert Celsius to Fahrenheit")
def celsius_to_fahrenheit(celsius: Annotated[float, "Temperature in °C"]) -> str:
    return f"{celsius * 9/5 + 32:.1f}°F"
```

---

## Full example

```python
import asyncio
from typing import Annotated
from rtvoice import RealtimeAgent, Tools

tools = Tools()

@tools.action(
    "Look up the current stock price for a ticker symbol",
    result_instruction="State the price clearly and mention the currency.",
    is_long_running=True,
    holding_instruction="Let me check the markets for you.",
)
async def get_stock_price(
    ticker: Annotated[str, "Stock ticker symbol, e.g. AAPL"],
) -> str:
    # replace with real API call
    return f"{ticker}: $192.50 USD"


async def main():
    agent = RealtimeAgent(
        instructions=(
            "You are a financial assistant. "
            "Use get_stock_price when the user asks about stock prices."
        ),
        tools=tools,
    )
    await agent.run()

asyncio.run(main())
```

---

## API reference

See [`Tools`](../api/tools.md) for the complete class documentation.
