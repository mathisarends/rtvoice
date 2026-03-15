# MCP Servers

[Model Context Protocol (MCP)](https://modelcontextprotocol.io/) is an open standard for exposing tools to LLMs. `rtvoice` supports connecting any MCP server that communicates over stdio, and registering its tools with the voice agent or one or more subagents.

---

## Quick start

```python
from rtvoice import RealtimeAgent
from rtvoice.mcp import MCPServerStdio

server = MCPServerStdio(
    command="npx",
    args=["-y", "@modelcontextprotocol/server-filesystem", "/tmp"],
)

agent = RealtimeAgent(
    instructions="You can read and write files in /tmp.",
    mcp_servers=[server],
)
await agent.run()
```

`RealtimeAgent` connects to the server during `prepare()`, lists all available tools, and registers them automatically. No extra code required.

---

## `MCPServerStdio`

Spawns a subprocess and communicates with it over stdin/stdout using JSON-RPC 2.0.

```python
from rtvoice.mcp import MCPServerStdio

server = MCPServerStdio(
    command="python",           # executable to spawn
    args=["my_mcp_server.py"],  # arguments
    env={"MY_VAR": "value"},    # optional environment variables (inherits current env if None)
    cache_tools_list=True,      # cache tool discovery after first call (default: True)
    allowed_tools=["read_file", "write_file"],  # whitelist; None = expose all tools
)
```

| Parameter          | Description                                                                             |
| ------------------ | --------------------------------------------------------------------------------------- |
| `command`          | Executable to spawn as the MCP server process.                                          |
| `args`             | Command-line arguments.                                                                 |
| `env`              | Environment variables for the subprocess. Inherits the current environment when `None`. |
| `cache_tools_list` | Cache the tool list after the first `list_tools()` call.                                |
| `allowed_tools`    | Whitelist of tool names to expose. All tools are exposed when `None`.                   |

---

## Using a server directly

`MCPServerStdio` implements an async context manager:

```python
async with MCPServerStdio(command="npx", args=["-y", "@my/server"]) as server:
    tools = await server.list_tools()
    print([t.name for t in tools])

    result = await server.call_tool("my_tool", {"param": "value"})
    print(result)
```

---

## Attaching MCP servers to a subagent (recommended)

For complex workflows, attach MCP servers to a `SubAgent` rather than `RealtimeAgent`. This keeps the realtime model's tool list short and delegates the actual work to the subagent:

```python
from rtvoice.llm import ChatOpenAI
from rtvoice import RealtimeAgent, SubAgent
from rtvoice.mcp import MCPServerStdio

calendar_server = MCPServerStdio(
    command="npx",
    args=["-y", "@example/calendar-mcp"],
)

calendar_agent = SubAgent(
    name="Calendar Assistant",
    description="Manages the user's calendar events.",
    instructions="Use the provided calendar tools to read and create events.",
    llm=ChatOpenAI(model="gpt-4o"),
    mcp_servers=[calendar_server],
)

agent = RealtimeAgent(
    instructions="Delegate all calendar requests to the Calendar Assistant.",
    subagents=[calendar_agent],
)
await agent.run()
```

!!! tip
Prefer attaching MCP servers to subagents rather than to `RealtimeAgent` directly. The realtime model has a limited tool window; keeping subagent domain tools separate reduces noise and improves routing accuracy.

---

## Multiple servers

Pass any number of servers at once:

```python
agent = RealtimeAgent(
    instructions="...",
    mcp_servers=[
        MCPServerStdio(command="npx", args=["-y", "@example/server-a"]),
        MCPServerStdio(command="npx", args=["-y", "@example/server-b"]),
    ],
)
```

All servers are connected in parallel during `prepare()`. If one fails, the others continue and an error is logged.

---

## Prewarming

MCP connections are established during `prepare()`. Call it explicitly before `run()` to avoid a startup delay when the first user speaks:

```python
await agent.prepare()
await agent.run()
```

---

## Writing your own MCP server

Any process that speaks JSON-RPC 2.0 over stdio and implements `initialize`, `tools/list`, and `tools/call` works. Here is a minimal Python example:

```python
# my_mcp_server.py
import json, sys

def handle(request: dict) -> dict:
    method = request.get("method")
    id_ = request.get("id")

    if method == "initialize":
        return {"jsonrpc": "2.0", "id": id_, "result": {"protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {"name": "demo", "version": "0.1"}}}
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": id_, "result": {"tools": [{"name": "greet", "description": "Say hello", "inputSchema": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}}]}}
    if method == "tools/call":
        name = request["params"]["arguments"]["name"]
        return {"jsonrpc": "2.0", "id": id_, "result": {"content": [{"type": "text", "text": f"Hello, {name}!"}]}}

for line in sys.stdin:
    req = json.loads(line)
    resp = handle(req)
    sys.stdout.write(json.dumps(resp) + "\n")
    sys.stdout.flush()
```

---

## API reference

See [`MCPServerStdio`](../api/audio.md) and the [Views reference](../api/views.md) for full parameter documentation.
