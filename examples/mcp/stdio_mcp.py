import os

from rtvoice.mcp.server import MCPServerStdio


async def main():
    server = MCPServerStdio(
        command="python",
        args=[os.path.join(os.path.dirname(__file__), "mock_mcp_server.py")],
    )

    async with server:
        tools = await server.list_tools()
        print([t.name for t in tools])  # ["greet", "add"]
        print("Tools", tools)

        result = await server.call_tool("greet", {"name": "Mathis"})
        print(result)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
