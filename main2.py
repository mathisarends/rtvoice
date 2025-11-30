import asyncio
import os

from rtvoice import Agent
from rtvoice.tools.mcp.models import MCPServerConfig
from rtvoice.tools.mcp.server import MCPServer


async def main():
    user_docs = os.path.join(os.path.expanduser("~"), "Documents")

    async with MCPServer(
        MCPServerConfig(
            name="filesystem",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-filesystem", user_docs],
        )
    ) as mcp_server:
        agent = Agent(
            instructions="You are a helpful assistant with file system access.",
            mcp_servers=[mcp_server],
        )

        await agent.start()


if __name__ == "__main__":
    asyncio.run(main())
