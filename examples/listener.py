from dotenv import load_dotenv

from rtvoice import RealtimeAgent
from rtvoice.agent.listener import AgentListener

load_dotenv(override=True)


class LifecycleLogger(AgentListener):
    async def on_agent_session_connected(self) -> None:
        print("Agent gestartet")

    async def on_agent_interrupted(self) -> None:
        print("\n[Unterbrochen]")


async def main():
    agent = RealtimeAgent(
        extends_system_prompt="Du bist Jarvis. Antworte kurz und bündig.",
        listener=LifecycleLogger(),
    )
    await agent.run()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
