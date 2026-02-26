from rtvoice import RealtimeAgent
from rtvoice.views import AgentListener


class LifecycleLogger(AgentListener):
    async def on_agent_started(self) -> None:
        print("Agent gestartet")

    async def on_agent_interrupted(self) -> None:
        print("\n[Unterbrochen]")


async def main():
    agent = RealtimeAgent(
        instructions="Du bist Jarvis. Antworte kurz und bündig.",
        agent_listener=LifecycleLogger(),
    )
    await agent.run()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
