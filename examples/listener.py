from rtvoice import Agent
from rtvoice.views import AgentHistory, AgentListener


class LifecycleLogger(AgentListener):
    async def on_agent_started(self) -> None:
        print("Agent gestartet")

    async def on_agent_stopped(self, history: AgentHistory) -> None:
        print(f"\nAgent gestoppt - {len(history.conversation_turns)} Turns")
        for turn in history.conversation_turns:
            print(f"  [{turn.role}]: {turn.transcript}")

    async def on_agent_interrupted(self) -> None:
        print("\n[Unterbrochen]")


async def main():
    agent = Agent(
        instructions="Du bist Jarvis. Antworte kurz und bündig.",
        agent_listener=LifecycleLogger(),
    )
    await agent.start()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
