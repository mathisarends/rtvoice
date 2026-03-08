import asyncio

from rtvoice import AgentListener, RealtimeAgent


class AgentListenerImpl(AgentListener):
    async def on_user_inactivity_countdown(self, remaining_seconds):
        print(f"User inactivity countdown: {remaining_seconds} seconds remaining")


async def main():
    agent = RealtimeAgent(
        instructions="Du bist Jarvis. Antworte kurz und bündig.",
        inactivity_timeout_enabled=True,
        inactivity_timeout_seconds=10,
    )
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
