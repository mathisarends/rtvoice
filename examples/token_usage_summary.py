import asyncio

from dotenv import load_dotenv

from rtvoice import RealtimeAgent

load_dotenv(override=True)


async def main() -> None:
    agent = RealtimeAgent(
        instructions="Du bist ein hilfreicher Voice-Assistent.",
        inactivity_timeout_enabled=True,
        inactivity_timeout_seconds=5.0,
    )

    result = await agent.run()
    print(f"Turns: {len(result.turns)}")
    for turn in result.turns:
        print(f"  {turn.role}: {turn.text}")


if __name__ == "__main__":
    asyncio.run(main())
