import logging

from dotenv import load_dotenv

from rtvoice import EchoCancellation, RealtimeAgent

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)


async def main():
    agent = RealtimeAgent(
        system_prompt="Du bist Jarvis. Antworte kurz und bündig.",
        inactivity_timeout_seconds=7,
        echo_cancellation=EchoCancellation(),
    )
    result = await agent.start()

    cost = result.usage.cost
    print(f"\nKosten: {cost.total} {cost.currency}")
    for item in cost.line_items:
        print(
            f"  {item.category}: {item.quantity} {item.unit} -> {item.cost} {cost.currency}"
        )


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
