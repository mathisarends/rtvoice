from dotenv import load_dotenv

from rtvoice import RealtimeAgent

load_dotenv(override=True)


async def main():
    agent = RealtimeAgent(instructions="Du bist Jarvis. Antworte kurz und bündig.")
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
