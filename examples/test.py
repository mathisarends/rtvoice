from rtvoice import Agent


async def main():
    agent = Agent()
    await agent.start()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
