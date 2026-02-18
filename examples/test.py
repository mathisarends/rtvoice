from rtvoice import RealtimeAgent


async def main():
    instructions = """Du bist Jarvis. Antworte kurz und bündig. Wenn du eine Frage nicht beantworten kannst, sage "Das weiß ich leider nicht"."""

    agent = RealtimeAgent(instructions=instructions)
    await agent.start()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
