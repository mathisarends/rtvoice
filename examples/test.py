from rtvoice import Agent, RealtimeModel


async def main():
    instructions = """Du bist Jarvis. Nutze immer die verfügbaren Tools wenn du gefragt wirst
    was die aktuelle Uhrzeit ist - rate sie niemals."""

    agent = Agent(instructions=instructions, model=RealtimeModel.GPT_REALTIME)
    await agent.start()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
