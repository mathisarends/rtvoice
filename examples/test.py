from rtvoice import RealtimeAgent, Tools


async def main():
    instructions = """Du bist Jarvis. Antworte kurz und bündig. Wenn du eine Frage nicht beantworten kannst, sage "Das weiß ich leider nicht"."""

    tools = Tools()

    @tools.action("get_current_time")
    def get_current_time():
        from datetime import datetime

        return datetime.now().isoformat()

    agent = RealtimeAgent(instructions=instructions, tools=tools)
    await agent.run()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
