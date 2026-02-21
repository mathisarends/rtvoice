from rtvoice import RealtimeAgent
from rtvoice.views import TranscriptListener


class ConsolePrinter(TranscriptListener):
    async def on_user_chunk(self, chunk: str) -> None:
        print(chunk, end="", flush=True)

    async def on_user_completed(self, transcript: str) -> None:
        print()

    async def on_assistant_chunk(self, chunk: str) -> None:
        print(chunk, end="", flush=True)

    async def on_assistant_completed(self, transcript: str) -> None:
        print()


async def main():
    instructions = """Du bist Jarvis. Antworte kurz und bündig."""

    agent = RealtimeAgent(
        instructions=instructions,
        transcript_listener=ConsolePrinter(),
    )
    await agent.start()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
