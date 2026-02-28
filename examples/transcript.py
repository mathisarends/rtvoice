from rtvoice import RealtimeAgent
from rtvoice.views import TranscriptionModel, TranscriptListener


class ConsolePrinter(TranscriptListener):
    async def on_user_completed(self, transcript: str) -> None:
        print(f"\033[36mDu: {transcript}\033[0m")

    async def on_assistant_completed(self, transcript: str) -> None:
        print(f"Jarvis: {transcript}")


async def main():
    instructions = """Du bist Jarvis. Antworte kurz und bündig."""

    agent = RealtimeAgent(
        instructions=instructions,
        transcription_model=TranscriptionModel.WHISPER_1,
        transcript_listener=ConsolePrinter(),
    )
    await agent.run()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
