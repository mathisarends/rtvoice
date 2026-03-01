from rtvoice import AgentListener, RealtimeAgent
from rtvoice.views import TranscriptionModel


class ConsolePrinter(AgentListener):
    async def on_user_transcript(self, transcript: str) -> None:
        print(f"\033[36mDu: {transcript}\033[0m")

    async def on_assistant_transcript(self, transcript: str) -> None:
        print(f"Jarvis: {transcript}")


async def main():
    instructions = """Du bist Jarvis. Antworte kurz und bündig."""

    agent = RealtimeAgent(
        instructions=instructions,
        transcription_model=TranscriptionModel.WHISPER_1,
        listener=ConsolePrinter(),
    )
    await agent.run()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
