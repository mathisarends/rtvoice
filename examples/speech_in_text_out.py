import asyncio

from dotenv import load_dotenv

from rtvoice import AgentListener, RealtimeAgent
from rtvoice.audio import AudioOutput

load_dotenv(override=True)


class NoAudioOutput(AudioOutput):
    @property
    def is_playing(self) -> bool:
        return False

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def play_chunk(self, chunk: bytes) -> None:
        pass

    async def clear_buffer(self) -> None:
        pass


class TextStream(AgentListener):
    async def on_user_transcript(self, transcript: str) -> None:
        print(f"\nYou: {transcript}\nAssistant: ", end="", flush=True)

    async def on_assistant_transcript_delta(self, delta: str) -> None:
        print(delta, end="", flush=True)

    async def on_assistant_transcript(self, transcript: str) -> None:
        print()


async def main() -> None:
    agent = RealtimeAgent(
        system_prompt="Answer concisely.",
        output_modalities=["text"],
        audio_output=NoAudioOutput(),
        listener=TextStream(),
    )
    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())
