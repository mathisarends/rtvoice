"""
Example demonstrating assistant response output modalities (audio and text).

This example shows:
- Text output modality: streams assistant responses as text chunks (deltas)
- Audio output modality: streams assistant responses as audio (default behavior)
- Combined modalities: both text and audio streamed simultaneously
"""

from dotenv import load_dotenv

from rtvoice import RealtimeAgent
from rtvoice.views import AgentListener

load_dotenv(override=True)


class TextAndAudioListener(AgentListener):
    """Listener that demonstrates text delta streaming and completion."""

    async def on_agent_starting(self) -> None:
        print("🚀 Agent starting...\n")

    async def on_agent_session_connected(self) -> None:
        print("✅ Session connected and ready\n")

    async def on_user_started_speaking(self) -> None:
        print("\n👤 User started speaking...")

    async def on_user_stopped_speaking(self) -> None:
        print("👤 User stopped speaking")

    async def on_assistant_started_responding(self) -> None:
        print("🤖 Assistant responding...\n")

    async def on_assistant_transcript_delta(self, delta: str) -> None:
        """Called when text output streamed as chunks (resp.text.delta events)."""
        print(delta, end="", flush=True)

    async def on_assistant_transcript(self, transcript: str) -> None:
        """Called when assistant response is complete."""
        print("\n\n✔️ Response complete")

    async def on_agent_interrupted(self) -> None:
        print("\n⏸️ Assistant response interrupted")

    async def on_agent_error(self, error) -> None:
        print(f"\n❌ Error: {error}")

    async def on_agent_stopped(self) -> None:
        print("\n\n👋 Agent stopped\n")


async def main_text_and_audio():
    """Demonstrate streaming both text and audio simultaneously.

    The assistant response is:
    1. Streamed as text chunks via on_assistant_transcript_delta
    2. Streamed as audio being played back in real-time
    """
    print("=" * 60)
    print("TEXT + AUDIO OUTPUT MODALITIES")
    print("=" * 60)
    print(
        "The assistant will respond with both text deltas AND audio\n"
        "Listen to the audio while seeing live text updates\n"
    )

    agent = RealtimeAgent(
        instructions="Du bist ein hilfsbereiter Assistent. Antworte kurz und präzise.",
        listener=TextAndAudioListener(),
        output_modalities=["text", "audio"],  # Enable both modalities
        inactivity_timeout_seconds=30.0,
        inactivity_timeout_enabled=True,
    )
    await agent.run()


async def main_text_only():
    """Demonstrate text-only output without audio.

    Useful for text-based interfaces or when you only want the transcript.
    """
    print("=" * 60)
    print("TEXT ONLY OUTPUT MODALITY")
    print("=" * 60)
    print("The assistant will respond with text only (no audio)\n")

    agent = RealtimeAgent(
        instructions="Du bist ein hilfsbereiter Assistent. Antworte kurz und präzise.",
        listener=TextAndAudioListener(),
        output_modalities=["text"],
        inactivity_timeout_seconds=30.0,
        inactivity_timeout_enabled=True,
    )
    await agent.run()


async def main_audio_only():
    """Demonstrate audio-only output (default)."""
    print("=" * 60)
    print("AUDIO ONLY OUTPUT MODALITY (DEFAULT)")
    print("=" * 60)
    print("The assistant will respond with audio only\n")

    agent = RealtimeAgent(
        instructions="Du bist ein hilfsbereiter Assistent. Antworte kurz und präzise.",
        listener=TextAndAudioListener(),
        output_modalities=["audio"],  # Audio only (default)
        inactivity_timeout_seconds=30.0,
        inactivity_timeout_enabled=True,
    )
    await agent.run()


def print_menu():
    print("\n" + "=" * 60)
    print("OUTPUT MODALITIES DEMO")
    print("=" * 60)
    print("Choose an example:")
    print("1. Text + Audio (recommended)")
    print("2. Text only")
    print("3. Audio only (default)")
    print("0. Exit")
    print("-" * 60)


async def main():
    while True:
        print_menu()
        choice = input("Enter your choice (0-3): ").strip()

        if choice == "1":
            await main_text_and_audio()
        elif choice == "2":
            await main_text_only()
        elif choice == "3":
            await main_audio_only()
        elif choice == "0":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
