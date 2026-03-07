import asyncio
import logging
from pathlib import Path

from rtvoice import RealtimeAgent
from rtvoice.views import AssistantVoice

logging.basicConfig(level=logging.INFO)


async def main():
    agent = RealtimeAgent(
        instructions="Du bist ein hilfreicher Assistent.",
        voice=AssistantVoice.MARIN,
        recording_path=Path("recordings/session.wav"),
        inactivity_timeout_seconds=5,
    )

    result = await agent.run()

    print("\n--- Session beendet ---")
    print(f"Recording: {result.recording_path}")


if __name__ == "__main__":
    asyncio.run(main())
