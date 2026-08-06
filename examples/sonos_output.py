"""Run a voice agent with microphone input and Sonos clip output.

Set OPENAI_API_KEY, SONOS_IP_ADDRESS, and SONOS_SPEAKER_NAME in `.env`, then run:
    uv run --extra audio --extra sonos python examples/sonos_output.py
"""

import asyncio
import logging

from dotenv import load_dotenv

from rtvoice import RealtimeAgent
from rtvoice.audio import SonosOutput

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)


async def main() -> None:
    agent = RealtimeAgent(
        system_prompt="Answer concisely.",
        audio_output=SonosOutput(),
    )
    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())
