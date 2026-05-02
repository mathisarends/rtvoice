"""
rtvoice Showcase — report_progress Demo
========================================

Try saying
----------
- "Start my morning routine."

Running
-------
::

    OPENAI_API_KEY=sk-... python showcase.py
"""

import asyncio
import logging

from dotenv import load_dotenv

from rtvoice import RealtimeAgent, SubAgent, Tools
from rtvoice.llm import ChatOpenAI

load_dotenv(override=True)
logging.getLogger("rtvoice.events.bus").setLevel(logging.WARNING)


def build_tools() -> Tools:
    tools = Tools()

    @tools.action("Fetch current weather.")
    async def get_weather() -> dict:
        await asyncio.sleep(0.6)
        return {"condition": "partly cloudy", "temp_c": 14}

    @tools.action("Fetch unread emails from inbox.")
    async def get_unread_emails() -> dict:
        await asyncio.sleep(0.7)
        return {"unread": 3, "senders": ["jonas@example.com", "boss@viadee.de"]}

    @tools.action("Turn lights on or off in a room.")
    async def set_room_power(room: str, on: bool) -> dict:
        await asyncio.sleep(0.5)
        return {"room": room, "state": "on" if on else "off"}

    @tools.action("Activate a lighting scene in a room.")
    async def apply_scene(room: str, scene: str) -> dict:
        await asyncio.sleep(0.6)
        return {"room": room, "scene": scene, "applied": True}

    @tools.action("Start a music playlist in a room.")
    async def play_music(room: str, playlist: str) -> dict:
        await asyncio.sleep(0.8)
        return {"room": room, "playlist": playlist, "playing": True}

    return tools


def build_morning_agent() -> SubAgent:
    return SubAgent(
        name="Morning Routine Agent",
        description="Runs the user's morning routine: checks weather and inbox, sets up lights and music.",
        instructions=(
            "You are a morning routine assistant. "
            "Execute these steps in order:\n\n"
            "1. get_weather()\n"
            "2. get_unread_emails()\n"
            "3. set_room_power(room='bedroom', on=False)\n"
            "4. apply_scene(room='kitchen', scene='Energize')\n"
            "5. play_music(room='kitchen', playlist='Morning Vibes')\n\n"
            "After step 2, call report_progress with a brief summary of weather and inbox. "
            "After all steps, call done() with a full spoken summary."
        ),
        tools=build_tools(),
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        max_iterations=15,
        holding_instruction="Say one short sentence like 'Starting your morning routine!' then stop.",
        result_instructions="Summarise what was set up in one or two natural sentences.",
    )


async def main() -> None:
    agent = RealtimeAgent(
        instructions=(
            "You are Jarvis, a calm personal voice assistant.\n"
            "For morning routine requests, hand off to the Morning Routine Agent.\n"
            "Do not attempt those tasks yourself."
        ),
        subagents=[build_morning_agent()],
        inactivity_timeout_seconds=90,
        inactivity_timeout_enabled=True,
    )

    print("🌅  Say: 'Start my morning routine.'\n")
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
