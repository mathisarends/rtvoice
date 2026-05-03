"""
rtvoice Showcase — Supervisor Tool Chaining Demo
=================================================

Demonstrates how the supervisor autonomously combines multiple tools
to answer a location- and time-aware question.

Try saying
----------
- "Is the IKEA near me open right now?"
- "What time does IKEA close today?"
- "Can I still make it to IKEA?"

Running
-------
::

    OPENAI_API_KEY=sk-... python showcase_ikea_hours.py
"""

import asyncio
import logging
from datetime import datetime

from dotenv import load_dotenv

from rtvoice import RealtimeAgent, Supervisor, Tools
from rtvoice.llm import ChatOpenAI

load_dotenv(override=True)
logging.basicConfig(level=logging.INFO)


# ---------------------------------------------------------------------------
# Mock data
# ---------------------------------------------------------------------------

_IKEA_HOURS: dict[str, dict[str, str]] = {
    "monday": {"open": "10:00", "close": "20:00"},
    "tuesday": {"open": "10:00", "close": "20:00"},
    "wednesday": {"open": "10:00", "close": "20:00"},
    "thursday": {"open": "10:00", "close": "21:00"},
    "friday": {"open": "10:00", "close": "21:00"},
    "saturday": {"open": "09:00", "close": "20:00"},
    "sunday": {"open": "10:00", "close": "19:00"},
}

_USER_LOCATION = {
    "city": "Münster",
    "country": "Germany",
    "lat": 51.9607,
    "lon": 7.6261,
    "nearest_ikea": "IKEA Münster",
    "ikea_address": "Albersloher Weg 99, 48167 Münster",
    "distance_km": 4.2,
}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


def build_tools() -> Tools:
    tools = Tools()

    @tools.action(
        "Return the current local date, time, weekday, and timezone. "
        "Call this whenever you need to know what time or day it is."
    )
    async def get_current_datetime() -> dict:
        await asyncio.sleep(0.1)
        now = datetime.now()
        return {
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M"),
            "weekday": now.strftime("%A").lower(),
            "timezone": "Europe/Berlin",
        }

    @tools.action(
        "Return the user's current geographic location and the nearest IKEA store. "
        "Call this to find out where the user is before searching for local store info."
    )
    async def get_user_location() -> dict:
        await asyncio.sleep(0.4)
        return _USER_LOCATION

    @tools.action(
        "Perform a web search and return the result. "
        "Use this to look up store opening hours, business info, or similar facts. "
        "Pass a natural-language query string."
    )
    async def search_web(query: str) -> dict:
        await asyncio.sleep(0.6)
        # Mock: always return IKEA Münster opening hours regardless of query
        return {
            "query": query,
            "source": "ikea.com (mock)",
            "result": {
                "store": "IKEA Münster",
                "address": "Albersloher Weg 99, 48167 Münster",
                "phone": "+49 251 7008-0",
                "opening_hours": _IKEA_HOURS,
                "note": "Hours may differ on public holidays.",
            },
        }

    return tools


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------


def build_supervisor() -> Supervisor:
    return Supervisor(
        description=(
            "Answers questions about whether a local store is currently open, "
            "combining the user's location, the current time, and a web search."
        ),
        instructions=(
            "You are a helpful local-search assistant.\n\n"
            "When asked whether a store is open right now, follow these steps:\n\n"
            "1. Call get_current_datetime() to find out today's weekday and current time.\n"
            "2. Call get_user_location() to find the nearest relevant store.\n"
            "3. Call search_web() with a query like "
            "'IKEA <city> opening hours' to retrieve the store's schedule.\n"
            "4. Compare the current time against the opening hours for today's weekday.\n"
            "5. Call report_progress() with a brief intermediate status if steps 1–3 took a while.\n"
            "6. Call done() with a clear spoken answer: is the store open right now, "
            "when does it close (or when does it next open), and how far away it is."
        ),
        tools=build_tools(),
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        max_iterations=12,
        holding_instruction="On it...",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    agent = RealtimeAgent(
        instructions=(
            "You are Jet, a calm and friendly personal voice assistant.\n"
            "Your only responsible for user facing conversation. If the question is more complex - you delegate to the supervisor agent."
        ),
        supervisor=build_supervisor(),
        inactivity_timeout_seconds=90,
        inactivity_timeout_enabled=True,
    )

    print("🛒  Say: 'Is the IKEA near me open right now?'\n")
    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())
