"""
Small Talk Bridging example
===========================
Demonstrates the Small Talk Bridging pattern: when a slow SubAgent is
called, the voice assistant immediately produces a one-sentence bridging
response ("Let me check that for you!") while the SubAgent runs in the
background.  Once the SubAgent finishes, the holding response has already
ended naturally and the final result is delivered without any awkward
silence or interrupted speech.

Flow
----
User   : "Book me a table at Mario's tonight at 8."
Agent  : "Let me arrange that for you!"          ← holding response
         [SubAgent runs in background ~2 s]
Agent  : "Done! Table for two at Mario's tonight ← result response
          at 8 PM, confirmed as booking #42."
"""

import asyncio
from typing import Annotated

from llmify import ChatOpenAI

from rtvoice import RealtimeAgent, SubAgent, Tools

# ---------------------------------------------------------------------------
# Mock booking tool (simulates ~2 s network latency)
# ---------------------------------------------------------------------------

_BOOKINGS: dict[str, str] = {}
_NEXT_ID = 1


def build_booking_tools() -> Tools:
    tools = Tools()

    @tools.action("Reserve a restaurant table for the given date, time and party size.")
    async def reserve_table(
        restaurant: Annotated[str, "Restaurant name."],
        date: Annotated[str, "Date in YYYY-MM-DD format."],
        time: Annotated[str, "Time in HH:MM format (24 h)."],
        party_size: Annotated[int, "Number of guests."],
    ) -> str:
        global _NEXT_ID
        await asyncio.sleep(2)  # simulate slow external booking system
        booking_id = f"BK{_NEXT_ID:04d}"
        _NEXT_ID += 1
        _BOOKINGS[booking_id] = f"{restaurant} on {date} at {time} for {party_size}"
        return (
            f"Booking confirmed. ID: {booking_id}. "
            f"{restaurant} on {date} at {time} for {party_size} guest(s)."
        )

    return tools


def build_booking_agent() -> SubAgent:
    return SubAgent(
        name="Booking Agent",
        description=(
            "Reserves restaurant tables. Use this agent whenever the user wants "
            "to book, reserve, or make a dining reservation."
        ),
        handoff_instructions=(
            "Always include restaurant name, date, time, and party size in the task. "
            "Infer reasonable defaults from context (e.g. today's date, party of 2)."
        ),
        instructions=(
            "You are a restaurant booking assistant. "
            "Use the reserve_table tool to complete bookings. "
            "Confirm all details clearly in your done() call."
        ),
        tools=build_booking_tools(),
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.2),
        # One sentence the assistant says WHILE the booking runs in the background.
        holding_instruction=(
            "The booking is being arranged right now. "
            "Say ONE warm, brief sentence to the user while they wait "
            "(e.g. 'I'm reserving that table for you right now!'). "
            "Then stop immediately. Do not mention the result yet."
        ),
        # How to present the finished result.
        result_instructions=(
            "The booking is complete. Present the confirmation naturally and "
            "conversationally – mention the restaurant, time, and booking ID."
        ),
    )


async def main() -> None:
    agent = RealtimeAgent(
        instructions=(
            "You are a friendly voice assistant. "
            "For restaurant reservations, delegate to the Booking Agent. "
            "For all other requests, answer directly."
        ),
        subagents=[build_booking_agent()],
    )

    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
