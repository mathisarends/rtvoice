"""
Clarification + Small Talk Bridging example
============================================
Demonstrates two patterns working together:

1. **Clarification**: When the Booking Agent lacks essential info (party size),
   it calls ``clarify()`` - the main agent asks the user, waits for their voice
   reply, then feeds the answer back so the SupervisorAgent can continue.

2. **Small Talk Bridging**: Once all details are known, the booking runs in the
   background while the main agent says a natural holding phrase.

Flow
----
User   : "Book me a table at Mario's tonight at 8."
Agent  : "For how many people?"                    ← clarify(), SupervisorAgent pauses
User   : "Two please."                             ← answer_future resolved
         [SupervisorAgent continues with party_size=2]
Agent  : "I'm reserving that table right now!"    ← holding response
         [reserve_table runs ~2 s in background]
Agent  : "Done! Table for two at Mario's tonight  ← result response
          at 8 PM, confirmed as booking #BK0001."
"""

import asyncio
from typing import Annotated

from llmify import ChatOpenAI

from rtvoice import RealtimeAgent, SupervisorAgent, Tools

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


def build_booking_agent() -> SupervisorAgent:
    return SupervisorAgent(
        name="Booking Agent",
        description=(
            "Reserves restaurant tables. Use this agent whenever the user wants "
            "to book, reserve, or make a dining reservation."
        ),
        handoff_instructions=(
            "Include restaurant name, date, and time in the task if the user mentioned them. "
            "Do NOT guess party size – the agent will ask if it is missing."
        ),
        instructions=(
            "You are a restaurant booking assistant. "
            "You MUST know: restaurant, date, time, and party size before booking. "
            "If party size is not provided in the task, use clarify() to ask the user. "
            "Then call reserve_table() and finally done() with the confirmation."
        ),
        tools=build_booking_tools(),
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.2),
        holding_instruction=(
            "The booking is being arranged right now. "
            "Say ONE warm, brief sentence to the user while they wait "
            "(e.g. 'I'm reserving that table for you right now!'). "
            "Then stop immediately. Do not mention the result yet."
        ),
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
        SupervisorAgents=[build_booking_agent()],
    )

    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
