"""
rtvoice Showcase — Email & Smart Home
======================================

Two subagents handle distinct domains:

- **Mail Assistant** — reads, summarises, and sends emails (mocked)
- **Hue Assistant** — controls Philips Hue lights by room and scene

Try saying
----------
- "Do I have any unread emails?"
    → Mail Assistant fetches and summarises the inbox.

- "Send a reply to Jonas saying I'll be there at 3."
    → Mail Assistant drafts and sends a reply.

- "Turn off the lights in the living room."
    → Hue Assistant finds the room and switches it off.

- "Set the bedroom to a warm reading scene."
    → Hue Assistant picks the closest matching scene and applies it.

- "What time is it?"
    → Answered directly by the RealtimeAgent — no subagent involved.

Running
-------
::

    OPENAI_API_KEY=sk-... python showcase.py
"""

import asyncio
import logging
from datetime import datetime
from typing import Annotated

from dotenv import load_dotenv
from pydantic import BaseModel

from rtvoice import RealtimeAgent, SubAgent, Tools
from rtvoice.llm import ChatOpenAI

load_dotenv(override=True)

logging.getLogger("rtvoice.events.bus").setLevel(logging.WARNING)


class Email(BaseModel):
    id: str
    sender: str
    subject: str
    preview: str
    read: bool


class Inbox(BaseModel):
    unread_count: int
    emails: list[Email]


class EmailBody(BaseModel):
    id: str
    sender: str
    subject: str
    body: str


class SendResult(BaseModel):
    success: bool
    message: str


class HueRoom(BaseModel):
    id: str
    name: str
    lights_on: bool
    brightness: int
    color_temp: str


class HueScene(BaseModel):
    id: str
    name: str
    room: str


class HueApplyResult(BaseModel):
    success: bool
    room: str
    action: str


def _build_mail_tools() -> Tools:
    tools = Tools()

    _mock_emails = [
        Email(
            id="msg-001",
            sender="jonas@example.com",
            subject="Team lunch tomorrow",
            preview="Hey, are you joining us for lunch tomorrow at noon?",
            read=False,
        ),
        Email(
            id="msg-002",
            sender="newsletter@techdigest.io",
            subject="Your weekly tech digest",
            preview="This week: Rust 2025 edition, Claude 4, and more.",
            read=False,
        ),
        Email(
            id="msg-003",
            sender="boss@viadee.de",
            subject="Q2 planning",
            preview="Can we sync Thursday about the Q2 roadmap?",
            read=True,
        ),
    ]

    @tools.action(
        "Fetch the inbox and return all emails with sender, subject, and read state.",
        status="Lade Posteingang...",
    )
    async def get_inbox() -> Inbox:
        await asyncio.sleep(0.8)
        return Inbox(
            unread_count=sum(1 for e in _mock_emails if not e.read),
            emails=_mock_emails,
        )

    @tools.action(
        "Fetch the full body of a specific email by its ID.",
        status="Lade Email {email_id}...",
    )
    async def get_email_body(
        email_id: Annotated[str, "The email ID from the inbox listing."],
    ) -> EmailBody:
        await asyncio.sleep(0.6)
        email = next((e for e in _mock_emails if e.id == email_id), None)
        if not email:
            return EmailBody(
                id=email_id, sender="", subject="", body="Email not found."
            )
        return EmailBody(
            id=email.id,
            sender=email.sender,
            subject=email.subject,
            body=f"{email.preview} [... full body ...]",
        )

    @tools.action(
        "Send an email. Use this to reply to an existing email or compose a new one.",
        status="Sende Email an {to}...",
    )
    async def send_email(
        to: Annotated[str, "Recipient email address."],
        subject: Annotated[str, "Email subject line."],
        body: Annotated[str, "Full email body to send."],
    ) -> SendResult:
        await asyncio.sleep(1.0)
        return SendResult(success=True, message=f"Email sent to {to}.")

    return tools


def build_mail_assistant() -> SubAgent:
    return SubAgent(
        name="Mail Assistant",
        description=(
            "Reads, summarises, and sends emails on behalf of the user. "
            "Use for any request about the inbox, unread messages, "
            "reading an email, or sending and replying to emails."
        ),
        handoff_instructions=(
            "Include the user's intent: whether they want to read, summarise, or send. "
            "If replying, mention who to reply to and what to say."
        ),
        instructions=(
            "You are a personal email assistant.\n\n"
            "Reading flow:\n"
            "1. Call get_inbox() to fetch all emails.\n"
            "2. If the user wants to read a specific email, call get_email_body().\n"
            "3. Call done() with a natural conversational summary.\n\n"
            "Sending flow:\n"
            "1. If recipient, subject, or message content is unclear, call clarify().\n"
            "2. Call send_email().\n"
            "3. Call done() confirming what was sent and to whom.\n\n"
            "Summarise emails conversationally — never read them out verbatim."
        ),
        tools=_build_mail_tools(),
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        max_iterations=8,
        holding_instruction=(
            "Say one natural sentence like 'Let me check your inbox real quick!', then stop."
        ),
        result_instructions=(
            "Summarise what was found or done conversationally. "
            "For unread emails, mention sender and topic. "
            "For sent emails, confirm recipient and subject."
        ),
    )


def _build_hue_tools() -> Tools:
    tools = Tools()

    _mock_rooms: list[HueRoom] = [
        HueRoom(
            id="room-1",
            name="Living Room",
            lights_on=True,
            brightness=80,
            color_temp="warm",
        ),
        HueRoom(
            id="room-2",
            name="Bedroom",
            lights_on=False,
            brightness=0,
            color_temp="neutral",
        ),
        HueRoom(
            id="room-3",
            name="Office",
            lights_on=True,
            brightness=100,
            color_temp="cold",
        ),
    ]

    _mock_scenes: list[HueScene] = [
        HueScene(id="scene-1", name="Reading", room="Bedroom"),
        HueScene(id="scene-2", name="Movie Night", room="Living Room"),
        HueScene(id="scene-3", name="Energize", room="Office"),
        HueScene(id="scene-4", name="Relax", room="Living Room"),
    ]

    @tools.action(
        "List all available Hue rooms and their current light state.",
        status="Lade verfügbare Räume...",
    )
    async def list_rooms() -> list[HueRoom]:
        await asyncio.sleep(0.5)
        return list(_mock_rooms)

    @tools.action(
        "List available Hue scenes, optionally filtered by room name.",
        status="Lade Szenen...",
    )
    async def list_scenes(
        room: Annotated[
            str | None, "Filter by room name. Pass None to list all."
        ] = None,
    ) -> list[HueScene]:
        await asyncio.sleep(0.4)
        if room:
            return [s for s in _mock_scenes if s.room.lower() == room.lower()]
        return list(_mock_scenes)

    @tools.action(
        "Turn the lights in a room on or off.",
        status="Schalte Licht in Raum {room_id}...",
    )
    async def set_room_power(
        room_id: Annotated[str, "The room ID from list_rooms()."],
        on: Annotated[bool, "True to turn on, False to turn off."],
    ) -> HueApplyResult:
        await asyncio.sleep(0.6)
        room = next((r for r in _mock_rooms if r.id == room_id), None)
        return HueApplyResult(
            success=True,
            room=room.name if room else room_id,
            action="turned on" if on else "turned off",
        )

    @tools.action(
        "Apply a Hue scene to a room.",
        status="Aktiviere Szene {scene_id}...",
    )
    async def apply_scene(
        scene_id: Annotated[str, "The scene ID from list_scenes()."],
        room_id: Annotated[str, "The room ID from list_rooms()."],
    ) -> HueApplyResult:
        await asyncio.sleep(0.7)
        scene = next((s for s in _mock_scenes if s.id == scene_id), None)
        room = next((r for r in _mock_rooms if r.id == room_id), None)
        return HueApplyResult(
            success=True,
            room=room.name if room else room_id,
            action=f"scene '{scene.name if scene else scene_id}' applied",
        )

    @tools.action(
        "Set the brightness of a room's lights to a value between 0 and 100.",
        status="Setze Helligkeit auf {brightness}%...",
    )
    async def set_brightness(
        room_id: Annotated[str, "The room ID from list_rooms()."],
        brightness: Annotated[int, "Brightness level: 0 (off) to 100 (full)."],
    ) -> HueApplyResult:
        await asyncio.sleep(0.5)
        room = next((r for r in _mock_rooms if r.id == room_id), None)
        return HueApplyResult(
            success=True,
            room=room.name if room else room_id,
            action=f"brightness set to {brightness}%",
        )

    return tools


def build_hue_assistant() -> SubAgent:
    return SubAgent(
        name="Hue Assistant",
        description=(
            "Controls Philips Hue smart lights by room and scene. "
            "Use for any request about turning lights on or off, "
            "adjusting brightness, or activating a lighting scene."
        ),
        handoff_instructions=(
            "Include the room name if the user specified one. "
            "Include the desired action: on/off, a scene name, or a brightness level. "
            "If the user was vague, pass their phrasing as-is."
        ),
        instructions=(
            "You are a smart home lighting assistant.\n\n"
            "Always start by calling list_rooms() to fetch available rooms. "
            "Match the user's room name to the closest result. "
            "If ambiguous, call clarify().\n\n"
            "Power control: call set_room_power() with the matched room ID.\n\n"
            "Scene activation: call list_scenes(room=...) to find available scenes, "
            "match the closest scene name, then call apply_scene().\n\n"
            "Brightness: call set_brightness() with the desired level.\n\n"
            "Always finish with done() confirming what changed and in which room."
        ),
        tools=_build_hue_tools(),
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
        max_iterations=8,
        holding_instruction=(
            "Say one natural sentence like 'On it — adjusting your lights now!', then stop."
        ),
        result_instructions=(
            "Confirm what changed and in which room in one short natural sentence."
        ),
    )


async def main() -> None:
    tools = Tools()

    @tools.action("Get the current time in a human-friendly format.")
    def get_current_time() -> str:
        return datetime.now().strftime("%I:%M %p")

    agent = RealtimeAgent(
        instructions=(
            "You are Jarvis, a calm and efficient personal voice assistant.\n\n"
            "Answer simple questions (time, definitions, quick maths) directly.\n\n"
            "For anything related to emails, inbox, or sending messages, "
            "hand off to the Mail Assistant.\n"
            "For anything related to lights, brightness, or scenes, "
            "hand off to the Hue Assistant.\n"
            "Do not attempt those tasks yourself."
        ),
        subagents=[build_mail_assistant(), build_hue_assistant()],
        tools=tools,
        inactivity_timeout_seconds=90,
        inactivity_timeout_enabled=True,
    )

    print("🎙  Jarvis is ready.\n")
    print("  Try: 'Do I have any unread emails?'")
    print("       'Reply to Jonas and say I will be there at 3.'")
    print("       'Turn off the living room lights.'")
    print("       'Set the bedroom to a reading scene.'")
    print("       'What time is it?'  ← answered directly, no subagent")
    print("\n  Speak mid-task to test interruption. Ctrl+C to stop.\n")

    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
