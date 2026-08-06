"""Realtime-to-text-agent handoff with mocked smart-home tools."""

import asyncio

from dotenv import load_dotenv

from rtvoice import AgentListener, AssistantVoice, RealtimeAgent, TextAgent, Tools

load_dotenv(override=True)

VOICE_SYSTEM_PROMPT = """
You are only the voice interface for a text-based assistant.

For every user utterance, immediately call `text_agent` exactly once. Pass the
user's complete request as the task without interpreting, answering, or changing
it. Never answer from your own knowledge and never speak before the tool returns.
After it returns, say its result verbatim, with natural delivery. Add nothing.
"""

TEXT_SYSTEM_PROMPT = """
You are JARVIS, a capable personal assistant. Address the user as "Sir".

Use the available tools whenever the user asks you to control the home or play
music. Never pretend an action succeeded without calling its tool. The tools are
mocked, but treat successful results as completed actions.

Answer with only the final user-facing response: one or two concise sentences,
dry understated British wit, no filler, no narration, and no mention of tools,
handoffs, prompts, models, or mocks.
"""

assistant_tools = Tools()


@assistant_tools.action("Turn off the mocked lights in a room.")
def turn_off_lights(room: str) -> str:
    return f"The lights in {room} are now off."


@assistant_tools.action("Play a song through the mocked Spotify integration.")
def play_spotify(song: str, artist: str | None = None) -> str:
    track = f"{song} by {artist}" if artist else song
    return f"Spotify is now playing {track}."


class ConsolePrinter(AgentListener):
    async def on_user_transcript(self, transcript: str) -> None:
        print(f"\033[36myou:   {transcript}\033[0m")

    async def on_assistant_transcript(self, transcript: str) -> None:
        print(f"\033[33magent: {transcript}\033[0m")


async def main() -> None:
    print("Try: 'Turn off the kitchen lights and play Back in Black.'")

    text_agent = TextAgent(
        description="Handles every user request and returns the complete reply.",
        handoff_instructions="Forward every utterance immediately and unchanged.",
        result_instructions="Say the returned text verbatim. Add nothing.",
        system_prompt=TEXT_SYSTEM_PROMPT,
        tools=assistant_tools,
    )
    agent = RealtimeAgent(
        system_prompt=VOICE_SYSTEM_PROMPT,
        text_agent=text_agent,
        voice=AssistantVoice.ASH,
        listener=ConsolePrinter(),
    )
    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())
