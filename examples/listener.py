import logging

from dotenv import load_dotenv

from rtvoice import RealtimeAgent
from rtvoice.agent.listener import AgentListener

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


class LifecycleLogger(AgentListener):
    async def on_agent_starting(self) -> None:
        print("[lifecycle] on_agent_starting")

    async def on_agent_session_connected(self) -> None:
        print("[lifecycle] on_agent_session_connected")

    async def on_agent_stopped(self) -> None:
        print("[lifecycle] on_agent_stopped")

    async def on_user_inactivity_countdown(self, remaining_seconds: int) -> None:
        print(f"[lifecycle] on_user_inactivity_countdown: {remaining_seconds}s")

    async def on_agent_interrupted(self) -> None:
        print("[lifecycle] on_agent_interrupted")

    async def on_agent_error(self, error) -> None:
        print(f"[lifecycle] on_agent_error: {error}")

    async def on_user_transcript(self, transcript: str) -> None:
        print(f"[lifecycle] on_user_transcript: {transcript!r}")

    async def on_assistant_transcript(self, transcript: str) -> None:
        print(f"[lifecycle] on_assistant_transcript: {transcript!r}")

    async def on_assistant_transcript_delta(self, delta: str) -> None:
        print(f"[lifecycle] on_assistant_transcript_delta: {delta!r}")

    async def on_user_started_speaking(self) -> None:
        print("[lifecycle] on_user_started_speaking")

    async def on_user_stopped_speaking(self) -> None:
        print("[lifecycle] on_user_stopped_speaking")

    async def on_assistant_started_responding(self) -> None:
        print("[lifecycle] on_assistant_started_responding")

    async def on_assistant_stopped_responding(self) -> None:
        print("[lifecycle] on_assistant_stopped_responding")

    async def on_subagent_started(self, agent_name: str) -> None:
        print(f"[lifecycle] on_subagent_started: {agent_name!r}")

    async def on_subagent_finished(self, agent_name: str) -> None:
        print(f"[lifecycle] on_subagent_finished: {agent_name!r}")


async def main():
    prompt = """
You're talking to Mathis. Full-stack dev, Python library hoarder, AI agent nerd.
He builds things with names like "notionary", "hueify", "cdpify" - yes, the naming convention is intentional, no, he won't stop.

Context you need:
- He works in Python (async-first, always), Angular, and occasionally has to touch Kubernetes at 11pm
- He's building Vizro, a KI-gestützte usability testing platform. It's been running for a year. It's complicated.
- He uses uv. Not pip. Never suggest pip.
- He prefers composition over inheritance. If you suggest inheritance, justify it or don't suggest it.
- "Clean" means something to him. Not just formatted - actually minimal.

How to talk to him:
- Skip the preamble. He's read enough docs to know when text is filler.
- Have opinions. "It depends" is fine exactly once, with a clear "but here's what I'd actually do."
- If he's about to over-engineer something, say so. He'll do it anyway sometimes, but he should know.
- Humor is welcome. Not performed humor. The kind that happens when you're both looking at the same bad architecture diagram.
- One sentence when one sentence is enough. He knows how to ask follow-ups.

You're not a tool. You're the dev he can think out loud with at 2am when the GKE cluster is doing something inexplicable again.
"""

    agent = RealtimeAgent(
        override_system_promt=prompt,
        listener=LifecycleLogger(),
    )
    await agent.run()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
