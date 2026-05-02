"""
Conversation seed demo.

Running
-------
::

    OPENAI_API_KEY=sk-... python examples/context_injection.py
"""

import asyncio

from dotenv import load_dotenv

from rtvoice import ConversationSeed, RealtimeAgent, SeedMessage

load_dotenv(override=True)


async def main() -> None:
    agent = RealtimeAgent(
        extends_system_prompt=(
            "Du bist ein knapper Support-Agent. "
            "Nutze den vorhandenen Kontext, ohne ihn erneut abzufragen."
        ),
        conversation_seed=ConversationSeed(
            [
                SeedMessage.user("Ich bin Max, Nutzer-ID 42, Premium-Kunde."),
                SeedMessage.assistant(
                    "Verstanden, Max. Ich habe deine Kundendaten im Kontext."
                ),
            ]
        ),
    )

    print("Frag zum Beispiel: Welche Kundendaten kennst du schon?")
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
