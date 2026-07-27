"""Run a mocked light tool without a follow-up model response."""

import asyncio
import logging

from dotenv import load_dotenv

from rtvoice import RealtimeAgent, Tools

load_dotenv(override=True)
logging.basicConfig(level=logging.DEBUG)


class MockLights:
    def __init__(self) -> None:
        self.enabled = False

    async def turn_on(self) -> None:
        await asyncio.sleep(0.2)
        self.enabled = True
        print("[mock] Lights are on")


def build_tools(lights: MockLights) -> Tools:
    tools = Tools()

    @tools.action(
        "Turn on the lights. Call this when the user asks to turn them on.",
        respond=False,
    )
    async def turn_on_lights() -> dict[str, bool]:
        await lights.turn_on()
        return {"enabled": lights.enabled}

    return tools


async def main() -> None:
    lights = MockLights()
    agent = RealtimeAgent(
        system_prompt=(
            "You control mocked smart-home lights. Use turn_on_lights when asked. "
            "The tool completes the request without a spoken confirmation."
        ),
        tools=build_tools(lights),
    )
    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())
