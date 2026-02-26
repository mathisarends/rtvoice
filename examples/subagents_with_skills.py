import asyncio
from pathlib import Path
from typing import Annotated

from llmify import ChatOpenAI

from rtvoice import RealtimeAgent, SubAgent, Tools

_MOCK_WEATHER = {
    "berlin": "12°C, partly cloudy",
    "munich": "8°C, sunny",
    "hamburg": "5°C, rainy",
}


def build_weather_tools() -> Tools:
    tools = Tools()

    @tools.action("Fetch the current weather for a given city.")
    def get_weather(
        city: Annotated[str, "The city name to look up weather for."],
    ) -> str:
        return _MOCK_WEATHER.get(
            city.lower(),
            f"No weather data available for '{city}'.",
        )

    return tools


async def main() -> None:
    weather_agent = SubAgent(
        name="Weather Assistant",
        description="Looks up current weather conditions for any city.",
        handoff_instructions=(
            "Use this agent whenever the user asks about weather, temperature, "
            "or conditions in a specific city. Always include the city name in the task."
        ),
        instructions=(
            "You are a helpful assistant. "
            "Before starting any task, check your available skills and load the relevant one."
        ),
        tools=build_weather_tools(),
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.2),
        skills_dir=Path(__file__).parent / "skills",
    )

    agent = RealtimeAgent(
        instructions=(
            "You are a friendly voice assistant. Answer general questions directly. "
            "For weather-related questions, delegate to the Weather Assistant."
        ),
        subagents=[weather_agent],
    )

    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
