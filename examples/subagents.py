import asyncio
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
            "You are a weather assistant. Use the get_weather tool to answer "
            "questions about current conditions. Be concise and friendly."
        ),
        tools=build_weather_tools(),
        llm=ChatOpenAI(model="gpt-4o", temperature=0.2),
        pending_message="One moment, I'll check the current weather for you.",
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
