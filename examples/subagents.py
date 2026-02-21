import asyncio
from typing import Annotated

from llmify import ChatOpenAI

from rtvoice import RealtimeAgent, SubAgent, Tools


def build_weather_tools() -> Tools:
    tools = Tools()

    @tools.action("Add two numbers together")
    def add(
        a: Annotated[float, "First number"],
        b: Annotated[float, "Second number"],
    ) -> float:
        return a + b

    @tools.action("Fetch the weather for a given city (mocked)")
    def get_weather(city: Annotated[str, "City name"]) -> str:
        mock_data = {
            "berlin": "12°C, partly cloudy",
            "munich": "8°C, sunny",
            "hamburg": "5°C, rainy",
        }
        return mock_data.get(city.lower(), f"No weather data available for '{city}'")

    return tools


async def main() -> None:
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

    assistant_agent = SubAgent(
        name="assistant",
        description="Handles math calculations and weather lookups",
        instructions="You are a helpful assistant. Use the available tools to answer accurately.",
        tools=build_weather_tools(),
        llm=llm,
    )

    agent = RealtimeAgent(
        instructions="You are a voice assistant. Delegate math and weather questions to your assistant agent.",
        subagents=[assistant_agent],
    )

    await agent.start()


if __name__ == "__main__":
    asyncio.run(main())
