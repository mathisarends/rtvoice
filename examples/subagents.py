import asyncio
from typing import Annotated

from llmify import ChatOpenAI

from rtvoice import SubAgent, Tools


def build_tools() -> Tools:
    tools = Tools()

    @tools.action("Add two numbers together")
    def add(
        a: Annotated[float, "First number"],
        b: Annotated[float, "Second number"],
    ) -> float:
        return a + b

    @tools.action("Fetch the weather for a given city (mocked)")
    def get_weather(
        city: Annotated[str, "City name"],
    ) -> str:
        mock_data = {
            "berlin": "12°C, partly cloudy",
            "munich": "8°C, sunny",
            "hamburg": "5°C, rainy",
        }
        return mock_data.get(city.lower(), f"No weather data available for '{city}'")

    return tools


async def main() -> None:
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

    agent = SubAgent(
        name="assistant",
        description="A helpful assistant that can do math and check the weather.",
        instructions="You are a helpful assistant. Use the available tools to answer the user's question accurately.",
        tools=build_tools(),
        llm=llm,
    )

    tasks = [
        "What is 123 + 456?",
        "What's the weather like in Berlin?",
        "What is the weather in Munich and Hamburg? Also calculate 99 + 1.",
    ]

    for task in tasks:
        print(f"\nTask: {task}")
        result = await agent.run(task)
        print(f"Result: {result}")


if __name__ == "__main__":
    asyncio.run(main())
