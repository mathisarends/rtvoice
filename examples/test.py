import asyncio

from llmify import ChatOpenAI

from rtvoice import RealtimeAgent
from rtvoice.supervisor import SupervisorAgent


async def main():
    summary_agent = SupervisorAgent(
        name="summary_agent",
        description="Summarizes the conversation so far.",
        instructions="Summarize the conversation concisely in German.",
        llm=ChatOpenAI(model="gpt-4o-mini"),
        result_instructions="Here is the summary of our conversation so far.",
    )

    agent = RealtimeAgent(
        instructions="Du bist Jarvis. Antworte kurz und bündig.",
        supervisor_agent=summary_agent,
    )
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
