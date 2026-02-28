import asyncio
from typing import Annotated

from llmify import ChatOpenAI

from rtvoice import RealtimeAgent, Tools
from rtvoice.subagents import SubAgent


async def main():
    instructions = """Du bist Jarvis. Antworte kurz und bündig."""

    email_tools = Tools()

    @email_tools.action(
        "Send an email to the given recipient with the given subject and body."
    )
    async def send_email(
        recipient: Annotated[str, "The email address of the recipient."],
        subject: Annotated[str, "The subject of the email."],
        body: Annotated[str, "The body of the email."],
    ) -> str:
        await asyncio.sleep(3)  # simuliert langlaufenden Versand
        print(f"\n📧 Email sent to {recipient} | Subject: {subject}\n{body}\n")
        return f"Email successfully sent to {recipient}."

    email_agent = SubAgent(
        name="email_agent",
        description="Sends an email in the background. Use when the user wants to send an email.",
        instructions="You are an email assistant. Send the email using the send_email tool and confirm.",
        llm=ChatOpenAI(model="gpt-4o-mini"),
        tools=email_tools,
        fire_and_forget=True,
        result_instructions="The email is being sent in the background.",
    )

    summary_agent = SubAgent(
        name="summary_agent",
        description="Summarizes the conversation so far. Use when the user asks for a summary of what was discussed.",
        instructions=(
            "You are a summarization assistant. "
            "The user will provide you with the conversation history. "
            "Summarize it concisely in German."
        ),
        llm=ChatOpenAI(model="gpt-4o-mini"),
        result_instructions="Here is the summary of our conversation so far.",
    )

    agent = RealtimeAgent(
        instructions=instructions,
        subagents=[email_agent, summary_agent],
    )
    await agent.run()


if __name__ == "__main__":
    asyncio.run(main())
