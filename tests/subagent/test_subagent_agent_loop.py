from unittest.mock import AsyncMock, MagicMock

import pytest

from rtvoice import Subagent
from rtvoice.llm import (
    ChatInvokeCompletion,
    Function,
    ToolCall,
    ToolResultMessage,
)
from rtvoice.tools import Tools


@pytest.mark.asyncio
async def test_returns_completion_when_no_tool_calls() -> None:
    llm = MagicMock()
    llm.invoke = AsyncMock(
        return_value=ChatInvokeCompletion(completion="Final answer", tool_calls=[])
    )
    subagent = Subagent(
        description="Planning helper",
        system_prompt="You are a planner.",
        llm=llm,
    )

    assert await subagent.start("Plan my day") == "Final answer"


@pytest.mark.asyncio
async def test_includes_context_in_messages() -> None:
    llm = MagicMock()
    llm.invoke = AsyncMock(return_value=ChatInvokeCompletion(completion="Done"))
    subagent = Subagent(
        description="Planning helper",
        system_prompt="You are a planner.",
        llm=llm,
    )

    await subagent.start("Plan my day", context="Morning focus")

    messages = llm.invoke.await_args.args[0]
    assert [message.content for message in messages] == [
        "You are a planner.",
        "<conversation_history>\nMorning focus\n</conversation_history>",
        "<task>\nPlan my day\n</task>",
    ]


@pytest.mark.asyncio
async def test_runs_tools_until_final_completion() -> None:
    call = ToolCall(
        id="call_search",
        function=Function(name="search_schedule", arguments='{"query":"Monday"}'),
    )
    llm = MagicMock()
    llm.invoke = AsyncMock(
        side_effect=[
            ChatInvokeCompletion(completion="Checking", tool_calls=[call]),
            ChatInvokeCompletion(completion="Found one appointment"),
        ]
    )
    tools = Tools()

    @tools.action(description="Search schedule")
    async def search_schedule(query: str) -> str:
        return f"result:{query}"

    subagent = Subagent(
        description="Planning helper",
        system_prompt="You are a planner.",
        llm=llm,
        tools=tools,
    )

    assert await subagent.start("Plan my day") == "Found one appointment"
    messages = llm.invoke.await_args_list[1].args[0]
    assert isinstance(messages[-1], ToolResultMessage)
    assert messages[-1].content == "result:Monday"


@pytest.mark.asyncio
async def test_returns_max_iterations_message() -> None:
    call = ToolCall(
        id="call_repeat",
        function=Function(name="echo", arguments='{"text":"ping"}'),
    )
    llm = MagicMock()
    llm.invoke = AsyncMock(
        return_value=ChatInvokeCompletion(completion="Looping", tool_calls=[call])
    )
    tools = Tools()

    @tools.action(description="Echo text")
    async def echo(text: str) -> str:
        return text

    subagent = Subagent(
        description="Planning helper",
        system_prompt="You are a planner.",
        llm=llm,
        tools=tools,
        max_iterations=2,
    )

    assert await subagent.start("Plan my day") == "Max iterations reached."
    assert llm.invoke.await_count == 2
