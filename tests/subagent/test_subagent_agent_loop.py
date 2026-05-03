from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from rtvoice import Supervisor
from rtvoice.agent.views import SupervisorClarificationNeeded, SupervisorDone
from rtvoice.llm import (
    AssistantMessage,
    ChatInvokeCompletion,
    ChatInvokeUsage,
    Function,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
)
from rtvoice.tools import Tools


class TestSupervisorRunAndLoop:
    @pytest.mark.asyncio
    async def test_run_returns_completion_when_no_tool_calls(self) -> None:
        llm = MagicMock()
        llm._model = "gpt-5.4-mini"
        llm.invoke = AsyncMock(
            return_value=ChatInvokeCompletion(
                completion="Final answer",
                tool_calls=[],
                usage=ChatInvokeUsage(
                    prompt_tokens=100,
                    prompt_cached_tokens=25,
                    completion_tokens=10,
                    total_tokens=110,
                ),
            )
        )

        agent = Supervisor(
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
        )

        result = await agent.run(task="Plan my day")

        assert isinstance(result, SupervisorDone)
        assert result.message == "Final answer"
        assert llm.invoke.await_count == 1

    @pytest.mark.asyncio
    async def test_run_includes_context_in_conversation_history_block(self) -> None:
        llm = MagicMock()
        llm.invoke = AsyncMock(
            return_value=ChatInvokeCompletion(completion="Done", tool_calls=[])
        )

        agent = Supervisor(
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
        )

        await agent.run(task="Plan my day", context="User asked for morning focus")

        invoke_messages = llm.invoke.await_args_list[0].args[0]
        assert invoke_messages[0].content == "You are a planner."
        assert (
            invoke_messages[1].content
            == "<conversation_history>\nUser asked for morning focus\n</conversation_history>"
        )
        assert invoke_messages[2].content == "<task>\nPlan my day\n</task>"

    @pytest.mark.asyncio
    async def test_run_returns_done_when_done_tool_is_called(self) -> None:
        llm = MagicMock()
        done_call = ToolCall(
            id="call_done",
            function=Function(name="done", arguments='{"result":"Done via tool"}'),
        )
        llm.invoke = AsyncMock(
            return_value=ChatInvokeCompletion(
                completion="Working on it",
                tool_calls=[done_call],
            )
        )

        agent = Supervisor(
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
        )

        result = await agent.run(task="Plan my day")

        assert isinstance(result, SupervisorDone)
        assert result.message == "Done via tool"

    @pytest.mark.asyncio
    async def test_run_returns_clarification_needed_when_clarify_tool_is_called(
        self,
    ) -> None:
        llm = MagicMock()
        clarify_call = ToolCall(
            id="call_clarify",
            function=Function(
                name="clarify",
                arguments='{"question":"Which day should I optimize?"}',
            ),
        )
        llm.invoke = AsyncMock(
            return_value=ChatInvokeCompletion(
                completion="Need one detail",
                tool_calls=[clarify_call],
            )
        )

        agent = Supervisor(
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
        )

        result = await agent.run(task="Plan my day")

        assert isinstance(result, SupervisorClarificationNeeded)
        assert result.question == "Which day should I optimize?"
        assert result.clarify_call_id == "call_clarify"
        assert isinstance(result.resume_history[-1], AssistantMessage)
        assert result.resume_history[-1].tool_calls[0].id == "call_clarify"

    @pytest.mark.asyncio
    async def test_loop_appends_tool_result_message_before_next_invoke(self) -> None:
        llm = MagicMock()
        search_call = ToolCall(
            id="call_search",
            function=Function(name="search_schedule", arguments='{"query":"Monday"}'),
        )
        llm.invoke = AsyncMock(
            side_effect=[
                ChatInvokeCompletion(completion="Checking", tool_calls=[search_call]),
                ChatInvokeCompletion(completion="Found one appointment", tool_calls=[]),
            ]
        )

        tools = Tools()

        @tools.action(description="Search schedule")
        async def search_schedule(query: str) -> str:
            return f"result:{query}"

        agent = Supervisor(
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
            tools=tools,
            max_iterations=3,
        )

        result = await agent.run(task="Plan my day")

        assert isinstance(result, SupervisorDone)
        assert result.message == "Found one appointment"
        assert llm.invoke.await_count == 2

        second_invoke_messages = llm.invoke.await_args_list[1].args[0]
        assert any(
            isinstance(message, ToolResultMessage)
            and message.tool_call_id == "call_search"
            and message.content == "result:Monday"
            for message in second_invoke_messages
        )

    @pytest.mark.asyncio
    async def test_loop_returns_max_iterations_reached_when_tool_chain_never_finishes(
        self,
    ) -> None:
        llm = MagicMock()
        repeat_call = ToolCall(
            id="call_repeat",
            function=Function(name="echo", arguments='{"text":"ping"}'),
        )
        llm.invoke = AsyncMock(
            side_effect=[
                ChatInvokeCompletion(completion="Looping", tool_calls=[repeat_call]),
                ChatInvokeCompletion(
                    completion="Still looping", tool_calls=[repeat_call]
                ),
            ]
        )

        tools = Tools()

        @tools.action(description="Echo text")
        async def echo(text: str) -> str:
            return text

        agent = Supervisor(
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
            tools=tools,
            max_iterations=2,
        )

        result = await agent.run(task="Plan my day")

        assert isinstance(result, SupervisorDone)
        assert result.message == "Max iterations reached."
        assert result.success is False
        assert llm.invoke.await_count == 2


class TestSupervisorResumeAndPrewarm:
    def test_name_is_fixed_to_supervisor(self) -> None:
        agent = Supervisor(
            description="Planning helper",
            instructions="You are a planner.",
            llm=MagicMock(),
        )

        assert agent.name == "supervisor"

    @pytest.mark.asyncio
    async def test_resume_appends_tool_result_message_with_clarification_answer(
        self,
    ) -> None:
        llm = MagicMock()
        llm.invoke = AsyncMock(
            return_value=ChatInvokeCompletion(completion="Completed after answer")
        )

        agent = Supervisor(
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
        )

        resume_history = [SystemMessage(content="You are a planner.")]

        result = await agent.resume(
            clarification_answer="Use Tuesday",
            resume_history=resume_history,
            clarify_call_id="call_clarify_1",
        )

        assert isinstance(result, SupervisorDone)
        assert result.message == "Completed after answer"

        invoke_messages = llm.invoke.await_args_list[0].args[0]
        assert isinstance(invoke_messages[-1], ToolResultMessage)
        assert invoke_messages[-1].tool_call_id == "call_clarify_1"
        assert invoke_messages[-1].content == "Use Tuesday"

    @pytest.mark.asyncio
    async def test_run_calls_prewarm_before_first_invoke(self) -> None:
        llm = MagicMock()
        llm.invoke = AsyncMock(
            return_value=ChatInvokeCompletion(completion="done", tool_calls=[])
        )

        agent = Supervisor(
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
        )

        await agent.run(task="Plan my day")

        assert llm.invoke.await_count == 1

    @pytest.mark.asyncio
    async def test_run_after_resume_handles_follow_up_clarification(self) -> None:
        llm = MagicMock()
        clarify_call = ToolCall(
            id="call_follow_up",
            function=Function(
                name="clarify",
                arguments='{"question":"Need one more detail"}',
            ),
        )
        llm.invoke = AsyncMock(
            return_value=ChatInvokeCompletion(
                completion="Still missing input",
                tool_calls=[clarify_call],
            )
        )

        agent = Supervisor(
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
        )

        result = await agent.resume(
            clarification_answer="Tuesday",
            resume_history=[SystemMessage(content="You are a planner.")],
            clarify_call_id="call_initial",
        )

        assert isinstance(result, SupervisorClarificationNeeded)
        assert result.question == "Need one more detail"
        assert result.clarify_call_id == "call_follow_up"
