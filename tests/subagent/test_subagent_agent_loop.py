from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from rtvoice.llm import (
    AssistantMessage,
    ChatInvokeCompletion,
    ChatInvokeUsage,
    Function,
    SystemMessage,
    ToolCall,
    ToolResultMessage,
)
from rtvoice.realtime.schemas import FunctionTool
from rtvoice.subagent import SubAgent
from rtvoice.subagent.views import AgentClarificationNeeded, AgentDone
from rtvoice.tools import Tools


class TestSubAgentRunAndLoop:
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

        agent = SubAgent(
            name="planner",
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
        )

        result = await agent.run(task="Plan my day")

        assert isinstance(result, AgentDone)
        assert result.message == "Final answer"
        assert result.token_usage.usage.input_text_tokens == 75
        assert result.token_usage.usage.cached_input_text_tokens == 25
        assert result.token_usage.usage.output_text_tokens == 10
        assert llm.invoke.await_count == 1

    @pytest.mark.asyncio
    async def test_run_includes_context_in_conversation_history_block(self) -> None:
        llm = MagicMock()
        llm.invoke = AsyncMock(
            return_value=ChatInvokeCompletion(completion="Done", tool_calls=[])
        )

        agent = SubAgent(
            name="planner",
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

        agent = SubAgent(
            name="planner",
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
        )

        result = await agent.run(task="Plan my day")

        assert isinstance(result, AgentDone)
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

        agent = SubAgent(
            name="planner",
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
        )

        result = await agent.run(task="Plan my day")

        assert isinstance(result, AgentClarificationNeeded)
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

        agent = SubAgent(
            name="planner",
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
            tools=tools,
            max_iterations=3,
        )

        result = await agent.run(task="Plan my day")

        assert isinstance(result, AgentDone)
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

        agent = SubAgent(
            name="planner",
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
            tools=tools,
            max_iterations=2,
        )

        result = await agent.run(task="Plan my day")

        assert isinstance(result, AgentDone)
        assert result.message == "Max iterations reached."
        assert result.success is False
        assert llm.invoke.await_count == 2


class TestSubAgentResumeAndPrewarm:
    def test_name_normalizes_spaces_to_underscores(self) -> None:
        agent = SubAgent(
            name="calendar helper",
            description="Planning helper",
            instructions="You are a planner.",
            llm=MagicMock(),
        )

        assert agent.name == "calendar_helper"

    @pytest.mark.asyncio
    async def test_resume_appends_tool_result_message_with_clarification_answer(
        self,
    ) -> None:
        llm = MagicMock()
        llm.invoke = AsyncMock(
            return_value=ChatInvokeCompletion(completion="Completed after answer")
        )

        agent = SubAgent(
            name="planner",
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

        assert isinstance(result, AgentDone)
        assert result.message == "Completed after answer"

        invoke_messages = llm.invoke.await_args_list[0].args[0]
        assert isinstance(invoke_messages[-1], ToolResultMessage)
        assert invoke_messages[-1].tool_call_id == "call_clarify_1"
        assert invoke_messages[-1].content == "Use Tuesday"

    @pytest.mark.asyncio
    async def test_prewarm_with_no_mcp_servers_sets_ready_event(self) -> None:
        llm = MagicMock()
        llm.invoke = AsyncMock(
            return_value=ChatInvokeCompletion(completion="done", tool_calls=[])
        )

        agent = SubAgent(
            name="planner",
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
        )

        await agent.prewarm()

        assert agent._mcp_ready.is_set() is True

    @pytest.mark.asyncio
    async def test_run_calls_prewarm_before_first_invoke(self) -> None:
        llm = MagicMock()
        llm.invoke = AsyncMock(
            return_value=ChatInvokeCompletion(completion="done", tool_calls=[])
        )

        agent = SubAgent(
            name="planner",
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
        )

        await agent.run(task="Plan my day")

        assert agent._mcp_ready.is_set() is True
        assert llm.invoke.await_count == 1

    @pytest.mark.asyncio
    async def test_prewarm_registers_tools_from_mcp_server_once(self) -> None:
        llm = MagicMock()
        llm.invoke = AsyncMock(
            return_value=ChatInvokeCompletion(completion="done", tool_calls=[])
        )
        server = MagicMock()
        server.connect = AsyncMock()
        server.list_tools = AsyncMock(
            return_value=[
                FunctionTool(name="calendar_lookup", parameters={"type": "object"})
            ]
        )

        agent = SubAgent(
            name="planner",
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
            mcp_servers=[server],
        )

        await agent.prewarm()
        await agent.prewarm()

        assert server.connect.await_count == 1
        assert server.list_tools.await_count == 1
        assert agent._tools.get("calendar_lookup") is not None

    @pytest.mark.asyncio
    async def test_prewarm_continues_when_one_mcp_server_fails(self) -> None:
        llm = MagicMock()
        llm.invoke = AsyncMock(
            return_value=ChatInvokeCompletion(completion="done", tool_calls=[])
        )

        failing_server = MagicMock()
        failing_server.connect = AsyncMock(side_effect=RuntimeError("boom"))
        failing_server.list_tools = AsyncMock(return_value=[])

        healthy_server = MagicMock()
        healthy_server.connect = AsyncMock()
        healthy_server.list_tools = AsyncMock(
            return_value=[
                FunctionTool(name="weather_lookup", parameters={"type": "object"})
            ]
        )

        agent = SubAgent(
            name="planner",
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
            mcp_servers=[failing_server, healthy_server],
        )

        await agent.prewarm()

        assert agent._mcp_ready.is_set() is True
        assert healthy_server.connect.await_count == 1
        assert healthy_server.list_tools.await_count == 1
        assert agent._tools.get("weather_lookup") is not None

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

        agent = SubAgent(
            name="planner",
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
        )

        result = await agent.resume(
            clarification_answer="Tuesday",
            resume_history=[SystemMessage(content="You are a planner.")],
            clarify_call_id="call_initial",
        )

        assert isinstance(result, AgentClarificationNeeded)
        assert result.question == "Need one more detail"
        assert result.clarify_call_id == "call_follow_up"


class TestSubAgentProgressReporting:
    @pytest.mark.asyncio
    async def test_progress_tool_invokes_callback_and_continues_loop(self) -> None:
        llm = MagicMock()
        progress_call = ToolCall(
            id="call_progress",
            function=Function(
                name="report_progress",
                arguments='{"message":"Loading calendar..."}',
            ),
        )
        done_call = ToolCall(
            id="call_done",
            function=Function(name="done", arguments='{"result":"All done"}'),
        )
        llm.invoke = AsyncMock(
            side_effect=[
                ChatInvokeCompletion(completion="Working", tool_calls=[progress_call]),
                ChatInvokeCompletion(completion="Finishing", tool_calls=[done_call]),
            ]
        )

        progress_messages: list[str] = []

        async def capture_progress(msg: str) -> None:
            progress_messages.append(msg)

        agent = SubAgent(
            name="planner",
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
        )

        result = await agent.run(task="Plan my day", on_progress=capture_progress)

        assert isinstance(result, AgentDone)
        assert result.message == "All done"
        assert progress_messages == ["Loading calendar..."]
        assert llm.invoke.await_count == 2

    @pytest.mark.asyncio
    async def test_progress_appends_acknowledgment_to_messages(self) -> None:
        llm = MagicMock()
        progress_call = ToolCall(
            id="call_progress",
            function=Function(
                name="report_progress",
                arguments='{"message":"Step 1 done"}',
            ),
        )
        llm.invoke = AsyncMock(
            side_effect=[
                ChatInvokeCompletion(completion="Working", tool_calls=[progress_call]),
                ChatInvokeCompletion(completion="Final answer", tool_calls=[]),
            ]
        )

        agent = SubAgent(
            name="planner",
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
        )
        agent.on_progress = AsyncMock()

        await agent.run(task="Plan my day")

        second_invoke_messages = llm.invoke.await_args_list[1].args[0]
        assert any(
            isinstance(m, ToolResultMessage)
            and m.tool_call_id == "call_progress"
            and m.content == "Progress noted, continue."
            for m in second_invoke_messages
        )

    @pytest.mark.asyncio
    async def test_progress_without_callback_continues_loop(self) -> None:
        llm = MagicMock()
        progress_call = ToolCall(
            id="call_progress",
            function=Function(
                name="report_progress",
                arguments='{"message":"Step 1 done"}',
            ),
        )
        done_call = ToolCall(
            id="call_done",
            function=Function(name="done", arguments='{"result":"Done"}'),
        )
        llm.invoke = AsyncMock(
            side_effect=[
                ChatInvokeCompletion(completion="Working", tool_calls=[progress_call]),
                ChatInvokeCompletion(completion="Finishing", tool_calls=[done_call]),
            ]
        )

        agent = SubAgent(
            name="planner",
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
        )

        result = await agent.run(task="Plan my day")

        assert isinstance(result, AgentDone)
        assert result.message == "Done"

    @pytest.mark.asyncio
    async def test_on_progress_param_overrides_instance_attribute(self) -> None:
        llm = MagicMock()
        progress_call = ToolCall(
            id="call_progress",
            function=Function(
                name="report_progress",
                arguments='{"message":"update"}',
            ),
        )
        done_call = ToolCall(
            id="call_done",
            function=Function(name="done", arguments='{"result":"ok"}'),
        )
        llm.invoke = AsyncMock(
            side_effect=[
                ChatInvokeCompletion(completion="Working", tool_calls=[progress_call]),
                ChatInvokeCompletion(completion="Done", tool_calls=[done_call]),
            ]
        )

        instance_callback = AsyncMock()
        param_callback = AsyncMock()

        agent = SubAgent(
            name="planner",
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
        )
        agent.on_progress = instance_callback

        await agent.run(task="Plan my day", on_progress=param_callback)

        param_callback.assert_awaited_once_with("update")
        instance_callback.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_instance_on_progress_used_when_no_param(self) -> None:
        llm = MagicMock()
        progress_call = ToolCall(
            id="call_progress",
            function=Function(
                name="report_progress",
                arguments='{"message":"update"}',
            ),
        )
        done_call = ToolCall(
            id="call_done",
            function=Function(name="done", arguments='{"result":"ok"}'),
        )
        llm.invoke = AsyncMock(
            side_effect=[
                ChatInvokeCompletion(completion="Working", tool_calls=[progress_call]),
                ChatInvokeCompletion(completion="Done", tool_calls=[done_call]),
            ]
        )

        instance_callback = AsyncMock()

        agent = SubAgent(
            name="planner",
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
        )
        agent.on_progress = instance_callback

        await agent.run(task="Plan my day")

        instance_callback.assert_awaited_once_with("update")

    @pytest.mark.asyncio
    async def test_resume_forwards_on_progress_callback(self) -> None:
        llm = MagicMock()
        progress_call = ToolCall(
            id="call_progress",
            function=Function(
                name="report_progress",
                arguments='{"message":"Resuming..."}',
            ),
        )
        done_call = ToolCall(
            id="call_done",
            function=Function(name="done", arguments='{"result":"Resumed"}'),
        )
        llm.invoke = AsyncMock(
            side_effect=[
                ChatInvokeCompletion(completion="Working", tool_calls=[progress_call]),
                ChatInvokeCompletion(completion="Done", tool_calls=[done_call]),
            ]
        )

        callback = AsyncMock()

        agent = SubAgent(
            name="planner",
            description="Planning helper",
            instructions="You are a planner.",
            llm=llm,
        )

        result = await agent.resume(
            clarification_answer="Tuesday",
            resume_history=[SystemMessage(content="You are a planner.")],
            clarify_call_id="call_clarify_1",
            on_progress=callback,
        )

        assert isinstance(result, AgentDone)
        assert result.message == "Resumed"
        callback.assert_awaited_once_with("Resuming...")
