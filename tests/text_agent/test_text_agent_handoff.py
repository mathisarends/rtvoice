from unittest.mock import AsyncMock, MagicMock

import pytest
from transitbus import EventBus

from rtvoice import TextAgent
from rtvoice.conversation import ConversationHistory
from rtvoice.tools import ActionResult, Inject, ToolContext, Tools, ToolSchemaFormat


class FileSystem:
    pass


class CommandPolicy:
    pass


def make_text_agent(**kwargs) -> TextAgent:
    return TextAgent(
        description="Planning helper",
        system_prompt="You are a planner.",
        llm=MagicMock(),
        **kwargs,
    )


def exposed_names(tools: Tools) -> set[str]:
    return {tool.name for tool in tools.get_schema()}


def test_handoff_tool_is_hidden_without_a_text_agent() -> None:
    tools = Tools()
    tools.set_context(ToolContext())

    assert "text_agent" not in exposed_names(tools)


def test_handoff_tool_is_exposed_once_a_text_agent_is_injected() -> None:
    tools = Tools()
    tools.set_context(ToolContext(make_text_agent()))

    assert "text_agent" in exposed_names(tools)


def test_handoff_description_includes_handoff_instructions() -> None:
    tools = Tools()
    tools.set_context(
        ToolContext(make_text_agent(handoff_instructions="Hand over the full date."))
    )

    tool = next(tool for tool in tools.get_schema() if tool.name == "text_agent")

    assert tool.description is not None
    assert "Planning helper" in tool.description
    assert "Hand over the full date." in tool.description


def test_text_agent_cannot_hand_off_to_itself() -> None:
    text_agent = make_text_agent()

    exposed = {
        tool.name for tool in text_agent._tools.get_schema(ToolSchemaFormat.TEXT)
    }

    assert "text_agent" not in exposed


def test_text_agent_context_drops_an_injected_text_agent() -> None:
    nested = make_text_agent()

    text_agent = make_text_agent(tool_dependencies=(nested,))

    assert text_agent._tools._context.resolve(TextAgent) is None


@pytest.mark.asyncio
async def test_text_agent_injects_multiple_user_dependencies() -> None:
    file_system = FileSystem()
    policy = CommandPolicy()
    received = {}
    tools = Tools()

    @tools.action("Use application dependencies")
    async def use_dependencies(
        injected_file_system: Inject[FileSystem],
        injected_policy: Inject[CommandPolicy],
    ) -> ActionResult:
        received["file_system"] = injected_file_system
        received["policy"] = injected_policy
        return ActionResult.success()

    text_agent = make_text_agent(
        tools=tools,
        tool_dependencies=(file_system, policy),
    )

    result = await text_agent._tools.execute("use_dependencies")

    assert result.ok
    assert received == {"file_system": file_system, "policy": policy}


@pytest.mark.asyncio
async def test_handoff_result_carries_the_agents_result_instructions() -> None:
    text_agent = make_text_agent(result_instructions="Read the answer verbatim.")
    text_agent.start = AsyncMock(return_value="Done")
    tools = Tools()
    tools.set_context(ToolContext(text_agent, ConversationHistory(EventBus())))

    result = await tools.execute("text_agent", {"task": "Plan my day"})

    assert result.value == "Done"
    assert result.instruction == "Read the answer verbatim."
