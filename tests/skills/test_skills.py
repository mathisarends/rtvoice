from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rtvoice import RealtimeAgent, Skills, Subagent
from rtvoice.skills import SkillManager
from rtvoice.skills.bash import BashArgs, BashRunner, validate_command
from rtvoice.tools import ToolContext, Tools


def make_skill(
    root: Path,
    *,
    name: str = "internet-research",
    description: str = "Research sources. Use when current facts are needed.",
    instructions: str = "# Internet Research\n\nUse the bundled workflow.",
) -> Path:
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        (
            "---\n"
            f"name: {name}\n"
            f"description: {description}\n"
            "license: Apache-2.0\n"
            "metadata:\n"
            "  author: test\n"
            '  version: "1.0"\n'
            "---\n"
            f"{instructions}\n"
        ),
        encoding="utf-8",
    )
    return skill_dir


class TestSkillManager:
    def test_discovery_discloses_metadata_but_not_instructions(
        self, tmp_path: Path
    ) -> None:
        make_skill(tmp_path)

        manager = SkillManager(Skills.from_local_dir(tmp_path))
        prompt = manager.discovery_prompt()

        assert "<name>internet-research</name>" in prompt
        assert "Research sources" in prompt
        assert "Use the bundled workflow" not in prompt
        assert manager.names() == ["internet-research"]

    def test_appends_discovery_prompt_to_instructions(self, tmp_path: Path) -> None:
        make_skill(tmp_path)
        manager = SkillManager(Skills.from_local_dir(tmp_path))

        instructions = manager.append_discovery_prompt("You are helpful.")

        assert instructions.startswith("You are helpful.")
        assert "<name>internet-research</name>" in instructions

    def test_load_skill_returns_instructions_base_dir_and_resources(
        self, tmp_path: Path
    ) -> None:
        skill_dir = make_skill(tmp_path)
        references = skill_dir / "references"
        references.mkdir()
        resource = references / "guide.md"
        resource.write_text("Only read me on demand.", encoding="utf-8")

        manager = SkillManager(Skills.from_local_dir(tmp_path))
        result = manager.load("internet-research")

        assert "Use the bundled workflow" in result
        assert f"Base directory for this skill: {skill_dir.resolve()}" in result
        assert f"<file>{resource.resolve()}</file>" in result
        assert "Only read me on demand." not in result

    def test_load_specific_resource_on_demand(self, tmp_path: Path) -> None:
        skill_dir = make_skill(tmp_path)
        references = skill_dir / "references"
        references.mkdir()
        (references / "guide.md").write_text("Loaded later.", encoding="utf-8")

        manager = SkillManager(Skills.from_local_dir(tmp_path))

        assert (
            manager.load("internet-research", "references/guide.md") == "Loaded later."
        )

    def test_resource_path_cannot_escape_skill(self, tmp_path: Path) -> None:
        make_skill(tmp_path)
        (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
        manager = SkillManager(Skills.from_local_dir(tmp_path))

        with pytest.raises(ValueError, match="outside skill"):
            manager.load("internet-research", "../secret.txt")

    def test_invalid_name_is_rejected(self, tmp_path: Path) -> None:
        make_skill(tmp_path, name="invalid--name")

        with pytest.raises(ValueError, match="without leading"):
            SkillManager(Skills.from_local_dir(tmp_path))

    def test_name_must_match_parent_directory(self, tmp_path: Path) -> None:
        skill_dir = make_skill(tmp_path)
        skill_file = skill_dir / "SKILL.md"
        skill_file.write_text(
            skill_file.read_text(encoding="utf-8").replace(
                "name: internet-research", "name: other-name"
            ),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="parent directory"):
            SkillManager(Skills.from_local_dir(tmp_path))

    def test_description_is_xml_escaped(self, tmp_path: Path) -> None:
        make_skill(tmp_path, description="Use for A < B & current facts.")

        prompt = SkillManager(Skills.from_local_dir(tmp_path)).discovery_prompt()

        assert "A &lt; B &amp; current facts" in prompt

    def test_later_source_overrides_duplicate(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        first = tmp_path / "first"
        second = tmp_path / "second"
        make_skill(first, description="First source.")
        make_skill(second, description="Second source.")

        manager = SkillManager(Skills.from_local_dir(first, second))

        assert "Second source." in manager.discovery_prompt()
        assert "First source." not in manager.discovery_prompt()
        assert any("overrides skill" in record.message for record in caplog.records)


class TestBash:
    def test_allows_whitelisted_commands_and_bundled_scripts(
        self, tmp_path: Path
    ) -> None:
        skill_dir = make_skill(tmp_path)
        scripts = skill_dir / "scripts"
        scripts.mkdir()
        script = scripts / "check.sh"
        script.write_text("#!/usr/bin/env bash\necho ok\n", encoding="utf-8")

        assert (
            validate_command(
                "cat SKILL.md",
                allowed_commands=frozenset({"cat"}),
                allowed_script_dirs=(skill_dir.resolve(),),
                cwd=str(skill_dir),
            )
            is None
        )
        assert (
            validate_command(
                "scripts/check.sh",
                allowed_commands=frozenset(),
                allowed_script_dirs=(skill_dir.resolve(),),
                cwd=str(skill_dir),
            )
            is None
        )

    def test_rejects_non_allowlisted_and_nested_commands(self, tmp_path: Path) -> None:
        skill_dir = make_skill(tmp_path)

        error = validate_command(
            "cat $(rm secret.txt)",
            allowed_commands=frozenset({"cat"}),
            allowed_script_dirs=(skill_dir.resolve(),),
            cwd=str(skill_dir),
        )

        assert error is not None
        assert "command_substitution" in error

    def test_rejects_cwd_outside_skill(self, tmp_path: Path) -> None:
        skill_dir = make_skill(tmp_path)
        outside = tmp_path / "outside"
        outside.mkdir()

        error = validate_command(
            "echo hello",
            allowed_commands=frozenset({"echo"}),
            allowed_script_dirs=(skill_dir.resolve(),),
            cwd=str(outside),
        )

        assert error is not None
        assert "not under an allowed skill directory" in error

    @pytest.mark.asyncio
    async def test_executes_allowlisted_command(self, tmp_path: Path) -> None:
        skill_dir = make_skill(tmp_path)
        runner = BashRunner(
            allowed_commands=("echo",),
            allowed_script_dirs=(skill_dir,),
        )

        result = await runner.execute(
            BashArgs(command="echo hello", cwd=str(skill_dir))
        )

        assert result == "hello"


class TestDefaultTools:
    def test_skill_defaults_are_hidden_without_dependencies(self) -> None:
        tools = Tools()

        exposed = {tool.name for tool in tools.get_tool_schema()}

        assert "load_skill" not in exposed
        assert "bash" not in exposed

    def test_skill_defaults_are_exposed_once_injected(self, tmp_path: Path) -> None:
        make_skill(tmp_path)
        manager = SkillManager(Skills.from_local_dir(tmp_path))
        tools = Tools()
        tools.set_context(ToolContext(manager, BashRunner.for_skills(manager, ["cat"])))

        exposed = {tool.name for tool in tools.get_tool_schema()}

        assert "load_skill" in exposed
        assert "bash" in exposed

    def test_merge_keeps_the_receivers_defaults(self) -> None:
        tools = Tools()

        tools.merge(Tools())

        assert tools.get("load_skill") is not None

    @pytest.mark.asyncio
    async def test_load_skill_uses_injected_manager(self, tmp_path: Path) -> None:
        make_skill(tmp_path)
        manager = SkillManager(Skills.from_local_dir(tmp_path))
        tools = Tools()
        tools.set_context(ToolContext(manager))

        loaded = await tools.execute("load_skill", {"name": "internet-research"})

        assert "Use the bundled workflow" in loaded.value


class TestAgentIntegration:
    def test_realtime_agent_gets_catalog_and_builtin_tools(
        self, tmp_path: Path
    ) -> None:
        make_skill(tmp_path)
        audio_input = MagicMock()
        audio_output = MagicMock()

        with patch("rtvoice.agent.realtime.OpenAIProvider"):
            agent = RealtimeAgent(
                instructions="You are helpful.",
                skills=Skills.from_local_dir(tmp_path),
                allowed_commands=["cat"],
                audio_input=audio_input,
                audio_output=audio_output,
            )

        instructions = agent._realtime_session._instructions
        assert instructions.startswith("You are helpful.")
        assert "<name>internet-research</name>" in instructions
        assert "Use the bundled workflow" not in instructions
        assert agent._tools.get("load_skill") is not None
        assert agent._tools.get("bash") is not None

    @pytest.mark.asyncio
    async def test_subagent_loads_skill_progressively(self, tmp_path: Path) -> None:
        make_skill(tmp_path)
        subagent = Subagent(
            description="Research subagent",
            instructions="Be accurate.",
            llm=MagicMock(),
            skills=Skills.from_local_dir(tmp_path),
            allowed_commands=["cat"],
        )

        assert "<name>internet-research</name>" in subagent._instructions
        assert "Use the bundled workflow" not in subagent._instructions
        assert subagent._tools.get("load_skill") is not None
        assert subagent._tools.get("bash") is not None

        loaded = await subagent._tools.execute(
            "load_skill", {"name": "internet-research"}
        )

        assert "Use the bundled workflow" in loaded.value
