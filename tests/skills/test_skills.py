from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from rtvoice import RealtimeAgent, Skills, Subagent
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


class TestSkills:
    def test_discovery_discloses_metadata_but_not_instructions(
        self, tmp_path: Path
    ) -> None:
        make_skill(tmp_path)

        skills = Skills.from_local_dir(tmp_path)
        prompt = skills.discovery_prompt()

        assert "<name>internet-research</name>" in prompt
        assert "Research sources" in prompt
        assert "Use the bundled workflow" not in prompt
        assert skills.names() == ["internet-research"]

    def test_appends_discovery_prompt_to_system_prompt(self, tmp_path: Path) -> None:
        make_skill(tmp_path)
        skills = Skills.from_local_dir(tmp_path)

        system_prompt = skills.append_discovery_prompt("You are helpful.")

        assert system_prompt.startswith("You are helpful.")
        assert "<name>internet-research</name>" in system_prompt

    def test_load_skill_lists_relative_resources_without_reading_them(
        self, tmp_path: Path
    ) -> None:
        skill_dir = make_skill(tmp_path)
        references = skill_dir / "references"
        references.mkdir()
        (references / "guide.md").write_text("Only read me on demand.", "utf-8")

        skills = Skills.from_local_dir(tmp_path)
        result = skills.load("internet-research")

        assert "Use the bundled workflow" in result
        assert "<file>references/guide.md</file>" in result
        assert "Only read me on demand." not in result

    def test_read_resource_on_demand(self, tmp_path: Path) -> None:
        skill_dir = make_skill(tmp_path)
        references = skill_dir / "references"
        references.mkdir()
        (references / "guide.md").write_text("Loaded later.", encoding="utf-8")

        skills = Skills.from_local_dir(tmp_path)

        assert (
            skills.read_resource("internet-research", "references/guide.md")
            == "Loaded later."
        )

    def test_resource_path_cannot_escape_skill(self, tmp_path: Path) -> None:
        make_skill(tmp_path)
        (tmp_path / "secret.txt").write_text("secret", encoding="utf-8")
        skills = Skills.from_local_dir(tmp_path)

        with pytest.raises(ValueError, match="outside skill"):
            skills.read_resource("internet-research", "../secret.txt")

    def test_skill_md_is_not_readable_as_a_resource(self, tmp_path: Path) -> None:
        make_skill(tmp_path)
        skills = Skills.from_local_dir(tmp_path)

        with pytest.raises(ValueError, match="load_skill"):
            skills.read_resource("internet-research", "SKILL.md")

    def test_invalid_name_is_rejected(self, tmp_path: Path) -> None:
        make_skill(tmp_path, name="invalid--name")

        with pytest.raises(ValueError, match="without leading"):
            Skills.from_local_dir(tmp_path)

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
            Skills.from_local_dir(tmp_path)

    def test_description_is_xml_escaped(self, tmp_path: Path) -> None:
        make_skill(tmp_path, description="Use for A < B & current facts.")

        prompt = Skills.from_local_dir(tmp_path).discovery_prompt()

        assert "A &lt; B &amp; current facts" in prompt

    def test_later_source_overrides_duplicate(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        first = tmp_path / "first"
        second = tmp_path / "second"
        make_skill(first, description="First source.")
        make_skill(second, description="Second source.")

        skills = Skills.from_local_dir(first, second)

        assert "Second source." in skills.discovery_prompt()
        assert "First source." not in skills.discovery_prompt()
        assert any("overrides skill" in record.message for record in caplog.records)


def make_script(skill_dir: Path, body: str, name: str = "check.py") -> str:
    scripts = skill_dir / "scripts"
    scripts.mkdir(exist_ok=True)
    (scripts / name).write_text(body, encoding="utf-8")
    return f"scripts/{name}"


class TestSkillScripts:
    @pytest.mark.asyncio
    async def test_runs_bundled_script_with_arguments(self, tmp_path: Path) -> None:
        skill_dir = make_skill(tmp_path)
        path = make_script(skill_dir, "import sys\nprint(' '.join(sys.argv[1:]))\n")
        skills = Skills.from_local_dir(tmp_path)

        result = await skills.run_script("internet-research", path, ["hello", "world"])

        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_runs_script_in_the_skill_directory(self, tmp_path: Path) -> None:
        skill_dir = make_skill(tmp_path)
        path = make_script(skill_dir, "from pathlib import Path\nprint(Path.cwd())\n")
        skills = Skills.from_local_dir(tmp_path)

        result = await skills.run_script("internet-research", path)

        assert Path(result) == skill_dir.resolve()

    @pytest.mark.asyncio
    async def test_reports_failure_with_exit_code_and_stderr(
        self, tmp_path: Path
    ) -> None:
        skill_dir = make_skill(tmp_path)
        path = make_script(skill_dir, "import sys\nsys.exit('boom')\n")
        skills = Skills.from_local_dir(tmp_path)

        result = await skills.run_script("internet-research", path)

        assert result.startswith("Error (exit code 1)")
        assert "boom" in result

    @pytest.mark.asyncio
    async def test_script_path_cannot_escape_skill(self, tmp_path: Path) -> None:
        make_skill(tmp_path)
        (tmp_path / "outside.py").write_text("print('leaked')", encoding="utf-8")
        skills = Skills.from_local_dir(tmp_path)

        with pytest.raises(ValueError, match="outside skill"):
            await skills.run_script("internet-research", "../outside.py")


_SKILL_TOOLS = {"load_skill", "read_skill_resource", "run_skill_script"}


class TestDefaultTools:
    def test_skill_defaults_are_hidden_without_dependencies(self) -> None:
        tools = Tools()

        exposed = {tool.name for tool in tools.get_schema()}

        assert exposed.isdisjoint(_SKILL_TOOLS)

    def test_skill_defaults_are_exposed_once_injected(self, tmp_path: Path) -> None:
        make_skill(tmp_path)
        skills = Skills.from_local_dir(tmp_path)
        tools = Tools()
        tools.set_context(ToolContext(skills))

        exposed = {tool.name for tool in tools.get_schema()}

        assert exposed >= _SKILL_TOOLS

    def test_merge_keeps_the_receivers_defaults(self) -> None:
        tools = Tools()

        tools.merge(Tools())

        assert tools.get("load_skill") is not None

    @pytest.mark.asyncio
    async def test_load_skill_uses_injected_skills(self, tmp_path: Path) -> None:
        make_skill(tmp_path)
        skills = Skills.from_local_dir(tmp_path)
        tools = Tools()
        tools.set_context(ToolContext(skills))

        loaded = await tools.execute("load_skill", {"name": "internet-research"})

        assert "Use the bundled workflow" in loaded.value

    @pytest.mark.asyncio
    async def test_resource_and_script_tools_reach_skills(self, tmp_path: Path) -> None:
        skill_dir = make_skill(tmp_path)
        (skill_dir / "notes.md").write_text("Read me.", encoding="utf-8")
        script = make_script(skill_dir, "import sys\nprint(sys.argv[1].upper())\n")
        tools = Tools()
        tools.set_context(ToolContext(Skills.from_local_dir(tmp_path)))

        resource = await tools.execute(
            "read_skill_resource", {"name": "internet-research", "path": "notes.md"}
        )
        ran = await tools.execute(
            "run_skill_script",
            {"name": "internet-research", "path": script, "args": ["ok"]},
        )

        assert resource.value == "Read me."
        assert ran.value == "OK"


class TestAgentIntegration:
    def test_realtime_agent_gets_catalog_and_builtin_tools(
        self, tmp_path: Path
    ) -> None:
        make_skill(tmp_path)
        audio_input = MagicMock()
        audio_output = MagicMock()

        with patch("rtvoice.agent.realtime_agent.OpenAIProvider"):
            agent = RealtimeAgent(
                system_prompt="You are helpful.",
                skills=Skills.from_local_dir(tmp_path),
                audio_input=audio_input,
                audio_output=audio_output,
            )

        system_prompt = agent._realtime_session._instructions
        assert system_prompt.startswith("You are helpful.")
        assert "<name>internet-research</name>" in system_prompt
        assert "Use the bundled workflow" not in system_prompt
        assert {tool.name for tool in agent._tools.get_schema()} >= _SKILL_TOOLS

    @pytest.mark.asyncio
    async def test_subagent_loads_skill_progressively(self, tmp_path: Path) -> None:
        make_skill(tmp_path)
        subagent = Subagent(
            description="Research subagent",
            system_prompt="Be accurate.",
            llm=MagicMock(),
            skills=Skills.from_local_dir(tmp_path),
        )

        assert "<name>internet-research</name>" in subagent._system_prompt
        assert "Use the bundled workflow" not in subagent._system_prompt
        assert {tool.name for tool in subagent._tools.get_schema()} >= _SKILL_TOOLS

        loaded = await subagent._tools.execute(
            "load_skill", {"name": "internet-research"}
        )

        assert "Use the bundled workflow" in loaded.value
