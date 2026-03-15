from pathlib import Path

from rtvoice.skills.registry import SkillRegistry
from rtvoice.skills.service import Skill


def _write_skill_file(directory: Path, name: str, description: str, body: str) -> None:
    content = f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n"
    (directory / "SKILL.md").write_text(content, encoding="utf-8")


class TestAdd:
    def test_add_accepts_skill_instances(self) -> None:
        registry = SkillRegistry()
        inline = Skill.from_content("inline", "Do inline", summary="Inline summary")

        registry.add(inline)

        assert registry.available_names() == ["inline"]

    def test_add_accepts_path_and_string_path(self, tmp_path: Path) -> None:
        weather = tmp_path / "weather"
        weather.mkdir()
        _write_skill_file(weather, "weather", "Weather summary", "Weather instructions")

        stocks = tmp_path / "stocks"
        stocks.mkdir()
        _write_skill_file(stocks, "stocks", "Stocks summary", "Stocks instructions")

        registry = SkillRegistry()
        registry.add(weather, str(stocks))

        assert sorted(registry.available_names()) == ["stocks", "weather"]


class TestGet:
    def test_returns_skill_when_present(self) -> None:
        registry = SkillRegistry(
            skills=[
                Skill.from_content(
                    "weather", "Weather instructions", summary="Weather summary"
                )
            ]
        )

        found = registry.get("weather")

        assert found is not None
        assert found.name == "weather"

    def test_returns_none_when_skill_missing(self) -> None:
        registry = SkillRegistry()

        assert registry.get("missing") is None


class TestAvailableNames:
    def test_returns_names_in_registry_order(self) -> None:
        registry = SkillRegistry(
            skills=[
                Skill.from_content(
                    "weather", "Weather instructions", summary="Weather summary"
                ),
                Skill.from_content(
                    "stocks", "Stocks instructions", summary="Stocks summary"
                ),
            ]
        )

        assert registry.available_names() == ["weather", "stocks"]


class TestFormatIndex:
    def test_formats_skills_as_xml_like_index(self) -> None:
        registry = SkillRegistry(
            skills=[
                Skill.from_content(
                    "weather", "Weather instructions", summary="Weather summary"
                ),
                Skill.from_content(
                    "stocks", "Stocks instructions", summary="Stocks summary"
                ),
            ]
        )

        formatted = registry.format_index()

        assert formatted == (
            "<available_skills>\n"
            "  <skill name='weather'>Weather summary</skill>\n"
            "  <skill name='stocks'>Stocks summary</skill>\n"
            "</available_skills>"
        )


class TestFormatAllInstructions:
    def test_formats_all_skill_instruction_blocks(self) -> None:
        registry = SkillRegistry(
            skills=[
                Skill.from_content(
                    "weather", "Weather instructions", summary="Weather summary"
                ),
                Skill.from_content(
                    "stocks", "Stocks instructions", summary="Stocks summary"
                ),
            ]
        )

        formatted = registry.format_all_instructions()

        assert formatted == (
            "<skills>\n"
            "<skill name='weather'>\n"
            "Weather instructions\n"
            "</skill>\n\n"
            "<skill name='stocks'>\n"
            "Stocks instructions\n"
            "</skill>\n"
            "</skills>"
        )
