from pathlib import Path

import pytest

from rtvoice.skills.service import Skill


def _write_skill_file(directory: Path, name: str, description: str, body: str) -> None:
    content = f"---\nname: {name}\ndescription: {description}\n---\n\n{body}\n"
    (directory / "SKILL.md").write_text(content, encoding="utf-8")


class TestFromContent:
    def test_uses_inline_name_summary_and_instructions(self) -> None:
        skill = Skill.from_content(
            name="inline",
            summary="Inline summary",
            instructions="Do the inline thing.",
        )

        assert skill.name == "inline"
        assert skill.summary == "Inline summary"
        assert skill.instructions == "Do the inline thing."


class TestFromDir:
    def test_loads_name_summary_and_instructions(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "weather"
        skill_dir.mkdir()
        _write_skill_file(
            directory=skill_dir,
            name="weather",
            description="Weather helper",
            body="Use this skill to answer weather questions.",
        )

        skill = Skill.from_dir(skill_dir)

        assert skill.name == "weather"
        assert skill.summary == "Weather helper"
        assert skill.instructions == "Use this skill to answer weather questions."

    def test_raises_when_directory_does_not_exist(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Skill path does not exist"):
            Skill.from_dir(tmp_path / "missing")

    def test_raises_when_frontmatter_missing(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "weather"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("No YAML frontmatter", encoding="utf-8")

        with pytest.raises(ValueError, match="missing YAML frontmatter"):
            Skill.from_dir(skill_dir)

    def test_raises_when_frontmatter_malformed(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "weather"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: weather\n", encoding="utf-8")

        with pytest.raises(ValueError, match="has malformed frontmatter"):
            Skill.from_dir(skill_dir)

    def test_raises_when_required_fields_missing(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "weather"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: weather\n---\n\nBody", encoding="utf-8"
        )

        with pytest.raises(ValueError, match="missing required frontmatter fields"):
            Skill.from_dir(skill_dir)

    def test_raises_when_name_does_not_match_directory(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "weather"
        skill_dir.mkdir()
        _write_skill_file(
            directory=skill_dir,
            name="stocks",
            description="Wrong name",
            body="Body",
        )

        with pytest.raises(ValueError, match="must match directory name"):
            Skill.from_dir(skill_dir)


class TestDiscover:
    def test_discovers_all_nested_skill_directories(self, tmp_path: Path) -> None:
        first = tmp_path / "weather"
        first.mkdir(parents=True)
        _write_skill_file(first, "weather", "Weather helper", "Body one")

        second = tmp_path / "nested" / "stocks"
        second.mkdir(parents=True)
        _write_skill_file(second, "stocks", "Stocks helper", "Body two")

        discovered = Skill.discover(tmp_path)

        assert sorted(skill.name for skill in discovered) == ["stocks", "weather"]


class TestInternals:
    def test_read_raw_raises_when_skill_file_missing(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "weather"
        skill_dir.mkdir()
        skill = Skill(path=skill_dir)

        with pytest.raises(FileNotFoundError, match=r"No SKILL\.md found"):
            skill._read_raw()

    def test_strip_frontmatter_returns_body(self) -> None:
        raw = "---\nname: weather\ndescription: Weather helper\n---\n\nUse weather skill.\n"

        assert Skill._strip_frontmatter(raw) == "Use weather skill."

    def test_strip_frontmatter_returns_trimmed_raw_when_no_frontmatter(self) -> None:
        assert Skill._strip_frontmatter("  plain text  ") == "plain text"
