from dataclasses import dataclass

from rtvoice.agent.system_prompt import SystemPrompt


@dataclass(frozen=True, slots=True)
class SkillInfo:
    name: str
    description: str


class TestSystemPrompt:
    def test_preserves_content_without_skills(self) -> None:
        prompt = SystemPrompt("You are helpful.")

        assert str(prompt) == "You are helpful."

    def test_appends_skill_discovery(self) -> None:
        prompt = SystemPrompt(
            "You are helpful.",
            skills=[SkillInfo("internet-research", "Research current sources.")],
        )

        content = str(prompt)
        assert content.startswith("You are helpful.\n\n## Available Skills")
        assert "<name>internet-research</name>" in content
        assert "<description>Research current sources.</description>" in content

    def test_builds_discovery_without_base_content(self) -> None:
        prompt = SystemPrompt(
            "",
            skills=[SkillInfo("internet-research", "Research current sources.")],
        )

        assert str(prompt).startswith("## Available Skills")

    def test_escapes_skill_metadata(self) -> None:
        prompt = SystemPrompt(
            "",
            skills=[SkillInfo("research", "Use for A < B & current facts.")],
        )

        assert "A &lt; B &amp; current facts." in str(prompt)
