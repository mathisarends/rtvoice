from pydantic import BaseModel, ConfigDict, Field

from rtvoice.skills.bash import BashArgs


class ToolParams(BaseModel):
    # forbid extras so malformed model output fails validation instead of being
    # silently dropped
    model_config = ConfigDict(extra="forbid")


class LoadSkillParams(ToolParams):
    name: str = Field(description="Name of the skill to load, exactly as listed.")
    path: str | None = Field(
        default="SKILL.md",
        description=(
            "Relative path of a bundled resource to load instead of the skill's "
            "instructions. Defaults to the skill's SKILL.md."
        ),
    )


# reuses BashArgs fields so the runner and the tool schema share one definition
class BashParams(ToolParams, BashArgs):
    pass


class DoneParams(ToolParams):
    result: str = Field(description="The final result to return to the user.")


class ClarifyParams(ToolParams):
    question: str = Field(
        description="The clarifying question to ask the user.",
    )


class HandoffParams(ToolParams):
    task: str = Field(description="The task to hand off to the supervisor.")
    clarification_answer: str | None = Field(
        default=None,
        description=(
            "The user's answer to a previously asked clarification question. "
            "Only set this when resuming a paused supervisor task."
        ),
    )


class UpdateSupervisorParams(ToolParams):
    message: str = Field(
        description="New context, corrections, or instructions for the running supervisor.",
    )
