from pydantic import BaseModel, ConfigDict, Field


class ToolParams(BaseModel):
    # forbid extras so malformed model output fails validation instead of being
    # silently dropped
    model_config = ConfigDict(extra="forbid")


class SkillParams(ToolParams):
    name: str = Field(description="Name of the skill, exactly as listed.")


class LoadSkillParams(SkillParams):
    pass


class ReadSkillResourceParams(SkillParams):
    path: str = Field(
        description="Path of the bundled file, relative to the skill directory."
    )


class RunSkillScriptParams(SkillParams):
    path: str = Field(
        description="Path of the bundled script, relative to the skill directory."
    )
    args: list[str] = Field(
        default_factory=list, description="Arguments passed to the script."
    )
    timeout: int = Field(
        default=60,
        ge=1,
        le=300,
        description="Timeout in seconds. Defaults to 60; maximum 300.",
    )


class SubagentParams(ToolParams):
    task: str = Field(description="The task to hand off to the subagent.")
