from __future__ import annotations

import json
from enum import StrEnum
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    Self,
)

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from rtvoice.events import EventBus
from rtvoice.events.schemas.base import RealtimeServerEvent
from rtvoice.events.schemas.conversation import ConversationItemCreateEvent
from rtvoice.sound.player import AudioPlayer

if TYPE_CHECKING:
    from rtvoice.config.models import VoiceSettings


class JsonType(StrEnum):
    OBJECT = "object"
    ARRAY = "array"
    STRING = "string"
    NUMBER = "number"
    INTEGER = "integer"
    BOOLEAN = "boolean"


class ToolChoiceMode(StrEnum):
    NONE = "none"
    AUTO = "auto"
    REQUIRED = "required"


class MCPConnectorId(StrEnum):
    DROPBOX = "connector_dropbox"
    GMAIL = "connector_gmail"
    GOOGLE_CALENDAR = "connector_googlecalendar"
    GOOGLE_DRIVE = "connector_googledrive"
    MICROSOFT_TEAMS = "connector_microsoftteams"
    OUTLOOK_CALENDAR = "connector_outlookcalendar"
    OUTLOOK_EMAIL = "connector_outlookemail"
    SHAREPOINT = "connector_sharepoint"


class MCPRequireApprovalMode(StrEnum):
    NEVER = "never"
    AUTO = "auto"
    ALWAYS = "always"
    FIRST_USE = "first_use"


class FunctionParameterProperty(BaseModel):
    type: JsonType
    description: str | None = None
    items: Self | None = None
    enum: list[str] | None = None
    properties: dict[str, Self] | None = None
    required: list[str] | None = None
    min_items: int | None = Field(None, alias="minItems")
    default: Any | None = None


class FunctionParameters(BaseModel):
    type: str = "object"
    strict: bool = True
    properties: dict[str, FunctionParameterProperty] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class FunctionTool(BaseModel):
    type: Literal["function"] = "function"
    name: str
    description: str | None = None
    parameters: FunctionParameters


class MCPToolFilter(BaseModel):
    patterns: list[str] | None = None
    exclude: list[str] | None = None


class MCPRequireApproval(BaseModel):
    tools: list[str] | None = None
    all: bool | None = None


class MCPTool(BaseModel):
    type: Literal["mcp"] = "mcp"
    server_label: str
    allowed_tools: list[str] | MCPToolFilter | None = None
    authorization: str | None = None
    connector_id: MCPConnectorId | str | None = None
    headers: dict[str, Any] | None = None
    require_approval: MCPRequireApproval | str | None = None
    server_description: str | None = None
    server_url: str | None = None

    @model_validator(mode="after")
    def validate_server_config(self) -> Self:
        if not self.server_url and not self.connector_id:
            raise ValueError("Either 'server_url' or 'connector_id' must be provided")
        return self


class ToolChoice(BaseModel):
    mode: ToolChoiceMode = ToolChoiceMode.AUTO
    function: FunctionTool | None = None
    mcp: MCPTool | None = None


class FunctionCallItem(BaseModel):
    type: Literal[RealtimeServerEvent.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE] = (
        RealtimeServerEvent.RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE
    )

    name: str | None = None
    call_id: str
    event_id: str
    item_id: str
    output_index: int
    response_id: str
    arguments: dict[str, Any]

    @field_validator("arguments", mode="before")
    @classmethod
    def parse_arguments(cls, v: Any) -> dict[str, Any]:
        if isinstance(v, dict):
            return v
        if isinstance(v, str):
            if not v.strip():
                return {}
            try:
                return json.loads(v)
            except json.JSONDecodeError:
                return {"__raw__": v}
        raise TypeError("arguments must be a dict or a JSON string")


class FunctionCallResult(BaseModel):
    tool_name: str
    call_id: str
    output: Any | None = None
    response_instruction: str | None = None

    def to_conversation_item(self) -> ConversationItemCreateEvent:
        return ConversationItemCreateEvent.function_call_output(
            call_id=self.call_id,
            output=self._format_output(),
        )

    def _format_output(self) -> str:
        if self.output is None:
            return ""
        if isinstance(self.output, str):
            return self.output
        try:
            return json.dumps(self.output, ensure_ascii=False)
        except (TypeError, ValueError):
            return str(self.output)


class SpecialToolParameters(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    audio_player: AudioPlayer
    event_bus: EventBus
    voice_settings: VoiceSettings
    tool_calling_model_name: str | None = None
