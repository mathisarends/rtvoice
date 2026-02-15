import json
from enum import StrEnum
from typing import Annotated, Any, Literal, Self

from pydantic import BaseModel, Field, TypeAdapter, field_validator, model_validator

from rtvoice.views import RealtimeModel

# ============================================================================
# Enums
# ============================================================================


class RealtimeClientEvent(StrEnum):
    SESSION_UPDATE = "session.update"
    INPUT_AUDIO_BUFFER_APPEND = "input_audio_buffer.append"
    INPUT_AUDIO_BUFFER_COMMIT = "input_audio_buffer.commit"
    INPUT_AUDIO_BUFFER_CLEAR = "input_audio_buffer.clear"
    CONVERSATION_ITEM_CREATE = "conversation.item.create"
    CONVERSATION_ITEM_RETRIEVE = "conversation.item.retrieve"
    CONVERSATION_ITEM_TRUNCATE = "conversation.item.truncate"
    CONVERSATION_ITEM_DELETE = "conversation.item.delete"
    RESPONSE_CREATE = "response.create"
    RESPONSE_CANCEL = "response.cancel"
    TRANSCRIPTION_SESSION_UPDATE = "transcription_session.update"
    OUTPUT_AUDIO_BUFFER_CLEAR = "output_audio_buffer.clear"


class RealtimeServerEvent(StrEnum):
    SESSION_CREATED = "session.created"
    SESSION_UPDATED = "session.updated"
    TRANSCRIPTION_SESSION_CREATED = "transcription_session.created"
    TRANSCRIPTION_SESSION_UPDATED = "transcription_session.updated"
    CONVERSATION_CREATED = "conversation.created"
    CONVERSATION_DELETED = "conversation.deleted"
    CONVERSATION_ITEM_CREATED = "conversation.item.created"
    CONVERSATION_ITEM_ADDED = "conversation.item.added"
    CONVERSATION_ITEM_DONE = "conversation.item.done"
    CONVERSATION_ITEM_RETRIEVED = "conversation.item.retrieved"
    CONVERSATION_ITEM_TRUNCATED = "conversation.item.truncated"
    CONVERSATION_ITEM_DELETED = "conversation.item.deleted"
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_DELTA = (
        "conversation.item.input_audio_transcription.delta"
    )
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED = (
        "conversation.item.input_audio_transcription.completed"
    )
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_SEGMENT = (
        "conversation.item.input_audio_transcription.segment"
    )
    CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_FAILED = (
        "conversation.item.input_audio_transcription.failed"
    )
    INPUT_AUDIO_BUFFER_COMMITTED = "input_audio_buffer.committed"
    INPUT_AUDIO_BUFFER_CLEARED = "input_audio_buffer.cleared"
    INPUT_AUDIO_BUFFER_SPEECH_STARTED = "input_audio_buffer.speech_started"
    INPUT_AUDIO_BUFFER_SPEECH_STOPPED = "input_audio_buffer.speech_stopped"
    INPUT_AUDIO_BUFFER_TIMEOUT_TRIGGERED = "input_audio_buffer.timeout_triggered"
    RESPONSE_CREATED = "response.created"
    RESPONSE_DONE = "response.done"
    RESPONSE_OUTPUT_ITEM_ADDED = "response.output_item.added"
    RESPONSE_OUTPUT_ITEM_DONE = "response.output_item.done"
    RESPONSE_CONTENT_PART_ADDED = "response.content_part.added"
    RESPONSE_CONTENT_PART_DONE = "response.content_part.done"
    RESPONSE_OUTPUT_TEXT_DELTA = "response.output_text.delta"
    RESPONSE_OUTPUT_TEXT_DONE = "response.output_text.done"
    RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DELTA = "response.output_audio_transcript.delta"
    RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE = "response.output_audio_transcript.done"
    RESPONSE_OUTPUT_AUDIO_DELTA = "response.output_audio.delta"
    RESPONSE_OUTPUT_AUDIO_DONE = "response.output_audio.done"
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DELTA = "response.function_call_arguments.delta"
    RESPONSE_FUNCTION_CALL_ARGUMENTS_DONE = "response.function_call_arguments.done"
    MCP_CALL_ARGUMENTS_DELTA = "response.mcp_call_arguments.delta"
    MCP_CALL_ARGUMENTS_DONE = "response.mcp_call_arguments.done"
    MCP_LIST_TOOLS_IN_PROGRESS = "mcp_list_tools.in_progress"
    MCP_LIST_TOOLS_COMPLETED = "mcp_list_tools.completed"
    MCP_LIST_TOOLS_FAILED = "mcp_list_tools.failed"
    RESPONSE_MCP_CALL_IN_PROGRESS = "response.mcp_call.in_progress"
    RESPONSE_MCP_CALL_COMPLETED = "response.mcp_call.completed"
    RESPONSE_MCP_CALL_FAILED = "response.mcp_call.failed"
    RATE_LIMITS_UPDATED = "rate_limits.updated"
    ERROR = "error"


class TranscriptionModel(StrEnum):
    WHISPER_1 = "whisper-1"
    GPT_4O_TRANSCRIBE = "gpt-4o-transcribe"
    GPT_4O_MINI_TRANSCRIBE = "gpt-4o-mini-transcribe"


class NoiseReductionType(StrEnum):
    NEAR_FIELD = "near_field"
    FAR_FIELD = "far_field"


class AudioFormat(StrEnum):
    PCM16 = "pcm16"
    G711_ULAW = "g711_ulaw"
    G711_ALAW = "g711_alaw"


class OutputModality(StrEnum):
    TEXT = "text"
    AUDIO = "audio"


class MessageRole(StrEnum):
    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"


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


# ============================================================================
# Audio & Session Configuration
# ============================================================================


class AudioFormatConfig(BaseModel):
    type: AudioFormat = AudioFormat.PCM16


class InputAudioNoiseReductionConfig(BaseModel):
    type: NoiseReductionType


class TurnDetectionConfig(BaseModel):
    type: Literal["server_vad"] = "server_vad"
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 500


class InputAudioTranscriptionConfig(BaseModel):
    model: TranscriptionModel = TranscriptionModel.WHISPER_1


class AudioOutputConfig(BaseModel):
    format: AudioFormatConfig = Field(default_factory=AudioFormatConfig)
    speed: float = 1.0
    voice: str | None = None


class AudioInputConfig(BaseModel):
    format: AudioFormatConfig = Field(default_factory=AudioFormatConfig)
    turn_detection: TurnDetectionConfig | None = Field(
        default_factory=TurnDetectionConfig
    )
    transcription: InputAudioTranscriptionConfig | None = None
    noise_reduction: InputAudioNoiseReductionConfig | None = None


class AudioConfig(BaseModel):
    output: AudioOutputConfig = Field(default_factory=AudioOutputConfig)
    input: AudioInputConfig = Field(default_factory=AudioInputConfig)


# ============================================================================
# Tools Configuration
# ============================================================================


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


# ============================================================================
# Usage & Logging
# ============================================================================


class LogProbEntry(BaseModel):
    token: str
    logprob: float
    bytes_: list[int] | None = Field(default=None, alias="bytes")


class TokenInputTokenDetails(BaseModel):
    audio_tokens: int | None = None
    text_tokens: int | None = None


class TokenUsage(BaseModel):
    type: Literal["tokens"]
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    input_token_details: TokenInputTokenDetails | None = None


class DurationUsage(BaseModel):
    type: Literal["duration"]
    seconds: float


Usage = Annotated[TokenUsage | DurationUsage, Field(discriminator="type")]


# ============================================================================
# Conversation Items
# ============================================================================


class ConversationContent(BaseModel):
    type: Literal["output_text"]
    text: str


class MessageConversationItem(BaseModel):
    type: Literal["message"]
    role: MessageRole
    content: list[ConversationContent]


class FunctionCallOutputConversationItem(BaseModel):
    type: Literal["function_call_output"] = "function_call_output"
    call_id: str
    output: str


ConversationItem = MessageConversationItem | FunctionCallOutputConversationItem


# ============================================================================
# Session Configuration
# ============================================================================


class ResponseInstructions(BaseModel):
    instructions: str | None = None


class ErrorDetails(BaseModel):
    message: str
    type: str
    code: str | None = None
    event_id: str | None = None
    param: str | None = None


class RealtimeSessionConfig(BaseModel):
    type: Literal["realtime"] = "realtime"
    model: RealtimeModel = RealtimeModel.GPT_REALTIME
    instructions: str | None = None
    voice: str | None = None
    audio: AudioConfig = Field(default_factory=AudioConfig)
    include: list[str] | None = None
    max_output_tokens: int | Literal["inf"] = "inf"
    input_audio_noise_reduction: InputAudioNoiseReductionConfig | None = None
    output_modalities: list[OutputModality] = Field(
        default_factory=lambda: [OutputModality.AUDIO]
    )
    tool_choice: ToolChoice | ToolChoiceMode = ToolChoiceMode.AUTO
    tools: list[FunctionTool | MCPTool] | None = None


# ============================================================================
# Client Events (sent to OpenAI)
# ============================================================================


class InputAudioBufferAppendEvent(BaseModel):
    type: Literal[RealtimeClientEvent.INPUT_AUDIO_BUFFER_APPEND] = Field(
        default=RealtimeClientEvent.INPUT_AUDIO_BUFFER_APPEND
    )
    event_id: str | None = None
    audio: str


class ConversationItemCreateEvent(BaseModel):
    type: Literal[RealtimeClientEvent.CONVERSATION_ITEM_CREATE] = (
        RealtimeClientEvent.CONVERSATION_ITEM_CREATE
    )
    item: ConversationItem

    @classmethod
    def assistant_message(cls, text: str) -> Self:
        return cls(
            item=MessageConversationItem(
                type="message",
                role=MessageRole.ASSISTANT,
                content=[ConversationContent(type="output_text", text=text)],
            ),
        )

    @classmethod
    def function_call_output(cls, call_id: str, output: str) -> Self:
        return cls(
            item=FunctionCallOutputConversationItem(
                type="function_call_output",
                call_id=call_id,
                output=output,
            ),
        )


class ConversationItemTruncateEvent(BaseModel):
    event_id: str | None = None
    type: Literal[RealtimeClientEvent.CONVERSATION_ITEM_TRUNCATE] = Field(
        default=RealtimeClientEvent.CONVERSATION_ITEM_TRUNCATE
    )
    item_id: str
    content_index: int = 0
    audio_end_ms: int


class ConversationResponseCreateEvent(BaseModel):
    type: Literal[RealtimeClientEvent.RESPONSE_CREATE] = (
        RealtimeClientEvent.RESPONSE_CREATE
    )
    response: ResponseInstructions | None = None

    @classmethod
    def from_instructions(cls, text: str) -> Self:
        return cls(
            response=ResponseInstructions(instructions=text),
        )


class SessionUpdateEvent(BaseModel):
    type: Literal[RealtimeClientEvent.SESSION_UPDATE] = (
        RealtimeClientEvent.SESSION_UPDATE
    )
    event_id: str | None = None
    session: RealtimeSessionConfig


class ResponseCancelEvent(BaseModel):
    type: Literal[RealtimeClientEvent.RESPONSE_CANCEL] = (
        RealtimeClientEvent.RESPONSE_CANCEL
    )
    event_id: str | None = None


class OutputAudioBufferClearEvent(BaseModel):
    type: Literal[RealtimeClientEvent.OUTPUT_AUDIO_BUFFER_CLEAR] = (
        RealtimeClientEvent.OUTPUT_AUDIO_BUFFER_CLEAR
    )
    event_id: str | None = None


# ============================================================================
# Server Events (received from OpenAI)
# ============================================================================


class SessionCreatedEvent(BaseModel):
    type: Literal[RealtimeServerEvent.SESSION_CREATED] = (
        RealtimeServerEvent.SESSION_CREATED
    )
    event_id: str
    session: RealtimeSessionConfig


class SessionUpdatedEvent(BaseModel):
    type: Literal[RealtimeServerEvent.SESSION_UPDATED] = (
        RealtimeServerEvent.SESSION_UPDATED
    )
    event_id: str
    session: RealtimeSessionConfig


class ErrorEvent(BaseModel):
    type: Literal[RealtimeServerEvent.ERROR]
    event_id: str
    error: ErrorDetails


class ResponseOutputAudioDeltaEvent(BaseModel):
    type: Literal[RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_DELTA] = (
        RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_DELTA
    )
    event_id: str
    item_id: str
    response_id: str
    output_index: int
    content_index: int
    delta: str


class InputAudioTranscriptionDelta(BaseModel):
    type: Literal[RealtimeServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_DELTA]
    event_id: str
    item_id: str
    content_index: int
    delta: str
    logprobs: list[LogProbEntry] | None = None


class InputAudioTranscriptionCompleted(BaseModel):
    type: Literal[
        RealtimeServerEvent.CONVERSATION_ITEM_INPUT_AUDIO_TRANSCRIPTION_COMPLETED
    ]
    event_id: str
    item_id: str
    content_index: int
    transcript: str
    logprobs: list[LogProbEntry] | None = None
    usage: Usage | None = None


class ResponseOutputAudioTranscriptDelta(BaseModel):
    type: Literal[RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DELTA]
    event_id: str
    item_id: str
    response_id: str
    output_index: int
    content_index: int
    delta: str


class ResponseOutputAudioTranscriptDone(BaseModel):
    type: Literal[RealtimeServerEvent.RESPONSE_OUTPUT_AUDIO_TRANSCRIPT_DONE]
    event_id: str
    item_id: str
    response_id: str
    output_index: int
    content_index: int
    transcript: str


class ConversationItemTruncatedEvent(BaseModel):
    type: Literal[RealtimeServerEvent.CONVERSATION_ITEM_TRUNCATED] = (
        RealtimeServerEvent.CONVERSATION_ITEM_TRUNCATED
    )
    event_id: str
    item_id: str
    content_index: int
    audio_end_ms: int


class InputAudioBufferSpeechStartedEvent(BaseModel):
    type: Literal[RealtimeServerEvent.INPUT_AUDIO_BUFFER_SPEECH_STARTED] = (
        RealtimeServerEvent.INPUT_AUDIO_BUFFER_SPEECH_STARTED
    )
    event_id: str
    item_id: str
    audio_start_ms: int


class InputAudioBufferSpeechStoppedEvent(BaseModel):
    type: Literal[RealtimeServerEvent.INPUT_AUDIO_BUFFER_SPEECH_STOPPED] = (
        RealtimeServerEvent.INPUT_AUDIO_BUFFER_SPEECH_STOPPED
    )
    event_id: str
    item_id: str
    audio_end_ms: int


class ResponseCreatedEvent(BaseModel):
    type: Literal[RealtimeServerEvent.RESPONSE_CREATED] = (
        RealtimeServerEvent.RESPONSE_CREATED
    )
    event_id: str
    response_id: str


class ResponseDoneEvent(BaseModel):
    type: Literal[RealtimeServerEvent.RESPONSE_DONE] = RealtimeServerEvent.RESPONSE_DONE
    event_id: str
    response_id: str


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


# ============================================================================
# Server Event Union
# ============================================================================

ServerEvent = Annotated[
    SessionCreatedEvent
    | SessionUpdatedEvent
    | ErrorEvent
    | ResponseOutputAudioDeltaEvent
    | InputAudioTranscriptionDelta
    | InputAudioTranscriptionCompleted
    | ResponseOutputAudioTranscriptDelta
    | ResponseOutputAudioTranscriptDone
    | ConversationItemTruncatedEvent
    | InputAudioBufferSpeechStartedEvent
    | InputAudioBufferSpeechStoppedEvent
    | ResponseCreatedEvent
    | ResponseDoneEvent
    | FunctionCallItem,
    Field(discriminator="type"),
]

ServerEventAdapter = TypeAdapter(ServerEvent)
