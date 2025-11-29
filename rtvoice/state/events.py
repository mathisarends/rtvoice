from enum import StrEnum


class VoiceAssistantEvent(StrEnum):
    WAKE_WORD_DETECTED = "wake_word_detected"
    USER_STARTED_SPEAKING = "user_started_speaking"
    USER_SPEECH_ENDED = "user_speech_ended"

    USER_TRANSCRIPT_CHUNK_RECEIVED = "user_transcript_chunk_received"
    USER_TRANSCRIPT_COMPLETED = "user_transcript_completed"

    AUDIO_CHUNK_RECEIVED = "audio_chunk_received"

    ASSISTANT_STARTED_RESPONDING = "assistant_started_responding"
    ASSISTANT_RESPONSE_COMPLETED = "assistant_response_completed"
    ASSISTANT_SPEECH_INTERRUPTED = "assistant_speech_interrupted"

    # these events have nothing to to with the audio streaming, but with the data which is sent and received
    ASSISTANT_STARTED_RESPONSE = "assistant_started_response"
    ASSISTANT_COMPLETED_RESPONSE = "assistant_completed_response"

    ASSISTANT_TRANSCRIPT_CHUNK_RECEIVED = "assistant_transcript_chunk_received"
    ASSISTANT_TRANSCRIPT_COMPLETED = "assistant_transcript_completed"

    ASSISTANT_STARTED_TOOL_CALL = "assistant_started_tool_call"
    ASSISTANT_RECEIVED_TOOL_CALL_RESULT = "assistant_received_tool_call"

    ASSISTANT_STARTED_MCP_TOOL_CALL = "assistant_started_mcp_tool_call"
    ASSISTANT_COMPLETED_MCP_TOOL_CALL_RESULT = "assistant_received_mcp_tool_call_result"
    ASSISTANT_FAILED_MCP_TOOL_CALL = "assistant_failed_mcp_tool_call"

    ASSISTANT_CONFIG_UPDATE_REQUEST = "assistant_config_update_request"

    TIMEOUT_OCCURRED = "timeout_occurred"
    IDLE_TRANSITION = "idle_transition"
    ERROR_OCCURRED = "error_occurred"
