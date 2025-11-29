from enum import StrEnum


class StateType(StrEnum):
    IDLE = "idle"
    TIMEOUT = "timeout"
    LISTENING = "listening"
    RESPONDING = "responding"
    TOOL_CALLING = "tool_calling"
    ERROR = "error"
