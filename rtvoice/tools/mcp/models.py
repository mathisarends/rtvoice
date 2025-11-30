from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class MCPServerType(StrEnum):
    STDIO = "stdio"
    REMOTE = "remote"


class MCPServerConfig(BaseModel):
    name: str
    command: str
    args: list[str] = []
    env: dict[str, str] | None = None


class MCPToolMetadata(BaseModel):
    server_name: str
    tool_name: str
    description: str
    input_schema: dict[str, Any]
