from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class JsonRpcRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    jsonrpc: str = "2.0"
    id: int
    method: str
    params: dict[str, Any] = Field(default_factory=dict)


class JsonRpcNotification(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: dict[str, Any] | None = None


class JsonRpcError(BaseModel):
    code: int
    message: str
    data: Any = None


class JsonRpcResponse(BaseModel):
    jsonrpc: str
    id: int | None = None
    result: dict[str, Any] | None = None
    error: JsonRpcError | None = None

    def unwrap(self) -> dict[str, Any]:
        if self.error:
            raise RuntimeError(f"MCP error: {self.error.model_dump()}")
        return self.result or {}


class ClientInfo(BaseModel):
    name: str
    version: str


class InitializeParams(BaseModel):
    protocolVersion: str = "2024-11-05"
    capabilities: dict[str, Any] = Field(default_factory=dict)
    clientInfo: ClientInfo


class MCPServerStdioConfig(BaseModel):
    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    cache_tools_list: bool = True
    allowed_tools: list[str] | None = None


class MCPToolInputSchema(BaseModel):
    properties: dict[str, Any] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)


class MCPToolDefinition(BaseModel):
    name: str
    description: str | None = None
    inputSchema: MCPToolInputSchema = Field(default_factory=MCPToolInputSchema)


class MCPToolsListResult(BaseModel):
    tools: list[MCPToolDefinition] = Field(default_factory=list)
