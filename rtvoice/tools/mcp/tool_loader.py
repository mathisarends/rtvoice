from typing import Any

from mcp import ClientSession, StdioServerParameters, Tool, stdio_client

from rtvoice.shared.logging_mixin import LoggingMixin
from rtvoice.tools.mcp.models import MCPServerConfig
from rtvoice.tools.models import (
    FunctionParameterProperty,
    FunctionParameters,
    FunctionTool,
    JsonType,
)


class MCPToolLoader(LoggingMixin):
    def __init__(self, server_config: MCPServerConfig):
        self.server_config = server_config

    async def load_tools(self) -> list[FunctionTool]:
        server_params = StdioServerParameters(
            command=self.server_config.command,
            args=self.server_config.args,
            env=self.server_config.env,
        )

        tools = []

        async with (
            stdio_client(server_params) as (read, write),
            ClientSession(read, write) as session,
        ):
            await session.initialize()

            tools_list = await session.list_tools()

            for tool in tools_list.tools:
                function_tool = self._convert_to_function_tool(tool)
                tools.append(function_tool)
                self.logger.info(
                    f"Loaded MCP tool '{tool.name}' from {self.server_config.name}"
                )

        return tools

    def _convert_to_function_tool(self, mcp_tool: Tool) -> FunctionTool:
        input_schema: dict[str, Any] = mcp_tool.inputSchema

        properties = self._parse_properties(input_schema)
        required = input_schema.get("required", [])

        parameters = FunctionParameters(
            type="object",
            properties=properties,
            required=required,
        )

        tool_name = f"{self.server_config.name}__{mcp_tool.name}"

        return FunctionTool(
            type="function",
            name=tool_name,
            description=mcp_tool.description or "",
            parameters=parameters,
        )

    def _parse_properties(
        self, schema: dict[str, Any]
    ) -> dict[str, FunctionParameterProperty]:
        properties: dict[str, FunctionParameterProperty] = {}

        schema_properties = schema.get("properties", {})

        for prop_name, prop_schema in schema_properties.items():
            json_type = self._map_json_type(prop_schema.get("type", "string"))
            description = prop_schema.get("description")

            properties[prop_name] = FunctionParameterProperty(
                type=json_type,
                description=description,
            )

        return properties

    def _map_json_type(self, type_str: str) -> JsonType:
        type_mapping: dict[str, JsonType] = {
            "string": JsonType.STRING,
            "number": JsonType.NUMBER,
            "integer": JsonType.INTEGER,
            "boolean": JsonType.BOOLEAN,
            "array": JsonType.ARRAY,
            "object": JsonType.OBJECT,
        }
        return type_mapping.get(type_str.lower(), JsonType.STRING)
