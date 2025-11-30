from mcp import ClientSession, Tool

from rtvoice.shared.logging_mixin import LoggingMixin
from rtvoice.tools.models import (
    FunctionParameterProperty,
    FunctionParameters,
    FunctionTool,
    JsonType,
)


class MCPToolConverter(LoggingMixin):
    async def convert_from_session(
        self, session: ClientSession, server_name: str
    ) -> list[FunctionTool]:
        tools_list = await session.list_tools()

        function_tools = []
        for tool in tools_list.tools:
            function_tool = self._convert_tool(tool, server_name)
            function_tools.append(function_tool)
            self.logger.info(f"Converted MCP tool '{tool.name}' from {server_name}")

        return function_tools

    def _convert_tool(self, mcp_tool: Tool, server_name: str) -> FunctionTool:
        input_schema = mcp_tool.inputSchema

        properties = self._parse_properties(input_schema)
        required = input_schema.get("required", [])

        parameters = FunctionParameters(
            type="object",
            properties=properties,
            required=required,
        )

        tool_name = f"{server_name}__{mcp_tool.name}"

        return FunctionTool(
            type="function",
            name=tool_name,
            description=mcp_tool.description or "",
            parameters=parameters,
        )

    def _parse_properties(self, schema: dict) -> dict[str, FunctionParameterProperty]:
        properties: dict[str, FunctionParameterProperty] = {}

        schema_properties = schema.get("properties", {})

        for prop_name, prop_schema in schema_properties.items():
            properties[prop_name] = self._parse_property(prop_schema)

        return properties

    def _parse_property(self, prop_schema: dict) -> FunctionParameterProperty:
        json_type = self._map_json_type(prop_schema.get("type", "string"))

        prop = FunctionParameterProperty(
            type=json_type,
            description=prop_schema.get("description"),
            enum=prop_schema.get("enum"),
            default=prop_schema.get("default"),
        )

        if json_type == JsonType.ARRAY and "items" in prop_schema:
            items_schema = prop_schema["items"]
            prop.items = self._parse_property(items_schema)

            # minItems
            if "minItems" in prop_schema:
                prop.min_items = prop_schema["minItems"]

        if json_type == JsonType.OBJECT and "properties" in prop_schema:
            nested_props = {}
            for nested_name, nested_schema in prop_schema["properties"].items():
                nested_props[nested_name] = self._parse_property(nested_schema)
            prop.properties = nested_props
            prop.required = prop_schema.get("required", [])

        return prop

    def _map_json_type(self, type_str: str) -> JsonType:
        type_mapping = {
            "string": JsonType.STRING,
            "number": JsonType.NUMBER,
            "integer": JsonType.INTEGER,
            "boolean": JsonType.BOOLEAN,
            "array": JsonType.ARRAY,
            "object": JsonType.OBJECT,
        }
        return type_mapping.get(type_str.lower(), JsonType.STRING)
