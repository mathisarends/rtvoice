from rtvoice.realtime.schemas import (
    FunctionCallConversationItem,
    McpToolCallConversationItem,
    MessageConversationItem,
    OutputAudioConversationContent,
    RealtimeResponseObject,
)


def test_response_output_parses_typed_items() -> None:
    response = RealtimeResponseObject.model_validate(
        {
            "id": "response-1",
            "output": [
                {
                    "id": "message-1",
                    "type": "message",
                    "status": "completed",
                    "role": "assistant",
                    "content": [
                        {
                            "type": "output_audio",
                            "transcript": "Hello!",
                        }
                    ],
                },
                {
                    "id": "call-1",
                    "type": "function_call",
                    "status": "completed",
                    "name": "lookup",
                    "call_id": "call-id-1",
                    "arguments": '{"city":"Berlin"}',
                },
                {
                    "id": "mcp-call-1",
                    "type": "mcp_call",
                    "name": "search",
                    "server_label": "docs",
                    "arguments": '{"query":"Realtime"}',
                    "output": "result",
                },
            ],
        }
    )

    message, function_call, mcp_call = response.output
    assert isinstance(message, MessageConversationItem)
    assert isinstance(message.content[0], OutputAudioConversationContent)
    assert isinstance(function_call, FunctionCallConversationItem)
    assert isinstance(mcp_call, McpToolCallConversationItem)


def test_function_call_ids_only_returns_present_call_ids() -> None:
    response = RealtimeResponseObject.model_validate(
        {
            "id": "response-1",
            "output": [
                {
                    "type": "function_call",
                    "name": "first",
                    "call_id": "call-1",
                    "arguments": "{}",
                },
                {
                    "type": "function_call",
                    "name": "second",
                    "arguments": "{}",
                },
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Done"}],
                },
            ],
        }
    )

    assert response.function_call_ids == ["call-1"]
