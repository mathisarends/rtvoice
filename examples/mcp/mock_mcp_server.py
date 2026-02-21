# mock_mcp_server.py
import json


def send(msg):
    print(json.dumps(msg), flush=True)


def recv():
    return json.loads(input())


while True:
    req = recv()
    method = req.get("method")
    msg_id = req.get("id")

    if method == "initialize":
        send(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {"protocolVersion": "2024-11-05", "capabilities": {}},
            }
        )

    elif method == "notifications/initialized":
        pass  # notification, keine response nötig

    elif method == "tools/list":
        send(
            {
                "jsonrpc": "2.0",
                "id": msg_id,
                "result": {
                    "tools": [
                        {
                            "name": "greet",
                            "description": "Says hello",
                            "inputSchema": {
                                "properties": {
                                    "name": {"type": "string", "description": "Name"}
                                },
                                "required": ["name"],
                            },
                        },
                        {
                            "name": "add",
                            "description": "Adds two numbers",
                            "inputSchema": {
                                "properties": {
                                    "a": {"type": "number"},
                                    "b": {"type": "number"},
                                },
                                "required": ["a", "b"],
                            },
                        },
                        {
                            "name": "secret_tool",
                            "description": "Should be filtered out",
                            "inputSchema": {"properties": {}, "required": []},
                        },
                    ]
                },
            }
        )

    elif method == "tools/call":
        name = req["params"]["name"]
        args = req["params"].get("arguments", {})
        if name == "greet":
            result = {"content": f"Hello, {args['name']}!"}
        elif name == "add":
            result = {"content": args["a"] + args["b"]}
        else:
            result = {"content": "unknown tool"}
        send({"jsonrpc": "2.0", "id": msg_id, "result": result})
