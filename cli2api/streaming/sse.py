"""Server-Sent Events (SSE) encoding utilities."""

import json
from typing import Any


def sse_encode(data: Any) -> str:
    """Encode data as an SSE event.

    Args:
        data: Data to encode (dict or string).

    Returns:
        SSE-formatted string: "data: {...}\\n\\n"
    """
    if isinstance(data, dict):
        json_str = json.dumps(data, ensure_ascii=False)
    else:
        json_str = str(data)

    return f"data: {json_str}\n\n"


def sse_error(message: str, code: str = "server_error") -> str:
    """Encode an error as an SSE event.

    Args:
        message: Error message.
        code: Error code.

    Returns:
        SSE-formatted error string.
    """
    error_data = {
        "error": {
            "message": message,
            "type": "server_error",
            "code": code,
        }
    }
    return sse_encode(error_data)
