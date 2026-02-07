"""Tool call argument validation against OpenAI tool schemas."""

import json
from typing import Optional

from cli2api.utils.logging import get_logger

logger = get_logger(__name__)


def build_required_params_index(tools: list[dict]) -> dict[str, list[str]]:
    """Build a lookup: tool_name -> list of required parameter names.

    Args:
        tools: OpenAI-format tool definitions.

    Returns:
        Dict mapping tool name to list of required param names.
    """
    index = {}
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = tool.get("function", {})
        name = func.get("name")
        if not name:
            continue
        params = func.get("parameters", {})
        index[name] = params.get("required", [])
    return index


def validate_tool_call(
    tool_call: dict,
    required_params_index: dict[str, list[str]],
) -> bool:
    """Validate a single tool call has all required arguments.

    Args:
        tool_call: Parsed tool call in OpenAI format.
        required_params_index: From build_required_params_index().

    Returns:
        True if valid (all required params present), False otherwise.
    """
    func = tool_call.get("function", {})
    name = func.get("name", "")
    arguments_str = func.get("arguments", "{}")

    required = required_params_index.get(name)
    if required is None:
        # Unknown tool — can't validate, let it pass
        return True

    if not required:
        # Tool has no required params — always valid
        return True

    try:
        arguments = json.loads(arguments_str)
    except (json.JSONDecodeError, TypeError):
        logger.warning(f"Tool call '{name}' has unparseable arguments: {arguments_str!r}")
        return False

    if not isinstance(arguments, dict):
        logger.warning(f"Tool call '{name}' arguments is not a dict: {type(arguments)}")
        return False

    missing = [p for p in required if p not in arguments]
    if missing:
        logger.warning(
            f"Tool call '{name}' missing required params: {missing}. "
            f"Got: {list(arguments.keys())}"
        )
        return False

    return True


def filter_valid_tool_calls(
    tool_calls: Optional[list[dict]],
    tools: Optional[list[dict]],
) -> Optional[list[dict]]:
    """Filter tool calls, removing those with missing required params.

    Invalid tool calls are dropped with a WARNING log.

    Args:
        tool_calls: Parsed tool calls (may be None).
        tools: Original tool definitions (may be None).

    Returns:
        Filtered list, or None if empty/input was None.
    """
    if not tool_calls or not tools:
        return tool_calls

    index = build_required_params_index(tools)
    valid = [tc for tc in tool_calls if validate_tool_call(tc, index)]

    if len(valid) < len(tool_calls):
        dropped = len(tool_calls) - len(valid)
        logger.warning(f"Dropped {dropped} invalid tool call(s) with missing required params")

    return valid if valid else None
