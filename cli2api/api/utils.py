"""Shared utilities for API endpoints."""


def parse_model_name(model_id: str) -> str:
    """Extract actual model name from prefixed format.

    Args:
        model_id: Model ID like "claude: sonnet" or "sonnet".

    Returns:
        Actual model name like "sonnet".
    """
    if ": " in model_id:
        return model_id.split(": ", 1)[1]
    return model_id
