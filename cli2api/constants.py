"""Application constants.

Centralized location for magic numbers and strings used throughout the codebase.
"""

# ====================
# ID Generation
# ====================

# Length of hex suffix for generated IDs (24 hex chars = 12 bytes of randomness)
ID_HEX_LENGTH = 24

# ID prefixes for different entity types
CHAT_COMPLETION_ID_PREFIX = "chatcmpl-"
RESPONSE_ID_PREFIX = "resp-"
MESSAGE_ID_PREFIX = "msg-"
TOOL_CALL_ID_PREFIX = "call_"

# ====================
# Streaming
# ====================

# Maximum size of content chunks when splitting for streaming
# Chosen to balance between responsiveness and overhead
STREAM_CHUNK_MAX_SIZE = 150

# Preferred split points for chunking (in priority order)
CHUNK_SPLIT_SEPARATORS = (" ", "\n", ".", ",", ";")

# Minimum position ratio for split point (don't split too early)
CHUNK_SPLIT_MIN_RATIO = 0.5

# ====================
# HTTP Status Codes
# ====================

HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_INTERNAL_ERROR = 500
HTTP_GATEWAY_TIMEOUT = 504

# ====================
# OpenAI API
# ====================

# Default object types for OpenAI compatibility
OBJECT_CHAT_COMPLETION = "chat.completion"
OBJECT_CHAT_COMPLETION_CHUNK = "chat.completion.chunk"
OBJECT_MODEL = "model"
OBJECT_LIST = "list"

# Finish reasons
FINISH_REASON_STOP = "stop"
FINISH_REASON_TOOL_CALLS = "tool_calls"

# ====================
# Tool Call Markers
# ====================

TOOL_CALL_START_MARKER = "<tool_call>"
TOOL_CALL_END_MARKER = "</tool_call>"
