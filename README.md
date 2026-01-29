# CLI2API

OpenAI-compatible API over Claude Code CLI.

## Why CLI2API?

Claude Code CLI is powerful but not all tools support it directly. CLI2API bridges this gap by exposing Claude Code as an OpenAI-compatible API.

**Use Cases:**

- **IDE Integration** — Connect Kilo Code, Roo Code, Cursor, or other editors that support OpenAI API
- **Custom Applications** — Build apps using familiar OpenAI SDK instead of spawning CLI processes
- **Team Sharing** — Run one CLI2API server and share Claude access across your team
- **Tool Compatibility** — Use Claude with any tool that supports OpenAI API format (LangChain, AutoGPT, etc.)

## Requirements

- Python 3.11+
- [Claude Code CLI](https://docs.anthropic.com/en/docs/claude-code) installed and authenticated

## Installation

### Quick Start (pip)

```bash
pip install cli2api
cli2api
```

Server starts at http://localhost:8000

### From Source

```bash
git clone https://github.com/anoxis/CLI2API.git
cd CLI2API
pip install -e .
cli2api
```

### Docker

Docker requires the repository to be cloned first:

```bash
# Clone and build
git clone https://github.com/anoxis/CLI2API.git
cd CLI2API
docker build -t cli2api .

# Run with Claude CLI mounted from host
docker run -p 8000:8000 \
  -v $(which claude):/usr/local/bin/claude:ro \
  -e CLI2API_CLAUDE_CLI_PATH=/usr/local/bin/claude \
  cli2api
```

Or use docker-compose:

```bash
# Edit docker-compose.yaml to mount your Claude CLI path
docker-compose up -d
```

### Development

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

## IDE Integration

### Kilo Code / Roo Code

1. Start CLI2API server: `cli2api`
2. Open extension settings
3. Add custom provider:
   - **Provider Name:** `CLI2API` (or any name)
   - **API Base URL:** `http://localhost:8000/v1`
   - **API Key:** `not-needed` (any non-empty value)
   - **Model:** `sonnet` (or `opus`, `haiku`)

### Cursor / Continue

1. Start CLI2API server
2. Settings → Models → Add Model
3. Configure:
   - **API Base:** `http://localhost:8000/v1`
   - **API Key:** `any-value`
   - **Model:** `sonnet`

### Generic OpenAI-compatible client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="sonnet",
    messages=[{"role": "user", "content": "Hello!"}]
)
print(response.choices[0].message.content)
```

## Usage

### Test with curl

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/v1/models

# Chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "sonnet", "messages": [{"role": "user", "content": "Hello"}]}'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "sonnet", "messages": [{"role": "user", "content": "Hello"}], "stream": true}'
```

## Configuration

Environment variables (prefix `CLI2API_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `0.0.0.0` | Server host |
| `PORT` | `8000` | Server port |
| `CLAUDE_CLI_PATH` | auto-detect | Path to Claude CLI executable |
| `DEFAULT_TIMEOUT` | `300` | CLI execution timeout (seconds) |
| `DEFAULT_MODEL` | `sonnet` | Default model |
| `CLAUDE_MODELS` | `sonnet,opus,haiku` | Available models (comma-separated) |
| `LOG_LEVEL` | `INFO` | Logging level |
| `LOG_JSON` | `false` | JSON format logging |
| `DEBUG` | `false` | Debug mode |

Copy `.env.example` to `.env` for local configuration.

## Available Models

| Model ID | Description |
|----------|-------------|
| `sonnet` | Claude Sonnet (default) |
| `opus` | Claude Opus |
| `haiku` | Claude Haiku |

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /` | API info |
| `GET /v1/models` | List available models |
| `GET /v1/models/{id}` | Get model info |
| `POST /v1/chat/completions` | Chat completions (OpenAI compatible) |
| `POST /v1/responses` | Responses API (OpenAI compatible) |

## Security Notice

CLI2API does **not** implement authentication. It is designed for **local use only**.

- Do not expose to the internet without additional security measures
- Use behind a reverse proxy with authentication if network access is needed
- The API inherits permissions from the Claude CLI authentication on the host

## License

MIT
