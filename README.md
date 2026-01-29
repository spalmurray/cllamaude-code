# cllamaude

Ollama-powered coding CLI assistant. Like Claude Code, but with local models.

## Setup

```bash
uv sync
```

## Usage

```bash
# Run with default model (glm4:latest)
uv run cllamaude

# Run with a specific model
uv run cllamaude -m llama3.1:latest
```

## Install globally

```bash
uv tool install -e .
cllamaude
```

## Tools

The assistant can:
- **read_file** - Read file contents
- **write_file** - Create/modify files (requires confirmation)
- **bash** - Run shell commands (requires confirmation)
