"""Ollama LLM client wrapper."""

import ollama
from pathlib import Path
from typing import Generator

from .tools import TOOLS


def load_file_if_exists(path: Path) -> str | None:
    """Load a file if it exists."""
    if path.exists():
        try:
            return path.read_text()
        except Exception:
            pass
    return None


def load_claude_md() -> str | None:
    """Load ~/.claude/CLAUDE.md if it exists."""
    return load_file_if_exists(Path.home() / ".claude" / "CLAUDE.md")


def load_project_instructions(cwd: str) -> str | None:
    """Load project CLAUDE.md or AGENTS.md if they exist."""
    cwd_path = Path(cwd)
    for name in ["CLAUDE.md", "AGENTS.md"]:
        content = load_file_if_exists(cwd_path / name)
        if content:
            return content
    return None


def get_system_prompt(cwd: str) -> str:
    """Generate the system prompt with context."""
    prompt = f"""You are an expert coding assistant running in a terminal.

Current working directory: {cwd}

You have access to the following tools:
- read_file(path): Read the contents of a file
- write_file(path, content): Write content to a file
- bash(command): Execute a bash command

Use these tools to help the user with coding tasks.

## Important Rules

- ALWAYS read a file before modifying it. Never assume you know what's in a file.
- When editing files, preserve existing code. Only change what's necessary for the task.
- Don't create new files in new directories unless explicitly asked. Work with existing project structure.
- If you make a mistake (wrong file, wrong directory), clean up after yourself.

## Code Style

Write clean, idiomatic code that a senior engineer would be proud of:

- Use clear, descriptive names. Variables should reveal intent (e.g., `user_count` not `n`, `is_valid` not `flag`).
- Follow the conventions of the language. PEP 8 for Python, standard formatting for JS/TS, etc.
- Keep functions small and focused. Each function should do one thing well.
- Prefer readability over cleverness. Simple, obvious code beats clever one-liners.
- Match the style of existing code in the project. Read files first to understand patterns.
- No unnecessary comments. Good code is self-documenting. Only comment the "why", not the "what".

Be concise in your responses. Explain what you're doing briefly."""

    # Append user's global CLAUDE.md if it exists
    claude_md = load_claude_md()
    if claude_md:
        prompt += f"\n\n## User Instructions\n\n{claude_md}"

    # Append project CLAUDE.md or AGENTS.md if they exist
    project_md = load_project_instructions(cwd)
    if project_md:
        prompt += f"\n\n## Project Instructions\n\n{project_md}"

    return prompt


def chat(
    messages: list[dict],
    model: str = "glm-4.7-flash",
    system_prompt: str | None = None,
) -> dict:
    """Send a chat request to Ollama and return the response."""
    full_messages = []

    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})

    full_messages.extend(messages)

    response = ollama.chat(
        model=model,
        messages=full_messages,
        tools=TOOLS,
    )

    return response


def chat_stream(
    messages: list[dict],
    model: str = "glm-4.7-flash",
    system_prompt: str | None = None,
) -> Generator[dict, None, None]:
    """Stream a chat response from Ollama."""
    full_messages = []

    if system_prompt:
        full_messages.append({"role": "system", "content": system_prompt})

    full_messages.extend(messages)

    for chunk in ollama.chat(
        model=model,
        messages=full_messages,
        tools=TOOLS,
        stream=True,
    ):
        yield chunk
