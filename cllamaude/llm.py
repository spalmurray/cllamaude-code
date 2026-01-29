"""Ollama LLM client wrapper."""

import ollama
from typing import Generator

from .tools import TOOLS


def get_system_prompt(cwd: str) -> str:
    """Generate the system prompt with context."""
    return f"""You are a helpful coding assistant running in a terminal.

Current working directory: {cwd}

You have access to the following tools:
- read_file(path): Read the contents of a file
- write_file(path, content): Write content to a file
- bash(command): Execute a bash command

Use these tools to help the user with coding tasks. When you need to see file contents, read them. When you need to make changes, write them. When you need to run commands, use bash.

Be concise in your responses. When making changes to files, explain what you're doing briefly."""


def chat(
    messages: list[dict],
    model: str = "glm4:latest",
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
    model: str = "glm4:latest",
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
