"""Conversation/message history management."""

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Conversation:
    """Manages the conversation history."""

    messages: list[dict] = field(default_factory=list)

    def save(self, path: str) -> None:
        """Save conversation to a JSON file."""
        Path(path).write_text(json.dumps(self.messages, indent=2))

    @classmethod
    def load(cls, path: str) -> "Conversation":
        """Load conversation from a JSON file."""
        p = Path(path)
        if p.exists():
            messages = json.loads(p.read_text())
            return cls(messages=messages)
        return cls()

    def add_user_message(self, content: str) -> None:
        """Add a user message to the history."""
        self.messages.append({"role": "user", "content": content})

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the history."""
        self.messages.append({"role": "assistant", "content": content})

    def add_assistant_tool_calls(self, tool_calls: list[dict]) -> None:
        """Add an assistant message with tool calls."""
        self.messages.append({"role": "assistant", "tool_calls": tool_calls})

    def add_tool_result(self, tool_call_id: str, name: str, result: str) -> None:
        """Add a tool result to the history."""
        self.messages.append({
            "role": "tool",
            "name": name,
            "content": result,
        })

    def clear(self) -> None:
        """Clear the conversation history."""
        self.messages.clear()
