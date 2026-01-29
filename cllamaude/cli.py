"""Main CLI entry point for cllamaude."""

import argparse
import difflib
import json
import os
import re
import sys
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax

from .conversation import Conversation
from .llm import chat, get_system_prompt
from .tools import execute_tool

console = Console()

TOOL_NAMES = {"read_file", "write_file", "bash"}


def parse_tool_calls_from_text(content: str) -> list[dict] | None:
    """Try to parse tool calls from raw text output.

    Some models output JSON tool calls as text instead of using
    the structured tool_calls format.
    """
    if not content:
        return None

    # Look for JSON objects that look like tool calls
    # Pattern: {"name": "tool_name", ...}
    json_pattern = r'\{[^{}]*"name"\s*:\s*"(\w+)"[^{}]*\}'

    tool_calls = []
    for match in re.finditer(json_pattern, content, re.DOTALL):
        try:
            # Try to parse the JSON - but the regex might not capture nested braces
            # So let's try to find the full JSON object starting from match
            start = match.start()
            # Find matching brace
            brace_count = 0
            end = start
            for i, c in enumerate(content[start:]):
                if c == '{':
                    brace_count += 1
                elif c == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = start + i + 1
                        break

            json_str = content[start:end]
            data = json.loads(json_str)

            name = data.get("name")
            if name not in TOOL_NAMES:
                continue

            # Handle both "arguments" and "parameters" keys
            args = data.get("arguments") or data.get("parameters") or {}

            tool_calls.append({
                "function": {
                    "name": name,
                    "arguments": args,
                }
            })
        except (json.JSONDecodeError, KeyError):
            continue

    return tool_calls if tool_calls else None


def format_tool_call(name: str, args: dict) -> str:
    """Format a tool call for display."""
    if name == "read_file":
        return f"read_file({args.get('path', '?')})"
    elif name == "write_file":
        path = args.get("path", "?")
        content = args.get("content", "")
        preview = content[:100] + "..." if len(content) > 100 else content
        return f"write_file({path}, {len(content)} bytes)"
    elif name == "bash":
        return f"bash({args.get('command', '?')})"
    return f"{name}({args})"


def make_diff(old_content: str, new_content: str, path: str) -> str:
    """Generate a colored diff between old and new content."""
    old_lines = old_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)

    diff = difflib.unified_diff(old_lines, new_lines, fromfile=path, tofile=path, lineterm="")

    result = []
    for line in diff:
        line = line.rstrip('\n')
        if line.startswith('+') and not line.startswith('+++'):
            result.append(f"[green]{line}[/green]")
        elif line.startswith('-') and not line.startswith('---'):
            result.append(f"[red]{line}[/red]")
        elif line.startswith('@@'):
            result.append(f"[cyan]{line}[/cyan]")
        else:
            result.append(line)

    return '\n'.join(result)


def is_path_in_cwd(path: str) -> bool:
    """Check if a path is inside the current working directory."""
    try:
        p = Path(path).expanduser().resolve()
        cwd = Path.cwd().resolve()
        return p.is_relative_to(cwd)
    except Exception:
        return False


def confirm_tool(name: str, args: dict) -> bool:
    """Ask for confirmation before executing a destructive tool."""
    if name == "read_file":
        # Read is safe, no confirmation needed
        return True

    console.print()
    if name == "write_file":
        path = args.get("path", "?")
        new_content = args.get("content", "")

        # Try to read existing file for diff
        old_content = ""
        p = Path(path).expanduser()
        if p.exists() and p.is_file():
            try:
                old_content = p.read_text()
            except Exception:
                pass

        if old_content:
            # Show diff
            diff = make_diff(old_content, new_content, path)
            console.print(Panel(diff, title=f"Changes to: {path}", border_style="yellow"))
        else:
            # New file, show full content
            console.print(Panel(
                Syntax(new_content, "text", theme="monokai", line_numbers=True),
                title=f"Create: {path}",
                border_style="yellow",
            ))

        # Auto-approve writes inside cwd
        if is_path_in_cwd(path):
            return True

    elif name == "bash":
        command = args.get("command", "?")
        console.print(Panel(
            command,
            title="Execute bash command",
            border_style="yellow",
        ))

    return Confirm.ask("Execute this?", default=True)


def run_agent_loop(conversation: Conversation, model: str, system_prompt: str) -> None:
    """Run the agent loop until no more tool calls."""
    while True:
        response = chat(conversation.messages, model=model, system_prompt=system_prompt)
        message = response.get("message", {})

        # Check for tool calls (structured or parsed from text)
        tool_calls = message.get("tool_calls")
        content = message.get("content", "")

        # If no structured tool calls, try to parse from text
        if not tool_calls and content:
            tool_calls = parse_tool_calls_from_text(content)
            if tool_calls:
                console.print("[dim](parsed tool call from text)[/dim]")

        if tool_calls:
            # Add the assistant's tool call message
            conversation.add_assistant_tool_calls(tool_calls)

            # Execute each tool
            for tool_call in tool_calls:
                func = tool_call.get("function", {})
                name = func.get("name", "")
                args = func.get("arguments", {})

                console.print(f"[dim]Tool:[/dim] {format_tool_call(name, args)}")

                # Confirm destructive operations
                if not confirm_tool(name, args):
                    result = "Tool execution cancelled by user."
                    console.print("[yellow]Cancelled[/yellow]")
                else:
                    result = execute_tool(name, args)
                    # Show result preview for read operations
                    if name == "read_file" and not result.startswith("Error"):
                        lines = result.split("\n")
                        preview = "\n".join(lines[:10])
                        if len(lines) > 10:
                            preview += f"\n... ({len(lines) - 10} more lines)"
                        console.print(Panel(preview, title="File contents", border_style="dim"))
                    elif name == "bash":
                        console.print(Panel(result, title="Output", border_style="dim"))
                    else:
                        console.print(f"[dim]{result}[/dim]")

                conversation.add_tool_result(
                    tool_call_id=str(id(tool_call)),
                    name=name,
                    result=result,
                )
        else:
            # No tool calls, just a text response
            if content:
                console.print()
                console.print(Markdown(content))
            break


def main():
    parser = argparse.ArgumentParser(description="Cllamaude - Ollama-powered coding CLI")
    parser.add_argument(
        "-m", "--model",
        default="glm4:latest",
        help="Ollama model to use (default: glm4:latest)"
    )
    args = parser.parse_args()

    console.print(Panel(
        f"[bold]Cllamaude[/bold] - Ollama-powered coding assistant\n"
        f"Model: {args.model}\n"
        f"Type 'exit' or 'quit' to exit, 'clear' to reset conversation",
        border_style="blue",
    ))

    conversation = Conversation()
    cwd = os.getcwd()
    system_prompt = get_system_prompt(cwd)

    while True:
        try:
            console.print()
            user_input = console.input("[bold blue]>[/bold blue] ")

            if not user_input.strip():
                continue

            if user_input.strip().lower() in ("exit", "quit"):
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.strip().lower() == "clear":
                conversation.clear()
                console.print("[dim]Conversation cleared.[/dim]")
                continue

            conversation.add_user_message(user_input)
            run_agent_loop(conversation, args.model, system_prompt)

        except KeyboardInterrupt:
            console.print("\n[dim]Use 'exit' to quit[/dim]")
            continue
        except EOFError:
            console.print("\n[dim]Goodbye![/dim]")
            break


if __name__ == "__main__":
    main()
