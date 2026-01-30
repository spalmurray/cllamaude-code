"""Main CLI entry point for cllamaude."""

import argparse
import difflib
import json
import os
import re
import sys
import time
import threading
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Prompt, Confirm
from rich.syntax import Syntax
from rich.live import Live
from rich.text import Text

from .conversation import Conversation
from .llm import chat, get_system_prompt
from .tools import execute_tool

console = Console()

TOOL_NAMES = {"read_file", "write_file", "bash", "edit_file", "glob", "grep"}


def estimate_tokens(text: str) -> int:
    """Estimate token count (roughly 4 chars per token)."""
    return len(text) // 4


def summarize_tool_result(name: str, result: str, max_lines: int = 50) -> str:
    """Summarize tool result to reduce context noise."""
    if result.startswith("Error") or result.startswith("No "):
        return result

    # Don't summarize read_file - model needs exact content for accurate edits
    if name == "read_file":
        return result

    lines = result.split("\n")
    if len(lines) <= max_lines:
        return result

    if name in ("bash", "glob", "grep"):
        # Truncate long discovery output
        return "\n".join(lines[:max_lines]) + f"\n... ({len(lines) - max_lines} more lines)"

    return result


def get_context_usage(messages: list, system_prompt: str, max_tokens: int = 32768) -> tuple[int, float]:
    """Calculate estimated token usage and percentage."""
    total_text = system_prompt
    for msg in messages:
        content = msg.get("content", "")
        if content:
            total_text += content
    tokens = estimate_tokens(total_text)
    percentage = (tokens / max_tokens) * 100
    return tokens, percentage


def parse_tool_calls_from_text(content: str) -> list[dict] | None:
    """Try to parse tool calls from raw text output.

    Some models output JSON tool calls as text instead of using
    the structured tool_calls format.
    """
    if not content:
        return None

    # Find potential tool call starts: {"name": "tool_name"
    # Use a simple pattern that just finds the start, then brace-match to get full JSON
    pattern = r'\{\s*"name"\s*:\s*"(read_file|write_file|bash|edit_file|glob|grep)"'

    tool_calls = []
    for match in re.finditer(pattern, content):
        try:
            start = match.start()
            # Find matching closing brace
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

            if end <= start:
                continue

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
        return f"write_file({path}, {len(content)} bytes)"
    elif name == "bash":
        return f"bash({args.get('command', '?')})"
    elif name == "edit_file":
        path = args.get("path", "?")
        old = args.get("old_string", "")
        new = args.get("new_string", "")
        return f"edit_file({path}, {len(old)} -> {len(new)} chars)"
    elif name == "glob":
        return f"glob({args.get('pattern', '?')})"
    elif name == "grep":
        pattern = args.get("pattern", "?")
        glob_pat = args.get("glob_pattern", "")
        if glob_pat:
            return f"grep({pattern}, {glob_pat})"
        return f"grep({pattern})"
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


def confirm_tool(name: str, args: dict, auto_approve: bool = False) -> bool:
    """Ask for confirmation before executing a destructive tool."""
    if name in ("read_file", "glob", "grep"):
        # Read operations are safe, no confirmation needed
        return True

    console.print()
    if name == "edit_file":
        path = args.get("path", "?")
        old_string = args.get("old_string", "")
        new_string = args.get("new_string", "")

        # Show what's being changed
        diff = make_diff(old_string, new_string, path)
        console.print(Panel(diff, title=f"Edit: {path}", border_style="yellow"))

        # Auto-approve edits inside cwd
        if is_path_in_cwd(path):
            return True

    elif name == "write_file":
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

    if auto_approve:
        return True
    return Confirm.ask("Execute this?", default=True)


def run_agent_loop(
    conversation: Conversation,
    model: str,
    system_prompt: str,
    auto_approve: bool = False,
    num_ctx: int = 32768,
) -> None:
    """Run the agent loop until no more tool calls."""
    while True:
        tokens, pct = get_context_usage(conversation.messages, system_prompt, num_ctx)

        response = None
        error = None
        start_time = time.time()

        def do_chat():
            nonlocal response, error
            try:
                response = chat(
                    conversation.messages,
                    model=model,
                    system_prompt=system_prompt,
                    num_ctx=num_ctx,
                )
            except Exception as e:
                error = e

        thread = threading.Thread(target=do_chat)
        thread.start()

        spinner_frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        frame_idx = 0
        with Live(console=console, refresh_per_second=10, transient=True) as live:
            while thread.is_alive():
                elapsed = int(time.time() - start_time)
                width = console.width
                left = f"{spinner_frames[frame_idx]} {elapsed}s"
                right = f"{tokens} tokens ({pct:.0f}%)"
                padding = width - len(left) - len(right) - 1

                status_text = Text()
                status_text.append(left, style="bold blue")
                status_text.append(" " * max(1, padding))
                status_text.append(right, style="dim")
                live.update(status_text)

                frame_idx = (frame_idx + 1) % len(spinner_frames)
                thread.join(timeout=0.1)

        if error:
            raise error

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
                if not confirm_tool(name, args, auto_approve):
                    result = "Tool execution cancelled by user."
                    console.print("[yellow]Cancelled[/yellow]")
                else:
                    result = execute_tool(name, args)
                    # Show result preview for some operations
                    if name == "read_file":
                        lines = result.split("\n")
                        console.print(f"[dim]Read {len(lines)} lines[/dim]")
                    elif name == "bash":
                        console.print(Panel(result, title="Output", border_style="dim"))
                    elif name in ("glob", "grep") and not result.startswith(("Error", "No ")):
                        lines = result.split("\n")
                        preview = "\n".join(lines[:20])
                        if len(lines) > 20:
                            preview += f"\n... ({len(lines) - 20} more)"
                        console.print(Panel(preview, title=f"{name} results", border_style="dim"))
                    else:
                        console.print(f"[dim]{result}[/dim]")

                # Summarize result to reduce context bloat
                summarized = summarize_tool_result(name, result)
                conversation.add_tool_result(
                    tool_call_id=str(id(tool_call)),
                    name=name,
                    result=summarized,
                )
        else:
            # No tool calls, just a text response
            if content:
                conversation.add_assistant_message(content)
                console.print()
                console.print(Markdown(content))
            break


def main():
    parser = argparse.ArgumentParser(description="Cllamaude - Ollama-powered coding CLI")
    parser.add_argument(
        "-m", "--model",
        default="glm-4.7-flash",
        help="Ollama model to use (default: glm-4.7-flash)"
    )
    parser.add_argument(
        "-s", "--session",
        help="Session file to persist conversation"
    )
    parser.add_argument(
        "-c", "--context",
        type=int,
        default=32768,
        help="Context window size in tokens (default: 32768)"
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        help="Single prompt to run (non-interactive mode)"
    )
    args = parser.parse_args()

    # Load or create conversation
    if args.session:
        conversation = Conversation.load(args.session)
        if conversation.messages:
            console.print(f"[dim]Loaded {len(conversation.messages)} messages from session[/dim]")
    else:
        conversation = Conversation()

    cwd = os.getcwd()
    system_prompt = get_system_prompt(cwd)

    # Non-interactive mode: run single prompt and exit
    if args.prompt:
        conversation.add_user_message(args.prompt)
        run_agent_loop(
            conversation,
            args.model,
            system_prompt,
            auto_approve=True,
            num_ctx=args.context,
        )
        if args.session:
            conversation.save(args.session)
        return

    # Interactive mode
    console.print(Panel(
        f"[bold]Cllamaude[/bold] - Ollama-powered coding assistant\n"
        f"Model: {args.model}\n"
        f"Type 'exit' or 'quit' to exit, 'clear' to reset conversation",
        border_style="blue",
    ))

    while True:
        try:
            # Show context usage right-aligned above prompt
            tokens, pct = get_context_usage(conversation.messages, system_prompt, args.context)
            context_info = f"{tokens} tokens ({pct:.0f}%)"
            console.print()
            console.print(f"[dim]{context_info:>{console.width - 1}}[/dim]")
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
            run_agent_loop(
                conversation,
                args.model,
                system_prompt,
                num_ctx=args.context,
            )

            if args.session:
                conversation.save(args.session)

        except KeyboardInterrupt:
            console.print("\n[dim]Use 'exit' to quit[/dim]")
            continue
        except EOFError:
            console.print("\n[dim]Goodbye![/dim]")
            break

    if args.session:
        conversation.save(args.session)


if __name__ == "__main__":
    main()
