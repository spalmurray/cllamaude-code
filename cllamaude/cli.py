"""Main CLI entry point for cllamaude."""

import argparse
import difflib
import json
import os
import re
import sys
import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm
from rich.syntax import Syntax
from rich.live import Live
from rich.text import Text
from rich.table import Table

from .conversation import Conversation
from .llm import chat, get_system_prompt
from .tools import execute_tool

console = Console()

TOOL_NAMES = {"read_file", "read_around", "write_file", "bash", "edit_file", "glob", "grep", "undo_changes", "ask_user", "remember_file", "forget_file", "remember_output", "forget_output", "git", "note", "clear_note"}


@dataclass
class FileChange:
    """Tracks a file change for undo functionality."""
    path: str
    old_content: str | None  # None if file didn't exist
    new_content: str
    operation: str  # "write" or "edit"
    timestamp: datetime
    turn: int  # Which conversation turn this belongs to


MAX_HISTORY = 50  # Keep last 50 changes


@dataclass
class Session:
    """Encapsulates all session state."""
    change_history: list[FileChange] = field(default_factory=list)
    current_turn: int = 0
    plan_mode: bool = False
    remembered_files: set[str] = field(default_factory=set)
    remembered_outputs: set[int] = field(default_factory=set)
    tool_output_counter: int = 0
    notes: list[str] = field(default_factory=list)

    def start_new_turn(self, messages: list | None = None) -> None:
        """Called when user sends a new prompt. Compresses outputs from previous turn."""
        self.current_turn += 1
        if messages is not None:
            compress_old_tool_outputs(messages, self)

    def record_change(self, path: str, old_content: str | None, new_content: str, operation: str) -> None:
        """Record a file change for undo."""
        self.change_history.append(FileChange(
            path=path,
            old_content=old_content,
            new_content=new_content,
            operation=operation,
            timestamp=datetime.now(),
            turn=self.current_turn,
        ))
        if len(self.change_history) > MAX_HISTORY:
            self.change_history.pop(0)

    def undo_turns(self, num_turns: int = 1) -> str:
        """Undo file changes from the last N turns. Returns status message."""
        if not self.change_history:
            return "Nothing to undo - no changes recorded."

        turns_in_history = sorted(set(c.turn for c in self.change_history), reverse=True)
        if not turns_in_history:
            return "Nothing to undo."

        turns_to_undo = set(turns_in_history[:num_turns])
        changes_to_undo = [c for c in self.change_history if c.turn in turns_to_undo]

        if not changes_to_undo:
            return "Nothing to undo."

        results = []
        for change in reversed(changes_to_undo):
            p = Path(change.path).expanduser().resolve()
            try:
                if change.old_content is None:
                    if p.exists():
                        p.unlink()
                        results.append(f"Deleted {change.path} (was newly created)")
                else:
                    p.write_text(change.old_content)
                    results.append(f"Restored {change.path}")
            except Exception as e:
                results.append(f"Error reverting {change.path}: {e}")

        for change in changes_to_undo:
            self.change_history.remove(change)

        turn_word = "turn" if num_turns == 1 else "turns"
        return f"Undid {len(changes_to_undo)} change(s) from {num_turns} {turn_word}:\n" + "\n".join(results)

    def show_history(self) -> None:
        """Display recent file changes."""
        if not self.change_history:
            console.print("[dim]No changes recorded yet.[/dim]")
            return

        table = Table(title="Recent Changes (newest first)")
        table.add_column("Turn", style="dim")
        table.add_column("Time", style="dim")
        table.add_column("Op")
        table.add_column("File")

        for change in reversed(self.change_history[-15:]):
            time_str = change.timestamp.strftime("%H:%M:%S")
            table.add_row(str(change.turn), time_str, change.operation, change.path)

        console.print(table)

    def remember_file(self, path: str) -> str:
        """Mark a file as important - it won't be compressed."""
        path_normalized = str(Path(path).expanduser().resolve())
        self.remembered_files.add(path_normalized)
        return f"Will keep {path} in context"

    def forget_file(self, path: str, messages: list | None = None) -> str:
        """Un-remember a file and immediately compress it in context."""
        path_normalized = str(Path(path).expanduser().resolve())
        self.remembered_files.discard(path_normalized)

        if messages is None:
            return f"Forgot {path}"

        compressed_count = 0
        for i, msg in enumerate(messages):
            if msg.get("role") == "tool" and msg.get("name") == "read_file":
                content = msg.get("content", "")
                if content.startswith("[File:") or content.startswith("Error"):
                    continue

                if i > 0:
                    prev = messages[i - 1]
                    tool_calls = prev.get("tool_calls", [])
                    for tc in tool_calls:
                        if tc.get("function", {}).get("name") == "read_file":
                            tc_path = tc.get("function", {}).get("arguments", {}).get("path", "")
                            tc_normalized = str(Path(tc_path).expanduser().resolve()) if tc_path else ""
                            if tc_normalized == path_normalized:
                                msg["content"] = compress_file_content(path, content)
                                compressed_count += 1
                                break

        if compressed_count > 0:
            return f"Forgot {path} ({compressed_count} read(s) compressed)"
        return f"Forgot {path}"

    def get_next_output_id(self) -> int:
        """Get the next tool output ID."""
        self.tool_output_counter += 1
        return self.tool_output_counter

    def remember_output(self, output_id: int | None = None) -> str:
        """Remember a tool output by ID. If no ID, remembers the most recent."""
        if output_id is None:
            output_id = self.tool_output_counter
        if output_id <= 0:
            return "No outputs to remember"
        self.remembered_outputs.add(output_id)
        return f"Remembered output #{output_id}"

    def forget_output(self, output_id: int | None = None, messages: list | None = None) -> str:
        """Forget a tool output by ID and immediately compress it."""
        if output_id is None:
            output_id = self.tool_output_counter

        self.remembered_outputs.discard(output_id)

        if messages is None:
            return f"Forgot output #{output_id}"

        for msg in messages:
            if msg.get("role") == "tool" and msg.get("output_id") == output_id:
                content = msg.get("content", "")
                name = msg.get("name", "")

                if content.startswith("[") or content.startswith("Error") or content.startswith("No "):
                    continue

                idx = messages.index(msg)
                args = {}
                if idx > 0:
                    prev = messages[idx - 1]
                    tool_calls = prev.get("tool_calls", [])
                    for tc in tool_calls:
                        if tc.get("function", {}).get("name") == name:
                            args = tc.get("function", {}).get("arguments", {})
                            break

                if name == "read_file":
                    path = args.get("path", "unknown")
                    msg["content"] = compress_file_content(path, content)
                else:
                    msg["content"] = summarize_tool_output(name, args, content)

                return f"Forgot and compressed output #{output_id}"

        return f"Forgot output #{output_id}"

    def add_note(self, content: str) -> str:
        """Add a note to persistent memory."""
        note_id = len(self.notes)
        self.notes.append(content)
        return f"Added note #{note_id}"

    def clear_note(self, note_id: int | None = None) -> str:
        """Clear a note by ID, or all notes if no ID given."""
        if note_id is None:
            count = len(self.notes)
            self.notes.clear()
            return f"Cleared all {count} notes"
        if 0 <= note_id < len(self.notes):
            self.notes[note_id] = ""
            return f"Cleared note #{note_id}"
        return f"Note #{note_id} not found"

    def get_notes_context(self) -> str:
        """Get notes formatted for inclusion in context."""
        active_notes = [(i, n) for i, n in enumerate(self.notes) if n]
        if not active_notes:
            return ""
        lines = ["## Your Notes"]
        for i, content in active_notes:
            lines.append(f"[{i}] {content}")
        return "\n".join(lines)

    def clear(self) -> None:
        """Clear all session state."""
        self.change_history.clear()
        self.current_turn = 0
        self.plan_mode = False
        self.remembered_files.clear()
        self.remembered_outputs.clear()
        self.tool_output_counter = 0
        self.notes.clear()



PLAN_MODE_PROMPT = """
You are in PLANNING MODE. The user wants you to create a plan before taking action.

You CAN use read-only tools to explore: read_file, glob, grep
You must NOT use tools that make changes: write_file, edit_file, bash, undo_changes

After exploring, output a plan that:
1. Lists the files you'll modify
2. Describes each change you'll make
3. Notes any questions or uncertainties

Format your plan as a numbered list. Wait for user approval before making changes.
"""

PLANNING_BLOCKED_TOOLS = {"write_file", "edit_file", "bash", "undo_changes"}

EXECUTE_TRIGGERS = {"do it", "doit", "ok", "go", "execute", "proceed", "yes", "run it", "looks good", "lgtm"}


def estimate_tokens(text: str) -> int:
    """Estimate token count (roughly 4 chars per token)."""
    return len(text) // 4


def get_attr(obj, key, default=None):
    """Get attribute from dict or object."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def detect_language(path: str) -> str:
    """Detect programming language from file extension."""
    ext_map = {
        ".py": "Python", ".js": "JavaScript", ".ts": "TypeScript",
        ".jsx": "JSX", ".tsx": "TSX", ".rb": "Ruby", ".go": "Go",
        ".rs": "Rust", ".java": "Java", ".c": "C", ".cpp": "C++",
        ".h": "C Header", ".hpp": "C++ Header", ".cs": "C#",
        ".php": "PHP", ".swift": "Swift", ".kt": "Kotlin",
        ".scala": "Scala", ".sh": "Shell", ".bash": "Bash",
        ".zsh": "Zsh", ".fish": "Fish", ".ps1": "PowerShell",
        ".sql": "SQL", ".html": "HTML", ".css": "CSS",
        ".scss": "SCSS", ".sass": "Sass", ".less": "Less",
        ".json": "JSON", ".yaml": "YAML", ".yml": "YAML",
        ".toml": "TOML", ".xml": "XML", ".md": "Markdown",
        ".txt": "Text", ".cfg": "Config", ".ini": "INI",
    }
    ext = Path(path).suffix.lower()
    return ext_map.get(ext, "Unknown")


def extract_file_structure(content: str, language: str) -> dict:
    """Extract imports, classes, and functions from file content."""
    lines = content.split("\n")
    imports = []
    classes = []
    functions = []

    for line in lines:
        stripped = line.strip()

        # Python
        if language == "Python":
            if stripped.startswith("import ") or stripped.startswith("from "):
                imports.append(stripped.split("#")[0].strip())
            elif stripped.startswith("class "):
                match = re.match(r"class\s+(\w+)", stripped)
                if match:
                    classes.append(match.group(1))
            elif stripped.startswith("def "):
                match = re.match(r"def\s+(\w+)", stripped)
                if match:
                    functions.append(match.group(1))

        # JavaScript/TypeScript
        elif language in ("JavaScript", "TypeScript", "JSX", "TSX"):
            if stripped.startswith("import "):
                imports.append(stripped.split("//")[0].strip())
            elif "require(" in stripped:
                imports.append(stripped.split("//")[0].strip())
            elif stripped.startswith("class "):
                match = re.match(r"class\s+(\w+)", stripped)
                if match:
                    classes.append(match.group(1))
            elif stripped.startswith("function ") or stripped.startswith("async function "):
                match = re.match(r"(?:async\s+)?function\s+(\w+)", stripped)
                if match:
                    functions.append(match.group(1))
            elif re.match(r"(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\(", stripped):
                match = re.match(r"(?:const|let|var)\s+(\w+)", stripped)
                if match:
                    functions.append(match.group(1))

        # Go
        elif language == "Go":
            if stripped.startswith("import "):
                imports.append(stripped)
            elif stripped.startswith("type ") and " struct" in stripped:
                match = re.match(r"type\s+(\w+)", stripped)
                if match:
                    classes.append(match.group(1))
            elif stripped.startswith("func "):
                match = re.match(r"func\s+(?:\([^)]+\)\s+)?(\w+)", stripped)
                if match:
                    functions.append(match.group(1))

    return {
        "imports": imports[:10],  # Limit to avoid bloat
        "classes": classes[:10],
        "functions": functions[:15],
    }


def compress_file_content(path: str, content: str) -> str:
    """Compress file content into a summary."""
    if content.startswith("Error") or content.startswith("[Lines"):
        return content  # Already an error or a line range read

    lines = content.split("\n")
    line_count = len(lines)
    language = detect_language(path)
    structure = extract_file_structure(content, language)

    summary_parts = [f"[File: {path}, {line_count} lines, {language}]"]

    if structure["imports"]:
        summary_parts.append(f"Imports: {', '.join(structure['imports'][:5])}")
        if len(structure["imports"]) > 5:
            summary_parts[-1] += f" (+{len(structure['imports']) - 5} more)"

    if structure["classes"]:
        summary_parts.append(f"Classes: {', '.join(structure['classes'])}")

    if structure["functions"]:
        summary_parts.append(f"Functions: {', '.join(structure['functions'][:10])}")
        if len(structure["functions"]) > 10:
            summary_parts[-1] += f" (+{len(structure['functions']) - 10} more)"

    return "\n".join(summary_parts)


def summarize_tool_output(name: str, args: dict, result: str) -> str:
    """Create a compressed summary of a tool output."""
    lines = result.strip().split("\n")
    line_count = len(lines)

    if name == "git":
        op = args.get("operation", "?")
        extra = args.get("args", "")
        if op == "status":
            # Count modified/untracked from output
            modified = sum(1 for l in lines if l.strip().startswith("modified:"))
            untracked = sum(1 for l in lines if l.strip().startswith("??") or "Untracked files" in l)
            return f"[git status: {modified} modified, {untracked} untracked, {line_count} lines]"
        elif op in ("diff", "diff_staged"):
            files_changed = sum(1 for l in lines if l.startswith("diff --git"))
            return f"[git {op}: {files_changed} files, {line_count} lines of diff]"
        elif op == "log":
            return f"[git log: {line_count} commits shown]"
        elif op == "blame":
            return f"[git blame {extra}: {line_count} lines]"
        elif op == "branch":
            return f"[git branch: {line_count} branches]"
        elif op == "show":
            return f"[git show {extra}: {line_count} lines]"
        return f"[git {op}: {line_count} lines]"

    elif name == "bash":
        cmd = args.get("command", "?")
        # Truncate long commands
        if len(cmd) > 50:
            cmd = cmd[:47] + "..."
        return f"[bash '{cmd}': {line_count} lines output]"

    elif name == "grep":
        pattern = args.get("pattern", "?")
        match_count = line_count
        return f"[grep '{pattern}': {match_count} matches]"

    elif name == "glob":
        pattern = args.get("pattern", "?")
        return f"[glob '{pattern}': {line_count} files found]"

    return f"[{name}: {line_count} lines]"


def compress_old_tool_outputs(messages: list, session: Session, keep_recent: int = 1) -> None:
    """Compress old tool outputs in conversation history in-place.

    Skips remembered files and remembered outputs.
    """
    # Tools that can be compressed
    compressible_tools = {"read_file", "git", "bash", "grep", "glob"}

    # Find all compressible tool results
    tool_entries = []  # [(index, name, args, output_id), ...]
    for i, msg in enumerate(messages):
        if msg.get("role") == "tool":
            name = msg.get("name", "")
            if name not in compressible_tools:
                continue

            content = msg.get("content", "")
            # Skip if already compressed or is an error
            if content.startswith("[") or content.startswith("Error") or content.startswith("No "):
                continue

            # Get args from the preceding assistant message's tool call
            args = {}
            if i > 0:
                prev = messages[i - 1]
                tool_calls = prev.get("tool_calls", [])
                for tc in tool_calls:
                    if tc.get("function", {}).get("name") == name:
                        args = tc.get("function", {}).get("arguments", {})
                        break

            # For read_file, check if path is remembered
            if name == "read_file":
                path = args.get("path", "")
                path_normalized = str(Path(path).expanduser().resolve()) if path else ""
                if path_normalized in session.remembered_files:
                    continue

            # Check if output ID is remembered
            output_id = msg.get("output_id", 0)
            if output_id in session.remembered_outputs:
                continue

            tool_entries.append((i, name, args, output_id))

    # Keep the most recent ones, compress the rest
    to_compress = tool_entries[:-keep_recent] if len(tool_entries) > keep_recent else []

    for idx, name, args, output_id in to_compress:
        if name == "read_file":
            path = args.get("path", "unknown")
            messages[idx]["content"] = compress_file_content(path, messages[idx]["content"])
        else:
            messages[idx]["content"] = summarize_tool_output(name, args, messages[idx]["content"])


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
    pattern = r'\{\s*"name"\s*:\s*"(read_file|write_file|bash|edit_file|glob|grep|undo_changes)"'

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
        path = args.get('path', '?')
        start = args.get('start_line')
        end = args.get('end_line')
        if start and end:
            return f"read_file({path}, {start}-{end})"
        return f"read_file({path})"
    elif name == "read_around":
        path = args.get('path', '?')
        line = args.get('line', '?')
        ctx = args.get('context', 10)
        return f"read_around({path}, line {line}, ±{ctx})"
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
    elif name == "ask_user":
        question = args.get("question", "?")
        preview = question[:50] + "..." if len(question) > 50 else question
        return f"ask_user({preview})"
    elif name == "remember_file":
        return f"remember_file({args.get('path', '?')})"
    elif name == "forget_file":
        return f"forget_file({args.get('path', '?')})"
    elif name == "remember_output":
        oid = args.get("output_id", "last")
        return f"remember_output(#{oid})"
    elif name == "forget_output":
        oid = args.get("output_id", "last")
        return f"forget_output(#{oid})"
    elif name == "note":
        content = args.get("content", "")
        preview = content[:40] + "..." if len(content) > 40 else content
        return f"note({preview})"
    elif name == "clear_note":
        nid = args.get("note_id", "all")
        return f"clear_note(#{nid})"
    elif name == "git":
        op = args.get("operation", "?")
        extra = args.get("args", "")
        if extra:
            return f"git {op} {extra}"
        return f"git {op}"
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


def is_safe_bash_command(command: str) -> bool:
    """Check if a bash command is safe to auto-approve."""
    import shlex
    try:
        parts = shlex.split(command)
    except ValueError:
        return False

    if not parts:
        return False

    cmd = parts[0]
    cwd = Path.cwd().resolve()

    # uv run pytest is always safe
    if cmd == "uv" and len(parts) >= 2 and parts[1] == "run" and len(parts) >= 3 and parts[2] == "pytest":
        return True

    # ls, find, grep are safe if targeting cwd or subdirectories
    if cmd in ("ls", "find", "grep"):
        # If no path argument, it's cwd (safe)
        # Check all non-flag arguments to see if they're in cwd
        for part in parts[1:]:
            if part.startswith("-"):
                continue  # Skip flags
            # Check if path is in cwd
            try:
                p = Path(part).expanduser().resolve()
                if not p.is_relative_to(cwd):
                    return False
            except Exception:
                return False
        return True

    return False


def is_dangerous_git_command(command: str) -> str | None:
    """Check if a command is a dangerous git operation. Returns warning message or None."""
    import shlex
    try:
        parts = shlex.split(command)
    except ValueError:
        return None

    if not parts:
        return None

    # Check for git commands
    if parts[0] != "git":
        return None

    if len(parts) < 2:
        return None

    subcommand = parts[1]

    # Block these entirely
    blocked = {
        "commit": "Git commits are blocked. Make commits manually.",
        "push": "Git push is blocked. Push manually after reviewing changes.",
        "reset": "Git reset is blocked. Reset manually if needed.",
        "checkout": None,  # Only block specific patterns
        "clean": "Git clean is blocked. Clean manually if needed.",
        "rebase": "Git rebase is blocked. Rebase manually.",
        "merge": "Git merge is blocked. Merge manually.",
        "branch": None,  # Only block -D
        "stash": None,  # stash drop is dangerous
    }

    if subcommand in blocked and blocked[subcommand]:
        return blocked[subcommand]

    # Check for dangerous flags/patterns
    if subcommand == "checkout":
        # Block `git checkout .` or `git checkout -- .` (discard all changes)
        if "." in parts or "--" in parts:
            return "Git checkout that discards changes is blocked."

    if subcommand == "branch":
        # Block force delete
        if "-D" in parts:
            return "Force branch deletion (-D) is blocked. Use -d for safe delete."

    if subcommand == "stash":
        if "drop" in parts or "clear" in parts:
            return "Git stash drop/clear is blocked."

    if subcommand == "reset":
        return "Git reset is blocked. Reset manually if needed."

    return None


def confirm_tool(name: str, args: dict, auto_approve: bool = False) -> bool:
    """Ask for confirmation before executing a destructive tool."""
    if name in ("read_file", "read_around", "glob", "grep", "git", "note", "clear_note"):
        # Read operations and notes are safe, no confirmation needed
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

        # Block dangerous git commands entirely
        git_warning = is_dangerous_git_command(command)
        if git_warning:
            console.print(Panel(
                f"[red]{git_warning}[/red]\n\nBlocked command: {command}",
                title="🚫 Blocked",
                border_style="red",
            ))
            return False

        # Auto-approve safe read-only commands in cwd
        if is_safe_bash_command(command):
            return True

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
    session: Session,
    model: str,
    system_prompt: str,
    auto_approve: bool = False,
    num_ctx: int = 32768,
    debug: bool = False,
    planning: bool = False,
    turn_start_time: float | None = None,
) -> None:
    """Run the agent loop until no more tool calls."""
    # In planning mode, append planning instructions to system prompt
    base_system_prompt = system_prompt
    if planning:
        base_system_prompt = base_system_prompt + "\n" + PLAN_MODE_PROMPT
    if turn_start_time is None:
        turn_start_time = time.time()
    while True:
        # Build full system prompt with notes
        full_system_prompt = base_system_prompt
        notes_ctx = session.get_notes_context()
        if notes_ctx:
            full_system_prompt = full_system_prompt + "\n\n" + notes_ctx

        tokens, pct = get_context_usage(conversation.messages, full_system_prompt, num_ctx)

        # Non-streaming request with spinner
        response = None
        error = None
        invocation_start = time.time()

        def do_chat():
            nonlocal response, error
            try:
                response = chat(
                    conversation.messages,
                    model=model,
                    system_prompt=full_system_prompt,
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
                invocation_elapsed = int(time.time() - invocation_start)
                turn_elapsed = int(time.time() - turn_start_time)
                width = console.width
                left = f"{spinner_frames[frame_idx]} {turn_elapsed}s ({invocation_elapsed}s)"
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
            error_str = str(error)
            if "failed to parse XML" in error_str or "syntax error" in error_str.lower():
                console.print("[red]Model generated malformed output (XML parse error). Try a simpler prompt or say 'continue'.[/red]")
                break
            raise error

        message = get_attr(response, "message", {})

        # Debug: show raw message
        if debug:
            console.print(f"[dim]DEBUG message: {message}[/dim]")

        # Check for tool calls (structured or parsed from text)
        tool_calls = get_attr(message, "tool_calls")
        content = get_attr(message, "content", "")

        if debug:
            console.print(f"[dim]DEBUG tool_calls: {tool_calls}, type: {type(tool_calls)}[/dim]")

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

                # Tools that generate outputs needing note/forget
                compressible = {"read_file", "read_around", "git", "bash", "grep", "glob"}

                # Block write tools in planning mode
                if planning and name in PLANNING_BLOCKED_TOOLS:
                    result = f"Blocked in planning mode: {name}. Only read-only tools allowed."
                    console.print(f"[yellow]{result}[/yellow]")
                # Handle undo_changes specially (not through execute_tool)
                elif name == "undo_changes":
                    num_turns = args.get("turns", 1) or 1
                    result = session.undo_turns(num_turns)
                    console.print(Panel(result, title="Undo", border_style="green"))
                # Handle ask_user specially (requires user interaction)
                elif name == "ask_user":
                    question = args.get("question", "")
                    console.print(Panel(question, title="🤔 Agent Question", border_style="cyan"))
                    result = console.input("[bold cyan]Your answer:[/bold cyan] ")
                    if not result.strip():
                        result = "(no answer provided)"
                # Handle remember_file specially (marks file to keep in context)
                elif name == "remember_file":
                    file_path = args.get("path", "")
                    result = session.remember_file(file_path)
                    console.print(f"[dim]{result}[/dim]")
                # Handle forget_file specially (un-remembers and compresses immediately)
                elif name == "forget_file":
                    file_path = args.get("path", "")
                    result = session.forget_file(file_path, conversation.messages)
                    console.print(f"[dim]{result}[/dim]")
                # Handle remember_output specially
                elif name == "remember_output":
                    oid = args.get("output_id")
                    result = session.remember_output(oid)
                    console.print(f"[dim]{result}[/dim]")
                # Handle forget_output specially (compresses immediately)
                elif name == "forget_output":
                    oid = args.get("output_id")
                    if oid is None:
                        oid = session.tool_output_counter
                    result = session.forget_output(oid, conversation.messages)
                    console.print(f"[dim]{result}[/dim]")
                # Handle note specially
                elif name == "note":
                    content = args.get("content", "")
                    result = session.add_note(content)
                    console.print(f"[dim]{result}[/dim]")
                # Handle clear_note specially
                elif name == "clear_note":
                    nid = args.get("note_id")
                    result = session.clear_note(nid)
                    console.print(f"[dim]{result}[/dim]")
                # Confirm destructive operations
                elif not confirm_tool(name, args, auto_approve):
                    result = "Tool execution cancelled by user."
                    console.print("[yellow]Cancelled[/yellow]")
                else:
                    # Capture file content before write/edit for undo
                    if name in ("write_file", "edit_file"):
                        file_path = args.get("path", "")
                        p = Path(file_path).expanduser().resolve()
                        old_content = None
                        if p.exists() and p.is_file():
                            try:
                                old_content = p.read_text()
                            except Exception:
                                pass

                    result = execute_tool(name, args)

                    # Record successful file changes for undo
                    if name in ("write_file", "edit_file") and not result.startswith("Error"):
                        new_content = args.get("content", "") if name == "write_file" else ""
                        if name == "edit_file":
                            # For edit, read the new content
                            try:
                                new_content = Path(file_path).expanduser().resolve().read_text()
                            except Exception:
                                new_content = ""
                        session.record_change(file_path, old_content, new_content, name.replace("_file", ""))
                    # Show result preview for some operations
                    # Get output ID for display
                    display_id = session.tool_output_counter if name in {"read_file", "read_around", "git", "bash", "grep", "glob"} else None
                    id_suffix = f" [output #{display_id}]" if display_id else ""

                    if name in ("read_file", "read_around"):
                        lines = result.split("\n")
                        console.print(f"[dim]Read {len(lines)} lines{id_suffix}[/dim]")
                    elif name == "bash":
                        console.print(Panel(result, title=f"Output{id_suffix}", border_style="dim"))
                    elif name == "git" and not result.startswith("Error"):
                        lines = result.split("\n")
                        op = args.get("operation", "")
                        # Don't show full diff output, just summary
                        if op in ("diff", "diff_staged", "blame", "show"):
                            console.print(f"[dim]git {op}: {len(lines)} lines{id_suffix}[/dim]")
                        else:
                            title = f"git {op}{id_suffix}"
                            console.print(Panel(result, title=title, border_style="dim"))
                    elif name in ("glob", "grep") and not result.startswith(("Error", "No ")):
                        lines = result.split("\n")
                        preview = "\n".join(lines[:20])
                        if len(lines) > 20:
                            preview += f"\n... ({len(lines) - 20} more)"
                        console.print(Panel(preview, title=f"{name} results", border_style="dim"))
                    else:
                        console.print(f"[dim]{result}[/dim]")

                # Assign output ID for compressible outputs and track pending
                output_id = session.get_next_output_id() if name in compressible else None

                result_for_context = result

                conversation.add_tool_result(
                    tool_call_id=str(id(tool_call)),
                    name=name,
                    result=result_for_context,
                    output_id=output_id,
                )

        else:
            # No tool calls, just a text response
            if content:
                conversation.add_assistant_message(content)
                console.print()
                console.print(Markdown(content))
            else:
                console.print("[dim](no response)[/dim]")
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
        "-d", "--debug",
        action="store_true",
        help="Show debug info (raw model responses)"
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

    # Create session state
    session = Session()

    cwd = os.getcwd()
    system_prompt = get_system_prompt(cwd)

    # Non-interactive mode: run single prompt and exit
    if args.prompt:
        session.start_new_turn(conversation.messages)
        conversation.add_user_message(args.prompt)
        run_agent_loop(
            conversation,
            session,
            args.model,
            system_prompt,
            auto_approve=True,
            num_ctx=args.context,
            debug=args.debug,
        )
        if args.session:
            conversation.save(args.session)
        return

    # Interactive mode
    console.print(Panel(
        f"[bold]Cllamaude[/bold] - Ollama-powered coding assistant\n"
        f"Model: {args.model}\n"
        f"Commands: exit, clear, /undo, /history, /plan <task>",
        border_style="blue",
    ))

    while True:
        try:
            # Show context usage right-aligned above prompt
            tokens, pct = get_context_usage(conversation.messages, system_prompt, args.context)
            context_info = f"{tokens} tokens ({pct:.0f}%)"
            console.print()
            console.print(f"[dim]{context_info:>{console.width - 1}}[/dim]")
            prompt_style = "[bold cyan]plan>[/bold cyan] " if session.plan_mode else "[bold blue]>[/bold blue] "
            user_input = console.input(prompt_style)

            if not user_input.strip():
                continue

            if user_input.strip().lower() in ("exit", "quit"):
                console.print("[dim]Goodbye![/dim]")
                break

            if user_input.strip().lower() == "clear":
                conversation.clear()
                session.clear()
                console.print("[dim]Conversation cleared.[/dim]")
                continue

            if user_input.strip().lower() in ("/undo", "undo"):
                result = session.undo_turns(1)
                console.print(f"[green]{result}[/green]")
                continue

            if user_input.strip().lower() in ("/history", "history"):
                session.show_history()
                continue

            # Handle /plan command
            if user_input.strip().lower().startswith("/plan "):
                task = user_input.strip()[6:]  # Remove "/plan "
                session.plan_mode = True
                console.print(Panel(
                    f"[bold]Planning mode[/bold] - I'll create a plan for:\n{task}\n\n"
                    f"[dim]Say 'do it' to execute, or give feedback to adjust.[/dim]",
                    border_style="cyan",
                ))
                session.start_new_turn(conversation.messages)
                conversation.add_user_message(task)
                run_agent_loop(
                    conversation,
                    session,
                    args.model,
                    system_prompt,
                    num_ctx=args.context,
                    debug=args.debug,
                    planning=True,
                )
                if args.session:
                    conversation.save(args.session)
                continue

            # Cancel plan mode
            if session.plan_mode and user_input.strip().lower() in ("/cancel", "cancel", "nevermind", "abort"):
                session.plan_mode = False
                console.print("[dim]Plan cancelled.[/dim]")
                continue

            # Check if user is approving a plan
            if session.plan_mode and user_input.strip().lower() in EXECUTE_TRIGGERS:
                session.plan_mode = False
                console.print("[cyan]Executing plan...[/cyan]")
                session.start_new_turn(conversation.messages)
                conversation.add_user_message("Execute the plan now. Use the tools to make the changes.")
                run_agent_loop(
                    conversation,
                    session,
                    args.model,
                    system_prompt,
                    num_ctx=args.context,
                    debug=args.debug,
                )
                if args.session:
                    conversation.save(args.session)
                continue

            # Exit plan mode on other input (feedback or new task)
            if session.plan_mode:
                # User is giving feedback on the plan
                session.start_new_turn(conversation.messages)
                conversation.add_user_message(user_input)
                run_agent_loop(
                    conversation,
                    session,
                    args.model,
                    system_prompt,
                    num_ctx=args.context,
                    debug=args.debug,
                    planning=True,  # Stay in planning mode for feedback
                )
                if args.session:
                    conversation.save(args.session)
                continue

            session.start_new_turn(conversation.messages)
            conversation.add_user_message(user_input)
            run_agent_loop(
                conversation,
                session,
                args.model,
                system_prompt,
                num_ctx=args.context,
                debug=args.debug,
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
