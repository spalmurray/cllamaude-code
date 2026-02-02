"""Tool definitions and execution for cllamaude."""

import fnmatch
import os
import re
import subprocess
from pathlib import Path

# Tool definitions in Ollama/OpenAI format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path. Optionally read only specific lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to read",
                    },
                    "start_line": {
                        "type": "integer",
                        "description": "First line to read (1-indexed, inclusive). Omit to start from beginning.",
                    },
                    "end_line": {
                        "type": "integer",
                        "description": "Last line to read (1-indexed, inclusive). Omit to read to end.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file at the given path. Creates the file if it doesn't exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "The content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a bash command and return its output",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The bash command to execute",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Make a surgical edit to a file by replacing an exact string with new content. The old_string must appear exactly once in the file. Use this instead of write_file when you only need to change part of a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path to the file to edit",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact string to find and replace (must be unique in the file)",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The string to replace it with",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "glob",
            "description": "Find files matching a glob pattern. Returns a list of file paths.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The glob pattern (e.g., '**/*.py', 'src/*.js', '*.md')",
                    },
                    "path": {
                        "type": "string",
                        "description": "The directory to search in (default: current directory)",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search for a pattern in files. Returns matching lines with file:line_number:content format.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "The regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "File or directory to search in (default: current directory)",
                    },
                    "glob_pattern": {
                        "type": "string",
                        "description": "Only search files matching this glob pattern (e.g., '*.py')",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "undo_changes",
            "description": "Undo file changes from recent turns. Use this when the user asks to undo, revert, or rollback recent changes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "turns": {
                        "type": "integer",
                        "description": "Number of turns to undo (default: 1). Each turn is one user prompt and all the file changes made in response.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "ask_user",
            "description": "Ask the user a question and wait for their response. Use this to clarify requirements, get preferences, or confirm before taking action.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask the user",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_file",
            "description": "Mark a file as important to keep in context. By default, old file reads get compressed to save space. Use this on files you'll need to reference later (e.g., files you plan to edit).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path of the file to remember",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_file",
            "description": "Un-remember a file, allowing it to be compressed. Use when you're done with a file and want to free up context space.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "The path of the file to forget",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember_output",
            "description": "Remember a tool output to keep it in context. By default, old outputs (git, bash, grep, glob) get compressed. Use this on outputs you need to reference later.",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_id": {
                        "type": "integer",
                        "description": "The output ID to remember. If not provided, remembers the most recent output.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget_output",
            "description": "Un-remember a tool output, allowing it to be compressed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "output_id": {
                        "type": "integer",
                        "description": "The output ID to forget. If not provided, forgets the most recent output.",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "git",
            "description": "Run read-only git commands to understand repository state.",
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["status", "diff", "diff_staged", "log", "branch", "show", "blame"],
                        "description": "The git operation to run",
                    },
                    "args": {
                        "type": "string",
                        "description": "Additional arguments (e.g., file path for blame/diff, commit hash for show, -n 5 for log)",
                    },
                },
                "required": ["operation"],
            },
        },
    },
]


def read_file(path: str, start_line: int | None = None, end_line: int | None = None) -> str:
    """Read a file and return its contents, optionally a specific line range."""
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"Error: File not found: {path}"
        if not p.is_file():
            return f"Error: Not a file: {path}"

        content = p.read_text()

        # If no line range specified, return full content
        if start_line is None and end_line is None:
            return content

        lines = content.splitlines()
        total_lines = len(lines)

        # Convert to 0-indexed, handle defaults
        start = (start_line - 1) if start_line else 0
        end = end_line if end_line else total_lines

        # Clamp to valid range
        start = max(0, min(start, total_lines))
        end = max(0, min(end, total_lines))

        selected = lines[start:end]
        result = "\n".join(f"{i + start + 1}: {line}" for i, line in enumerate(selected))

        if start > 0 or end < total_lines:
            result = f"[Lines {start + 1}-{end} of {total_lines}]\n{result}"

        return result
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


def write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        p = Path(path).expanduser().resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Successfully wrote {len(content)} bytes to {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error writing file: {e}"


def bash(command: str) -> str:
    """Execute a bash command and return output."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60,
            cwd=os.getcwd(),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            if output:
                output += "\n"
            output += f"stderr: {result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Command timed out after 60 seconds"
    except Exception as e:
        return f"Error executing command: {e}"


def edit_file(path: str, old_string: str, new_string: str) -> str:
    """Edit a file by replacing old_string with new_string."""
    try:
        p = Path(path).expanduser().resolve()
        if not p.exists():
            return f"Error: File not found: {path}"
        if not p.is_file():
            return f"Error: Not a file: {path}"

        content = p.read_text()
        count = content.count(old_string)

        if count == 0:
            return f"Error: String not found in {path}"
        if count > 1:
            return f"Error: String appears {count} times in {path}. Must be unique."

        new_content = content.replace(old_string, new_string, 1)
        p.write_text(new_content)
        return f"Successfully edited {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error editing file: {e}"


def glob_files(pattern: str, path: str | None = None) -> str:
    """Find files matching a glob pattern."""
    try:
        base = Path(path).expanduser().resolve() if path else Path.cwd()
        if not base.exists():
            return f"Error: Path not found: {path}"

        matches = sorted(base.glob(pattern))
        # Filter to files only
        files = [str(m.relative_to(base)) for m in matches if m.is_file()]

        if not files:
            return f"No files found matching '{pattern}'"

        # Limit output
        if len(files) > 100:
            return "\n".join(files[:100]) + f"\n... and {len(files) - 100} more files"
        return "\n".join(files)
    except Exception as e:
        return f"Error searching files: {e}"


def grep_files(pattern: str, path: str | None = None, glob_pattern: str | None = None) -> str:
    """Search for a pattern in files."""
    try:
        regex = re.compile(pattern)
    except re.error as e:
        return f"Error: Invalid regex pattern: {e}"

    try:
        base = Path(path).expanduser().resolve() if path else Path.cwd()
        if not base.exists():
            return f"Error: Path not found: {path}"

        results = []
        max_results = 100

        if base.is_file():
            files = [base]
        else:
            # Get all files, optionally filtered by glob
            if glob_pattern:
                files = [f for f in base.rglob(glob_pattern) if f.is_file()]
            else:
                files = [f for f in base.rglob("*") if f.is_file()]

        for file_path in files:
            if len(results) >= max_results:
                break
            try:
                content = file_path.read_text()
                for i, line in enumerate(content.splitlines(), 1):
                    if regex.search(line):
                        rel_path = file_path.relative_to(base) if not base.is_file() else file_path.name
                        results.append(f"{rel_path}:{i}:{line.strip()}")
                        if len(results) >= max_results:
                            break
            except (UnicodeDecodeError, PermissionError):
                # Skip binary or unreadable files
                continue

        if not results:
            return f"No matches found for '{pattern}'"

        output = "\n".join(results)
        if len(results) == max_results:
            output += f"\n... (limited to {max_results} results)"
        return output
    except Exception as e:
        return f"Error searching: {e}"


def git_command(operation: str, args: str | None = None) -> str:
    """Run read-only git commands."""
    commands = {
        "status": "git status",
        "diff": "git diff",
        "diff_staged": "git diff --staged",
        "log": "git log --oneline -20",
        "branch": "git branch -a",
        "show": "git show",
        "blame": "git blame",
    }

    if operation not in commands:
        return f"Error: Unknown git operation: {operation}"

    cmd = commands[operation]

    # Append args if provided
    if args:
        # For log, allow overriding the default -20
        if operation == "log" and args.strip().startswith("-n"):
            cmd = f"git log --oneline {args}"
        else:
            cmd = f"{cmd} {args}"

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.getcwd(),
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            if output:
                output += "\n"
            output += result.stderr
        return output if output else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: Git command timed out"
    except Exception as e:
        return f"Error running git: {e}"


TOOL_FUNCTIONS = {
    "read_file": read_file,
    "write_file": write_file,
    "bash": bash,
    "edit_file": edit_file,
    "glob": glob_files,
    "grep": grep_files,
    "git": git_command,
}


def execute_tool(name: str, arguments: dict) -> str:
    """Execute a tool by name with the given arguments."""
    if name not in TOOL_FUNCTIONS:
        return f"Error: Unknown tool: {name}"

    # Filter to only valid parameters (models sometimes hallucinate extra args)
    import inspect
    func = TOOL_FUNCTIONS[name]
    sig = inspect.signature(func)
    valid_params = set(sig.parameters.keys())
    filtered_args = {k: v for k, v in arguments.items() if k in valid_params}

    return TOOL_FUNCTIONS[name](**filtered_args)
