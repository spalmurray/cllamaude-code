# cllamaude

Ollama-powered coding CLI assistant. Like Claude Code, but with local models.

## Setup

```bash
uv sync
```

## Usage

```bash
# Run with default model (glm-4.7-flash)
uv run cllamaude

# Run with a specific model
uv run cllamaude -m llama3.1:latest

# Run with a single prompt (non-interactive)
uv run cllamaude "explain this codebase"

# Adjust context window size
uv run cllamaude -c 65536
```

## Install globally

```bash
uv tool install -e .
cllamaude
```

## Commands

- `exit` / `quit` - Exit the CLI
- `clear` - Clear conversation history
- `/undo` - Undo the last turn's file changes
- `/history` - Show recent file changes
- `/plan <task>` - Enter planning mode (read-only exploration, then approve to execute)
- `/edit` - Open your $EDITOR for multi-line input

## Tools

The assistant has access to:

**File operations:**
- `read_file` - Read specific line ranges from files
- `read_around` - Read lines around a specific line number
- `write_file` - Create or overwrite files
- `edit_file` - Surgical find-and-replace edits

**Search:**
- `glob` - Find files by pattern (e.g., `**/*.py`)
- `grep` - Search file contents with regex

**Shell:**
- `bash` - Run shell commands (dangerous git commands are blocked)
- `git` - Read-only git operations (status, diff, log, branch, show, blame)

**Memory management:**
- `remember_file` / `forget_file` - Keep important files in context
- `remember_output` / `forget_output` - Keep important tool outputs
- `note` / `clear_note` - Save persistent notes
- `undo_changes` - Revert recent file modifications

**Interaction:**
- `ask_user` - Ask clarifying questions

## Safety

- File writes inside the current directory are auto-approved
- Writes outside cwd require confirmation
- Dangerous git commands (commit, push, reset, rebase, etc.) are blocked
- Safe read-only commands (ls, find, grep in cwd) are auto-approved
