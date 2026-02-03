"""Centralized configuration and constants for cllamaude."""

# --- Context and History ---
DEFAULT_CONTEXT_WINDOW = 32768  # Default context window size in tokens
MAX_CHANGE_HISTORY = 50  # Maximum file changes to keep for undo
HISTORY_DISPLAY_LIMIT = 15  # Number of recent changes to show in /history

# --- File Reading ---
DEFAULT_READ_CONTEXT = 10  # Default lines above/below for read_around

# --- Search Results ---
MAX_GLOB_RESULTS = 100  # Maximum files to return from glob
MAX_GREP_RESULTS = 100  # Maximum matches to return from grep
RESULT_PREVIEW_LINES = 20  # Lines to show in result preview

# --- Timeouts (seconds) ---
BASH_TIMEOUT = 60  # Timeout for bash commands
GIT_TIMEOUT = 30  # Timeout for git commands

# --- Git ---
GIT_LOG_DEFAULT_LINES = 20  # Default number of commits to show in git log

# --- Display Truncation ---
COMMAND_DISPLAY_MAX = 50  # Max chars for command display before truncation
QUESTION_PREVIEW_MAX = 50  # Max chars for ask_user question preview
NOTE_PREVIEW_MAX = 40  # Max chars for note content preview

# --- File Structure Extraction ---
MAX_IMPORTS_DISPLAY = 10  # Max imports to show in file summary
MAX_CLASSES_DISPLAY = 10  # Max classes to show in file summary
MAX_FUNCTIONS_DISPLAY = 15  # Max functions to show in file summary
MAX_FUNCTIONS_IN_SUMMARY = 10  # Max functions to list in compressed summary

# --- UI ---
SPINNER_REFRESH_RATE = 10  # Refresh rate for spinner (per second)
DEFAULT_EDITOR = "vim"  # Fallback editor if $EDITOR not set

# --- Tool Names ---
# These are the tools that can have their output compressed
COMPRESSIBLE_TOOLS = {"read_file", "read_around", "git", "bash", "grep", "glob"}

# Read-only tools that don't need confirmation
SAFE_TOOLS = {"read_file", "read_around", "glob", "grep", "git", "note", "clear_note"}

# Tools blocked in planning mode
PLANNING_BLOCKED_TOOLS = {"write_file", "edit_file", "bash", "undo_changes"}

# Triggers for executing a plan
EXECUTE_TRIGGERS = {"do it", "doit", "ok", "go", "execute", "proceed", "yes", "run it", "looks good", "lgtm"}

# --- Ignored Directories ---
IGNORED_DIRS = {
    ".venv", "venv", "node_modules", ".git", "__pycache__",
    ".pytest_cache", ".mypy_cache", ".tox", "dist", "build",
    ".eggs", "*.egg-info", ".cache", ".ruff_cache",
}
