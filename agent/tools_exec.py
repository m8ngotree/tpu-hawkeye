"""The three tools exposed to the agent's LLM: read_file, write_file, run_bash.
All paths are resolved and clamped to a single workspace root -- the agent can't
read/write/execute anything outside the directory agent/workspace.py built for it.

`run_bash` rejects commands that name locations outside the workspace (a pattern guard,
not a sandbox: code the agent writes can still open arbitrary paths, so the harness
also counts rejections and results should be audited). There is no seccomp, container
or resource limit. That's a real tradeoff of the hand-rolled-loop choice
over a framework like OpenHands: you get simplicity and zero Docker dependency, but
you're relying on the *outer* environment (a disposable cloud VM) being the actual sandbox, not this code. Don't run this against a machine you
care about.
"""

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


class WorkspaceEscapeError(Exception):
    """Raised when a tool call tries to touch a path outside the workspace root."""


@dataclass
class ToolResult:
    ok: bool
    output: str


def _resolve(workspace: Path, path: str) -> Path:
    resolved = (workspace / path).resolve()
    workspace_resolved = workspace.resolve()
    if workspace_resolved not in resolved.parents and resolved != workspace_resolved:
        raise WorkspaceEscapeError(f"path {path!r} resolves outside workspace {workspace}")
    return resolved


def read_file(workspace: Path, path: str) -> ToolResult:
    try:
        target = _resolve(workspace, path)
        if not target.exists():
            return ToolResult(ok=False, output=f"no such file: {path}")
        return ToolResult(ok=True, output=target.read_text())
    except WorkspaceEscapeError as e:
        return ToolResult(ok=False, output=str(e))


def write_file(workspace: Path, path: str, content: str) -> ToolResult:
    try:
        target = _resolve(workspace, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        return ToolResult(ok=True, output=f"wrote {len(content)} bytes to {path}")
    except WorkspaceEscapeError as e:
        return ToolResult(ok=False, output=str(e))


_OUTSIDE_WORKSPACE = re.compile(
    r"(?<!\.)\.\.(?!\.)(/|\s|$)"          # parent-directory traversal
    r"|(^|[\s'\"=(])~"                       # home shorthand
    r"|\$\{?HOME|\$\{?PWD/\.\."
    r"|/(home|root|Users|mnt|opt|srv|media)\b"  # locations outside the workspace
    r"|tpu-hawkeye|hawkeye_work"
    r"|\bfind\s+/\s|\bls\s+/\s*$"
)

REJECTION = (
    "Command rejected: it refers to a location outside your workspace. "
    "Use only files in the current directory."
)


def run_bash(workspace: Path, command: str, timeout_s: int = 300) -> ToolResult:
    if _OUTSIDE_WORKSPACE.search(command.replace(str(workspace), "<WS>")):
        return ToolResult(ok=False, output=REJECTION)
    try:
        proc = subprocess.run(
            command,
            shell=True,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        output = proc.stdout + proc.stderr
        return ToolResult(ok=proc.returncode == 0, output=output[-8000:])
    except subprocess.TimeoutExpired:
        return ToolResult(ok=False, output=f"command timed out after {timeout_s}s")
