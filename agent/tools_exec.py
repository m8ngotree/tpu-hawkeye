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

import json
import re
import subprocess
import sys
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


# Files that exist in the workspace so eval.py can run and keep its records, but that the agent
# has no reason to see: the vendored evaluation harness and our bookkeeping. They are hidden from
# read_file and list_files (eval.py itself stays readable, as in the paper's workspace).
_HIDDEN = {"JAXBench", "eval_log.jsonl", "best_score.json", "best_kernel.py", "run_config.json", "__pycache__"}


def _hidden(workspace: Path, target: Path) -> bool:
    rel = target.relative_to(workspace.resolve())
    return bool(rel.parts) and rel.parts[0] in _HIDDEN


def read_file(workspace: Path, path: str, offset: int = 0, limit: int | None = None) -> ToolResult:
    """Read a file. `offset` (first line, 0-based) and `limit` (line count) are optional."""
    try:
        target = _resolve(workspace, path)
        if not target.exists() or _hidden(workspace, target):
            return ToolResult(ok=False, output=f"no such file: {path}")
        text = target.read_text(errors="replace")
        if offset or limit is not None:
            lines = text.splitlines(keepends=True)
            end = None if limit is None else int(offset) + int(limit)
            text = "".join(lines[int(offset):end])
        return ToolResult(ok=True, output=text)
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
    r"|pallas[./]ops"                        # kernels shipped inside JAX are off limits
    r"|\bfind\s+/\s|\bls\s+/\s*$"
)

REJECTION = (
    "Command rejected: it refers to a location outside your workspace, or to the kernels "
    "shipped inside JAX (jax.experimental.pallas.ops). Use only files in the current directory."
)


MAX_OUTPUT_CHARS = 8000
HEAD_CHARS = 3000


def _clip(output: str) -> str:
    """Keep the start and the end of long output, with a visible marker in between: error
    messages can be at either end (Python tracebacks end with the error; compiler errors
    often begin with it)."""
    if len(output) <= MAX_OUTPUT_CHARS:
        return output
    tail = MAX_OUTPUT_CHARS - HEAD_CHARS
    omitted = len(output) - MAX_OUTPUT_CHARS
    return f"{output[:HEAD_CHARS]}\n[... {omitted} characters omitted ...]\n{output[-tail:]}"


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
            encoding="utf-8",
            errors="replace",
            timeout=timeout_s,
        )
        output = proc.stdout + proc.stderr
        return ToolResult(ok=proc.returncode == 0, output=_clip(output))
    except subprocess.TimeoutExpired:
        return ToolResult(ok=False, output=f"command timed out after {timeout_s}s")


def list_files(workspace: Path, path: str = ".") -> ToolResult:
    """List a directory inside the workspace (directories end with '/')."""
    try:
        target = _resolve(workspace, path)
        if not target.is_dir() or _hidden(workspace, target):
            return ToolResult(ok=False, output=f"not a directory: {path}")
        hide = _HIDDEN if target == workspace.resolve() else {"__pycache__"}
        names = sorted(p.name + ("/" if p.is_dir() else "") for p in target.iterdir() if p.name not in hide)
        return ToolResult(ok=True, output="\n".join(names))
    except WorkspaceEscapeError as e:
        return ToolResult(ok=False, output=str(e))


def run_eval(workspace: Path, timeout_s: int = 900) -> ToolResult:
    """Evaluate kernel.py with the workspace's eval.py. This is the only way the agent can
    execute code: there is no general shell, so it cannot introspect installed libraries."""
    config = json.loads((workspace / "run_config.json").read_text())
    cmd = [sys.executable, "eval.py", "--workload", config["workload_name"],
           "--kernel", "kernel.py", "--tpu", config["generation"]]
    try:
        proc = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", timeout=timeout_s)
        return ToolResult(ok=proc.returncode == 0, output=_clip(proc.stdout + proc.stderr))
    except subprocess.TimeoutExpired:
        return ToolResult(ok=False, output=f"evaluation timed out after {timeout_s}s")
