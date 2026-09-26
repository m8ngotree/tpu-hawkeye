"""Minimal hand-rolled tool-use loop driving the agent -- replaces an earlier
OpenHands-based harness (its headless-mode docs had real gaps around exactly what
this project needs: workspace dir, max-iterations, event schema).

This trades OpenHands' polish for something small enough to fully read and debug,
with zero Docker dependency -- it runs wherever Python runs.

Model-agnostic via any OpenAI-compatible chat-completions API (DeepSeek, OpenAI,
local vLLM/Ollama, ...) -- set model/base_url/api_key_env accordingly. DeepSeek's API
is OpenAI-SDK compatible: base_url="https://api.deepseek.com", model="deepseek-chat".

SAFETY NOTE: agent/tools_exec.py's run_bash is not sandboxed beyond staying inside the
workspace directory (see that module's docstring). Run this inside a disposable
environment (a scratch cloud VM) -- not on a machine you care about.
"""

import json
import os
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from agent.tools_exec import REJECTION, ToolResult, list_files, read_file, run_eval, write_file

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's contents, path relative to the workspace root. "
            "Optionally read only `limit` lines starting at line `offset` (0-based).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "offset": {"type": "integer"},
                    "limit": {"type": "integer"},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Overwrite (or create) a file, path relative to the workspace root.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List a directory inside the workspace (default: the workspace root).",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_eval",
            "description": (
                "Evaluate kernel.py on the TPU: checks correctness against baseline.py and "
                "times it. Returns eval.py's JSON result (status, speedup, diagnosis, or the "
                "error). This is the only way to execute code."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
]

TOOL_IMPLS = {"read_file": read_file, "write_file": write_file, "list_files": list_files, "run_eval": run_eval}

SYSTEM_PROMPT = (
    "You are an expert TPU/Pallas kernel engineer. You have tools to list, read and write "
    "files in your workspace and to evaluate your kernel. Work the "
    "task below to completion, then stop calling tools once you're satisfied or out "
    "of ideas -- don't call tools just to keep going."
)


@dataclass
class AgentRunResult:
    workspace: Path
    trajectory_path: Path
    turns_used: int
    stopped_reason: str  # 'done' | 'max_productive_turns' | 'max_raw_turns'
    final_eval: dict | None = None
    guard_rejections: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    api_seconds: float = 0.0
    tool_seconds: float = 0.0
    productive_turns: int = 0
    tool_counts: dict | None = None
    tool_errors: int = 0
    path_escape_attempts: int = 0
    kernel_writes: int = 0
    taxonomy_reads: list | None = None  # [{turn, path}] every taxonomy file read, repeats included
    eval_history: list | None = None  # one entry per run_eval call


_TAXONOMY_FILE = re.compile(r"taxonomy/[\w./*-]+\.(?:py|md|json)")


def _productive_actions(name: str, args: dict) -> bool:
    """Whether one tool call is a 'productive' action, as Hawkeye counts turns: a kernel edit,
    an evaluation, or a taxonomy read. Every taxonomy read counts, repeats included (the paper
    deduplicates only kernel-pool reads). Listings and reads of other files are free."""
    if name == "run_eval":
        return True
    if name == "write_file":
        return str(args.get("path", "")).strip("./") == "kernel.py"
    if name == "read_file":
        return bool(_TAXONOMY_FILE.fullmatch(str(args.get("path", "")).lstrip("./")))
    return False


def run_agent(
    workspace: Path,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    api_key_env: str = "LLM_API_KEY",
    max_productive_turns: int = 50,
    max_raw_turns: int = 200,
    on_event=None,
) -> AgentRunResult:
    """Run the tool-use loop against `workspace` (built by agent/workspace.py) until
    the model stops calling tools, `max_productive_turns` productive turns have been used (see
    _productive_actions; this is the budget, as in Hawkeye), or `max_raw_turns` model calls have been
    made (a safety cap). Logs every turn to
    <workspace>.trajectory.jsonl beside the workspace directory (mirrors Hawkeye's own trajectory.jsonl, Appendix G.4,
    so a run can be audited turn-by-turn afterward)."""
    api_key = os.environ.get(api_key_env)
    if not api_key:
        raise RuntimeError(f"set {api_key_env} to your LLM provider's API key")

    client = OpenAI(api_key=api_key, base_url=base_url)

    task_prompt = (workspace / "task_prompt.md").read_text()
    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_prompt},
    ]

    trajectory_path = workspace.parent / f"{workspace.name}.trajectory.jsonl"  # outside the workspace: the agent must not see it
    final_eval = None
    stopped_reason = "max_raw_turns"
    productive_turns = 0
    tool_counts: Counter = Counter()
    tool_errors = path_escape_attempts = kernel_writes = 0
    taxonomy_reads: list = []
    eval_history: list = []
    kernels_dir = workspace.parent / f"{workspace.name}.kernels"  # every kernel.py the agent wrote
    kernels_dir.mkdir(exist_ok=True)
    turn = 0
    guard_rejections = 0
    prompt_tokens = completion_tokens = 0
    api_seconds = tool_seconds = 0.0

    with open(trajectory_path, "w") as trajectory_file:
        for turn in range(1, max_raw_turns + 1):
            if productive_turns >= max_productive_turns:
                stopped_reason = "max_productive_turns"
                turn -= 1
                break
            if on_event:
                on_event("turn_start", turn, max_raw_turns, productive_turns, max_productive_turns)
            t0 = time.time()
            response = client.chat.completions.create(model=model, messages=messages, tools=TOOL_SCHEMAS)
            call_seconds = time.time() - t0
            api_seconds += call_seconds
            usage = getattr(response, "usage", None)
            turn_prompt = getattr(usage, "prompt_tokens", 0) or 0
            turn_completion = getattr(usage, "completion_tokens", 0) or 0
            prompt_tokens += turn_prompt
            completion_tokens += turn_completion
            message = response.choices[0].message
            messages.append(message.model_dump(exclude_none=True))
            trajectory_file.write(
                json.dumps(
                    {
                        "turn": turn,
                        "role": "assistant",
                        "content": message.content,
                        "reasoning": getattr(message, "reasoning_content", None)
                        or (getattr(message, "model_extra", None) or {}).get("reasoning_content"),
                        "tool_calls": [tc.model_dump() for tc in (message.tool_calls or [])],
                        "finish_reason": response.choices[0].finish_reason,
                        "productive_turns_so_far": productive_turns,
                        "time": t0,
                        "api_seconds": round(call_seconds, 2),
                        "prompt_tokens": turn_prompt,
                        "completion_tokens": turn_completion,
                    }
                )
                + "\n"
            )

            if on_event:
                on_event("assistant", turn, message.content, message.tool_calls or [])
            if not message.tool_calls:
                stopped_reason = "done"
                break

            turn_productive = False
            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                if _productive_actions(name, args):
                    turn_productive = True
                impl = TOOL_IMPLS.get(name)

                tool_elapsed = 0.0
                result = None
                if impl is None:
                    result_text = f"unknown tool: {name}"
                else:
                    t1 = time.time()
                    try:
                        result = impl(workspace, **args)
                    except Exception as e:  # a bad tool call must not end the run
                        result = ToolResult(ok=False, output=f"tool error: {type(e).__name__}: {e}")
                    tool_elapsed = time.time() - t1
                    tool_seconds += tool_elapsed
                    result_text = result.output
                    if result_text == REJECTION:
                        guard_rejections += 1
                    if name == "run_eval" and '"correctness"' in result_text:
                        try:
                            final_eval = json.loads(result_text)
                        except json.JSONDecodeError:
                            pass

                tool_counts[name] += 1
                tool_ok = result is not None and result.ok
                if not tool_ok:
                    tool_errors += 1
                if "resolves outside workspace" in result_text:
                    path_escape_attempts += 1
                snapshot = None
                if name == "write_file" and tool_ok and str(args.get("path", "")).strip("./") == "kernel.py":
                    kernel_writes += 1
                    snapshot = f"kernel_t{turn:03d}_{kernel_writes:02d}.py"
                    (kernels_dir / snapshot).write_text(str(args.get("content", "")))
                if name == "read_file" and _TAXONOMY_FILE.fullmatch(str(args.get("path", "")).lstrip("./")):
                    taxonomy_reads.append({"turn": turn, "path": str(args["path"]).lstrip("./")})
                if name == "run_eval":
                    entry = {"turn": turn, "eval_number": len(eval_history) + 1, "kernel_writes_so_far": kernel_writes}
                    try:
                        parsed = json.loads(result_text)
                    except json.JSONDecodeError:
                        parsed = {}
                    entry["status"] = parsed.get("status", "error")
                    entry["speedup"] = parsed.get("speedup_vs_baseline")
                    diag = parsed.get("diagnosis") or {}
                    entry["pct_of_roofline_limit"] = diag.get("pct_of_roofline_limit")
                    entry["workload_limit"] = diag.get("workload_limit")
                    entry["error"] = str(parsed.get("error") or "")[:300] or None
                    eval_history.append(entry)

                if on_event:
                    on_event("tool", turn, name, args, result_text)
                trajectory_file.write(
                    json.dumps({"turn": turn, "role": "tool", "name": name, "args": args, "output": result_text,
                                "ok": tool_ok, "seconds": round(tool_elapsed, 2), "turn_productive": turn_productive,
                                "kernel_snapshot": snapshot}) + "\n"
                )
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result_text})
            if turn_productive:
                productive_turns += 1
            for entry in eval_history:
                if entry["turn"] == turn:
                    entry["productive_turns_used"] = productive_turns

    return AgentRunResult(
        workspace=workspace,
        trajectory_path=trajectory_path,
        turns_used=turn,
        stopped_reason=stopped_reason,
        final_eval=final_eval,
        guard_rejections=guard_rejections,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        api_seconds=round(api_seconds, 1),
        tool_seconds=round(tool_seconds, 1),
        productive_turns=productive_turns,
        tool_counts=dict(tool_counts),
        tool_errors=tool_errors,
        path_escape_attempts=path_escape_attempts,
        kernel_writes=kernel_writes,
        taxonomy_reads=taxonomy_reads,
        eval_history=eval_history,
    )
