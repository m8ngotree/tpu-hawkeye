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
import time
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from agent.tools_exec import REJECTION, read_file, run_bash, write_file

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file's contents, path relative to the workspace root.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
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
            "name": "run_bash",
            "description": (
                "Run a shell command from the workspace root, e.g. "
                "`python eval.py --workload X --kernel kernel.py --interpret`. "
                "Returns combined stdout+stderr."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
]

TOOL_IMPLS = {"read_file": read_file, "write_file": write_file, "run_bash": run_bash}

SYSTEM_PROMPT = (
    "You are an expert TPU/Pallas kernel engineer. You have tools to read/write "
    "files and run shell commands, all relative to your workspace root. Work the "
    "task below to completion, then stop calling tools once you're satisfied or out "
    "of ideas -- don't call tools just to keep going."
)


@dataclass
class AgentRunResult:
    workspace: Path
    trajectory_path: Path
    turns_used: int
    stopped_reason: str  # 'done' | 'max_turns' | 'error'
    final_eval: dict | None = None
    guard_rejections: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    api_seconds: float = 0.0
    tool_seconds: float = 0.0


def run_agent(
    workspace: Path,
    model: str = "deepseek-chat",
    base_url: str = "https://api.deepseek.com",
    api_key_env: str = "LLM_API_KEY",
    max_turns: int = 100,
    on_event=None,
) -> AgentRunResult:
    """Run the tool-use loop against `workspace` (built by agent/workspace.py) until
    the model stops calling tools or max_turns is hit. Logs every turn to
    workspace/trajectory.jsonl (mirrors Hawkeye's own trajectory.jsonl, Appendix G.4,
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

    trajectory_path = workspace / "trajectory.jsonl"
    final_eval = None
    stopped_reason = "max_turns"
    turn = 0
    guard_rejections = 0
    prompt_tokens = completion_tokens = 0
    api_seconds = tool_seconds = 0.0

    with open(trajectory_path, "w") as trajectory_file:
        for turn in range(1, max_turns + 1):
            if on_event:
                on_event("turn_start", turn, max_turns)
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
                        "tool_calls": [tc.model_dump() for tc in (message.tool_calls or [])],
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

            for tool_call in message.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments or "{}")
                impl = TOOL_IMPLS.get(name)

                tool_elapsed = 0.0
                if impl is None:
                    result_text = f"unknown tool: {name}"
                else:
                    t1 = time.time()
                    result = impl(workspace, **args)
                    tool_elapsed = time.time() - t1
                    tool_seconds += tool_elapsed
                    result_text = result.output
                    if result_text == REJECTION:
                        guard_rejections += 1
                    if name == "run_bash" and '"correctness"' in result_text:
                        try:
                            final_eval = json.loads(result_text)
                        except json.JSONDecodeError:
                            pass

                if on_event:
                    on_event("tool", turn, name, args, result_text)
                trajectory_file.write(
                    json.dumps({"turn": turn, "role": "tool", "name": name, "args": args, "output": result_text,
                                "seconds": round(tool_elapsed, 2)}) + "\n"
                )
                messages.append({"role": "tool", "tool_call_id": tool_call.id, "content": result_text})

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
    )
