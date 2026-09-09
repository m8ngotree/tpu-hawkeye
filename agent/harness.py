"""Drive one OpenHands agent run over a workspace built by agent/workspace.py.

CAVEAT: the exact OpenHands CLI flags below (--headless, --json, --always-approve,
LLM config via env vars) are from OpenHands' public docs, not verified against a real
run in this environment -- installing OpenHands (pip install openhands, needs Docker
for its sandboxed runtime, needs an LLM API key + billing) is real setup work to do
where you'll actually run this (Kaggle/GCP), not something to fake here. Run
`openhands --help` after installing and adjust run_agent() if flags differ.

pip install openhands   # needs Python 3.12+
# LLM config: set env vars for your provider, e.g. for Anthropic:
#   export LLM_MODEL="anthropic/claude-sonnet-5"
#   export LLM_API_KEY="sk-ant-..."
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AgentRunResult:
    workspace: Path
    trajectory_path: Path
    returncode: int
    final_eval: dict | None  # last eval.py result the agent produced, if any


def run_agent(workspace: Path, max_iterations: int = 100, timeout_s: int = 3600) -> AgentRunResult:
    """Run OpenHands headless against `workspace` (built by agent/workspace.py's
    build_workspace()), reading task_prompt.md as the task and logging every
    tool-call/result event as JSONL -- matches trajectory.jsonl in the paper
    (Appendix G.4), so a run can be audited turn-by-turn after the fact.
    """
    trajectory_path = workspace / "trajectory.jsonl"

    cmd = [
        "openhands",
        "--headless",
        "--always-approve",  # unattended: no human in the loop to confirm each action
        "--json",
        "-f", str(workspace / "task_prompt.md"),
        "--max-iterations", str(max_iterations),
    ]

    with open(trajectory_path, "w") as trajectory_file:
        proc = subprocess.run(
            cmd,
            cwd=workspace,  # relies on OpenHands treating cwd as the agent's workspace; verify with --help
            stdout=trajectory_file,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
        )

    final_eval = _last_eval_result(trajectory_path)
    return AgentRunResult(
        workspace=workspace,
        trajectory_path=trajectory_path,
        returncode=proc.returncode,
        final_eval=final_eval,
    )


def _last_eval_result(trajectory_path: Path) -> dict | None:
    """Scan the trajectory for the last JSON blob eval.py printed (see eval/eval.py --
    it always prints a JSON result dict). Best-effort: exact event schema depends on
    OpenHands' --json output format, verify once a real trajectory file exists."""
    last = None
    for line in trajectory_path.read_text().splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        output = event.get("output") or event.get("content") or ""
        if isinstance(output, str) and '"correctness"' in output:
            try:
                last = json.loads(output)
            except json.JSONDecodeError:
                pass
    return last
