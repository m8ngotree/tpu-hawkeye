"""Run the coding agent over JAXBench workloads and score the kernel it leaves behind.

    python -m eval.run_agent_eval --workloads 12p_RMSNorm --conditions taxonomy none \
        --max-turns 30 --tag pilot

Conditions: `taxonomy` (workspace includes taxonomy/) and `none` (it does not).
Every other setting is identical between conditions. The final kernel is re-scored
independently of anything the agent reported: the best correct kernel the agent evaluated
(best_kernel.py, else its final kernel.py) is re-checked with the JAXBench correctness
check and timed against the JAXBench baseline.

Runs that already have a result.json are skipped unless they ended in a harness_error
(--force redoes all), so a crashed or
preempted sweep can simply be re-run.

Agent workspaces are built under --work-dir (default ~/hawkeye_work, outside the repo)
and each run's trajectory, final kernel and result are copied to results/runs/.

Requires LLM_API_KEY. --interpret scores on CPU (correctness only); omit it on a TPU.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.harness import run_agent
from agent.workspace import build_workspace


def score_in_subprocess(workload, kernel, interpret):
    """Score in a separate process: a TPU can be opened by one process at a time, so an
    in-process score would hold it and starve the next agent run's eval.py calls."""
    cmd = [sys.executable, "-m", "agent.score_cli", "--workload", workload, "--kernel", str(kernel)]
    if interpret:
        cmd.append("--interpret")
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, errors="replace", timeout=1800)
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("SCORE_JSON "):
            return json.loads(line[len("SCORE_JSON "):])
    raise RuntimeError(f"scoring failed: {(proc.stdout + proc.stderr)[-600:]}")


def make_printer(label):
    """Return an on_event callback that prints one compact line per agent action."""

    def clip(text, n):
        text = " ".join(str(text).split())
        return text if len(text) <= n else text[:n] + "..."

    def on_event(kind, *info):
        if kind == "turn_start":
            turn, max_turns = info
            print(f"[{label}] turn {turn}/{max_turns}", flush=True)
        elif kind == "assistant":
            turn, content, tool_calls = info
            if content:
                print(f"    says: {clip(content, 160)}", flush=True)
        elif kind == "tool":
            turn, name, tool_args, output = info
            cmd = tool_args.get("command") or tool_args.get("path") or ""
            line = f"    {name}: {clip(cmd, 110)}"
            if "eval.py" in str(cmd):
                status = re.search(r'"status":\s*"(\w+)"', output)
                speed = re.search(r'"speedup_vs_baseline":\s*([0-9.]+)', output)
                line += f"\n      -> status={status.group(1) if status else '?'}" + (f" speedup={speed.group(1)}" if speed else "")
            else:
                line += f"\n      -> {clip(output, 130)}"
            print(line, flush=True)

    return on_event


def run_one(workload, condition, rep, args):
    run_dir = ROOT / "results" / "runs" / args.tag / condition / f"{workload}_r{rep}"
    work_dir = Path(args.work_dir).expanduser() / args.tag / condition / f"{workload}_r{rep}"
    record = {"workload": workload, "condition": condition, "rep": rep, "tag": args.tag}
    result_path = run_dir / "result.json"
    if result_path.exists() and not args.force:
        previous = json.loads(result_path.read_text())
        if previous.get("status") != "harness_error":
            return previous
    t0 = time.time()
    try:
        ws = build_workspace(
            workload,
            problem_type=workload.split("_", 1)[1].replace("_", " "),
            out_dir=work_dir,
            use_taxonomy=(condition == "taxonomy"),
            use_kernel_pool=False,
        )
        if not args.dry_run:
            agent_res = run_agent(
                ws, model=args.model, base_url=args.base_url,
                api_key_env=args.api_key_env, max_turns=args.max_turns,
                on_event=None if args.quiet else make_printer(f"{workload} {condition}"),
            )
            record.update(turns_used=agent_res.turns_used, stopped_reason=agent_res.stopped_reason,
                          guard_rejections=agent_res.guard_rejections,
                          prompt_tokens=agent_res.prompt_tokens,
                          completion_tokens=agent_res.completion_tokens,
                          api_seconds=agent_res.api_seconds, tool_seconds=agent_res.tool_seconds)
        scored = ws / "best_kernel.py" if (ws / "best_kernel.py").exists() else ws / "kernel.py"
        record["scored_file"] = scored.name
        log = ws / "eval_log.jsonl"
        entries = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
        own = [e for e in entries if not e.get("is_reference")]
        own_correct = [e for e in own if e.get("status") == "correct"]
        record["agent_eval_calls"] = len(entries)
        record["agent_own_kernel_evals"] = len(own)
        record["agent_own_kernels_correct"] = len(own_correct)
        record["agent_own_best_speedup"] = max((e["speedup_vs_baseline"] for e in own_correct), default=None)
        record.update(score_in_subprocess(workload, scored, args.interpret))
    except Exception as e:
        record.update(status="harness_error", error=f"{type(e).__name__}: {e}",
                      traceback=traceback.format_exc()[-800:])
    record["wall_s"] = round(time.time() - t0, 1)
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("kernel.py", "best_kernel.py", "eval_log.jsonl", "task_prompt.md"):
        if (work_dir / name).exists():
            shutil.copy(work_dir / name, run_dir / name)
    trajectory = work_dir.parent / f"{work_dir.name}.trajectory.jsonl"
    if trajectory.exists():
        shutil.copy(trajectory, run_dir / "trajectory.jsonl")
    result_path.write_text(json.dumps(record, indent=2))
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", required=True, help="comma-separated JAXBench names")
    ap.add_argument("--conditions", nargs="+", default=["taxonomy", "none"])
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--max-turns", type=int, default=30)
    ap.add_argument("--model", default="deepseek-chat")
    ap.add_argument("--base-url", default="https://api.deepseek.com")
    ap.add_argument("--api-key-env", default="LLM_API_KEY")
    ap.add_argument("--interpret", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="skip the agent; score the empty starting kernel")
    ap.add_argument("--tag", default="run")
    ap.add_argument("--quiet", action="store_true", help="do not print per-turn progress")
    ap.add_argument("--work-dir", default="~/hawkeye_work",
                    help="where agent workspaces are built; keep it outside the repo checkout")
    ap.add_argument("--force", action="store_true", help="redo runs that already have a result.json")
    args = ap.parse_args()

    records = []
    for workload in args.workloads.split(","):
        for condition in args.conditions:
            for rep in range(args.reps):
                r = run_one(workload, condition, rep, args)
                records.append(r)
                print(f"{workload:<28}{condition:<10}rep{rep}  status={r.get('status')}  "
                      f"speedup={r.get('speedup_vs_baseline')}  turns={r.get('turns_used')}  "
                      f"own_kernels_correct={r.get('agent_own_kernels_correct')}/{r.get('agent_own_kernel_evals')}")
    out = ROOT / "results" / f"{args.tag}_summary.json"
    out.write_text(json.dumps(records, indent=2))
    print("saved", out)


if __name__ == "__main__":
    main()
