"""Score an agent workspace whose run was interrupted (or is otherwise unscored).

    python scripts/score_workspace.py ~/hawkeye_work/pro100/taxonomy/12p_RMSNorm_r0

The path must have the layout used by eval.run_agent_eval:
<work-dir>/<tag>/<condition>/<workload>_r<rep>. The best correct kernel the agent evaluated
(best_kernel.py, else kernel.py) is re-checked and timed with the repo harness, and
result.json, the trajectory and kernels are saved under results/runs/ exactly as a
finished run would be (stopped_reason is recorded as "interrupted").
"""

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agent.runner import run_kernel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workspace", type=Path)
    ap.add_argument("--interpret", action="store_true")
    args = ap.parse_args()

    ws = args.workspace.expanduser().resolve()
    m = re.fullmatch(r"(?P<workload>.+)_r(?P<rep>\d+)", ws.name)
    if not m:
        sys.exit(f"unexpected workspace name {ws.name!r}; expected <workload>_r<rep>")
    workload, rep = m["workload"], int(m["rep"])
    condition, tag = ws.parent.name, ws.parent.parent.name

    scored = ws / "best_kernel.py" if (ws / "best_kernel.py").exists() else ws / "kernel.py"
    log = ws / "eval_log.jsonl"
    entries = [json.loads(l) for l in log.read_text().splitlines()] if log.exists() else []
    own = [e for e in entries if not e.get("is_reference")]
    own_correct = [e for e in own if e.get("status") == "correct"]
    turns = 0
    if (ws / "trajectory.jsonl").exists():
        turns = max((json.loads(l)["turn"] for l in (ws / "trajectory.jsonl").read_text().splitlines()), default=0)

    final = run_kernel(workload, scored, tpu="v5e", interpret=args.interpret, num_warmup=5, num_iters=50)
    record = {
        "workload": workload, "condition": condition, "rep": rep, "tag": tag,
        "turns_used": turns, "stopped_reason": "interrupted", "scored_file": scored.name,
        "agent_eval_calls": len(entries), "agent_own_kernel_evals": len(own),
        "agent_own_kernels_correct": len(own_correct),
        "agent_own_best_speedup": max((e["speedup_vs_baseline"] for e in own_correct), default=None),
        "status": final.status, "correct": final.correct,
        "speedup_vs_baseline": final.speedup_vs_baseline,
        "kernel_median_ms": final.kernel_median_ms, "baseline_median_ms": final.baseline_median_ms,
        "error": final.error,
    }
    run_dir = ROOT / "results" / "runs" / tag / condition / ws.name
    run_dir.mkdir(parents=True, exist_ok=True)
    for name in ("trajectory.jsonl", "kernel.py", "best_kernel.py", "eval_log.jsonl", "task_prompt.md"):
        if (ws / name).exists():
            shutil.copy(ws / name, run_dir / name)
    (run_dir / "result.json").write_text(json.dumps(record, indent=2))
    print(f"{workload:<28}{condition:<10}rep{rep}  status={record['status']}  "
          f"speedup={record['speedup_vs_baseline']}  turns={turns} (interrupted)  "
          f"own_kernels_correct={len(own_correct)}/{len(own)}")
    print("saved", run_dir / "result.json")


if __name__ == "__main__":
    main()
