"""Summarize how agents used the taxonomy in saved runs.

    python scripts/analyze_taxonomy_use.py results/runs/pro100
    python scripts/analyze_taxonomy_use.py results/runs/pro100/taxonomy/12p_RMSNorm_r0

For each run: which taxonomy files were opened and when, how many evaluations of the
agent's own kernels happened after the first taxonomy read, and which cell-specific code
constructs appear in the kernels the agent wrote.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

PATH_RE = re.compile(r"taxonomy/(\d\d_[a-z_]+)/([a-z_]+\.[a-z]+)")
CELL_SIGNATURES = {
    "emit_pipeline": "04_async_pipeline",
    "use_abstract_mesh": "04_async_pipeline",
    "PrefetchScalarGridSpec": "07_grouped_matmul",
    "keepdims=True": "06_lane_reduction",
    "preferred_element_type": "01_mxu_feed",
    "pl.Buffered": "04_async_pipeline",
    "BlockSpec": "02_vmem_tile_layout",
}


def analyze(run: Path) -> None:
    traj = run / "trajectory.jsonl"
    if not traj.exists():
        return
    reads, writes, eval_turns = [], [], []
    for line in traj.read_text().splitlines():
        e = json.loads(line)
        if e["role"] != "assistant":
            continue
        for tc in e.get("tool_calls", []):
            name = tc["function"]["name"]
            try:
                args = json.loads(tc["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                continue
            text = args.get("path", "") if name == "read_file" else args.get("command", "")
            for cell, fname in PATH_RE.findall(text):
                reads.append((e["turn"], cell, fname))
            if name == "run_bash" and "eval.py" in text:
                eval_turns.append(e["turn"])
            if name == "write_file":
                writes.append((e["turn"], args.get("content", "")))
    result = json.loads((run / "result.json").read_text()) if (run / "result.json").exists() else {}
    print(f"\n== {run}")
    print(f"   status={result.get('status')} turns={result.get('turns_used')} "
          f"own_correct={result.get('agent_own_kernels_correct')}/{result.get('agent_own_kernel_evals')} "
          f"best_own_speedup={result.get('agent_own_best_speedup')}")
    if not reads:
        print("   taxonomy files opened: none")
    else:
        first = min(t for t, _, _ in reads)
        print(f"   taxonomy files opened: {len(reads)} (first at turn {first})")
        print("   by cell:", dict(Counter(c for _, c, _ in reads)))
        print("   by file:", dict(Counter(f for _, _, f in reads)))
        print("   eval.py calls after first taxonomy read:", sum(1 for t in eval_turns if t > first))
    used = Counter()
    for _, content in writes:
        for sig, cell in CELL_SIGNATURES.items():
            if sig in content:
                used[f"{sig} (cell {cell})"] += 1
    print("   cell constructs in agent-written kernels:", dict(used) if used else "none")


def main():
    target = Path(sys.argv[1])
    runs = [target] if (target / "trajectory.jsonl").exists() else sorted(p.parent for p in target.rglob("trajectory.jsonl"))
    for run in runs:
        analyze(run)


if __name__ == "__main__":
    main()
