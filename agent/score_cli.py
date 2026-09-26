"""Score one kernel in its own process and print the result as a JSON line.

    python -m agent.score_cli --workload 12p_RMSNorm --kernel kernel.py [--interpret]

A TPU can be opened by only one process at a time, so the experiment runner scores
kernels through this CLI instead of in-process: when the process exits, the TPU is free
for the next agent run.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.runner import run_kernel


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--kernel", required=True, type=Path)
    ap.add_argument("--tpu", default="v5e")
    ap.add_argument("--interpret", action="store_true")
    args = ap.parse_args()
    r = run_kernel(args.workload, args.kernel, tpu=args.tpu, interpret=args.interpret,
                   num_warmup=5, num_iters=50)
    print("SCORE_JSON " + json.dumps({
        "status": r.status, "correct": r.correct, "speedup_vs_baseline": r.speedup_vs_baseline,
        "kernel_median_ms": r.kernel_median_ms, "baseline_median_ms": r.baseline_median_ms,
        "error": r.error,
    }))


if __name__ == "__main__":
    main()
