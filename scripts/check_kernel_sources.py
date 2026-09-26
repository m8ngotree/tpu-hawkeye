"""Run eval.py's source check over every kernel the agents wrote, to see which would be rejected.

    python scripts/check_kernel_sources.py [results/runs]

Prints, per run, how many saved kernel versions there are and how many the check rejects (and why).
"""

import collections
import glob
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "accelerator-agents"))
spec = importlib.util.spec_from_file_location("eval_cli", ROOT / "eval" / "eval.py")
eval_cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_cli)


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "results" / "runs")
    per_run = collections.defaultdict(lambda: [0, collections.Counter()])
    for f in sorted(glob.glob(f"{base}/*/*/*/kernels/*.py")):
        run = "/".join(Path(f).parts[-5:-2])
        problem = eval_cli._source_problem(Path(f).read_text())
        per_run[run][0] += 1
        if problem:
            per_run[run][1][problem] += 1
    for run, (n, reasons) in sorted(per_run.items()):
        rejected = sum(reasons.values())
        print(f"{run:48} kernels={n:3} rejected={rejected:3}")
        for reason, count in reasons.most_common(3):
            print(f"      {count:3}x {reason}")


if __name__ == "__main__":
    main()
