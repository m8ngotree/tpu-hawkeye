"""Drive agent.loop.optimize() over JAXBench workloads and collect results.

Usage (once implemented):
    python -m eval.run_agent_eval --workloads all --use-taxonomy --out results/taxonomy_run.json
    python -m eval.run_agent_eval --workloads all --no-use-taxonomy --out results/baseline_run.json

Then diff the two result files for the speedup delta the taxonomy provides.
"""

import argparse
import json
from pathlib import Path

JAXBENCH_ROOT = Path(__file__).parent.parent / "external" / "accelerator-agents" / "JAXBench"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workloads", default="all", help="'all' or comma-separated workload names")
    parser.add_argument("--use-taxonomy", dest="use_taxonomy", action="store_true", default=True)
    parser.add_argument("--no-use-taxonomy", dest="use_taxonomy", action="store_false")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    raise NotImplementedError(
        "wire up agent.loop.optimize() per workload in "
        f"{JAXBENCH_ROOT / 'benchmark'}, dump per-workload speedup to args.out"
    )


if __name__ == "__main__":
    main()
