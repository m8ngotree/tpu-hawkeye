"""Evaluate a candidate kernel against a JAXBench workload.

    python eval.py --workload 12p_RMSNorm --kernel kernel.py [--interpret]

Inside an agent workspace this uses the workspace's own copy of the JAXBench harness
(no access to anything else is needed). Run from the repo's eval/ directory it falls
back to the JAXBench checkout under external/. Prints the JSON result; exits 0 iff the
kernel is correct.
"""

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
if (HERE / "JAXBench").exists():
    sys.path.insert(0, str(HERE))
else:
    sys.path.insert(0, str(HERE.parent / "external" / "accelerator-agents"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True, help="JAXBench workload name, e.g. 12p_RMSNorm")
    parser.add_argument("--kernel", required=True, type=Path, help="path to your kernel.py (must define workload(*inputs))")
    parser.add_argument("--tpu", default="v5e", choices=["v5e", "v6e"])
    parser.add_argument("--interpret", action="store_true", help="run Pallas kernels in the CPU interpreter (correctness only)")
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=50)
    args = parser.parse_args()

    os.environ["PALLAS_INTERPRET"] = "1" if args.interpret else "0"
    from JAXBench.harness.evaluator import evaluate_kernel

    result = evaluate_kernel(
        workload_name=args.workload,
        kernel_path=str(args.kernel),
        tpu=args.tpu,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
    )
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] == "correct" else 1)


if __name__ == "__main__":
    main()
