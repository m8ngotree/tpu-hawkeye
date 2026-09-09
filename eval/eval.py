"""The evaluate_kernel tool, exposed as a CLI -- this is what the OpenHands agent
actually calls, since OpenHands drives everything through bash commands in its
sandboxed workspace rather than function-calling (unlike a Claude Agent SDK /
Messages API tool-use setup). Matches Fig. 48's eval.py exactly in role.

Placed inside each generated workspace (see agent/workspace.py) so the agent can run:

    python eval.py --workload 12p_RMSNorm --kernel kernel.py

and get back correctness + timing without needing to know anything about
agent/runner.py or JAXBench's internal layout.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.runner import run_kernel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True, help="JAXBench workload name, e.g. 12p_RMSNorm")
    parser.add_argument("--kernel", required=True, type=Path, help="path to your kernel.py (must define workload(*inputs))")
    parser.add_argument("--tpu", default="v5e", choices=["v5e", "v6e"])
    parser.add_argument("--interpret", action="store_true", help="run Pallas kernels in CPU interpreter mode (no TPU needed)")
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=50)
    args = parser.parse_args()

    result = run_kernel(
        workload_name=args.workload,
        kernel_path=args.kernel,
        tpu=args.tpu,
        interpret=args.interpret,
        num_warmup=args.num_warmup,
        num_iters=args.num_iters,
    )

    print(json.dumps(result.raw, indent=2))
    sys.exit(0 if result.status == "correct" else 1)


if __name__ == "__main__":
    main()
