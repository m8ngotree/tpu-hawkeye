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
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
if (HERE / "JAXBench").exists():
    sys.path.insert(0, str(HERE))
else:
    sys.path.insert(0, str(HERE.parent / "external" / "accelerator-agents"))


def _record(result: dict, kernel_path: Path) -> None:
    """Keep a log of evaluations and a copy of the fastest correct kernel seen so far."""
    speedup = result.get("speedup_vs_baseline")
    reference = HERE / "baseline.py"
    is_reference = reference.exists() and Path(kernel_path).read_text() == reference.read_text()
    entry = {"time": time.time(), "status": result.get("status"), "speedup_vs_baseline": speedup,
             "is_reference": is_reference}
    with open(HERE / "eval_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
    if result.get("status") != "correct" or speedup is None:
        return
    best_file = HERE / "best_score.json"
    best = json.loads(best_file.read_text())["speedup_vs_baseline"] if best_file.exists() else -1
    if speedup > best:
        shutil.copy(kernel_path, HERE / "best_kernel.py")
        best_file.write_text(json.dumps({"speedup_vs_baseline": speedup}))


def _diagnosis(result: dict, workload: str, tpu: str) -> dict | None:
    """Roofline view of the candidate kernel's timing.

    Minimum HBM traffic is the workload's inputs plus its output, each moved once. Compared
    against the chip's peak bandwidth and peak FLOPs this says whether the workload is
    limited by memory or by compute, and how close the kernel is to that limit.
    """
    kernel = result.get("kernel") or {}
    ms = kernel.get("median_ms")
    if not ms:
        return None
    try:
        import jax
        import jax.numpy as jnp
        import JAXBench
        from JAXBench.harness.loader import load_module
        from JAXBench.harness.tpu_specs import TPU_SPECS

        path = Path(JAXBench.__file__).parent / "benchmark" / workload / "baseline.py"
        mod = load_module(str(path), f"{workload}.diag")
        create = mod.create_inputs
        inputs = create(dtype=jnp.bfloat16) if "dtype" in create.__code__.co_varnames else create()
        inputs = inputs if isinstance(inputs, (list, tuple)) else (inputs,)
        in_bytes = sum(x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(inputs))
        out = jax.eval_shape(mod.workload, *inputs)
        out_bytes = sum(x.size * x.dtype.itemsize for x in jax.tree_util.tree_leaves(out))
    except Exception:  # noqa: BLE001 - diagnosis is best-effort, never fail an evaluation
        return None

    spec = TPU_SPECS[tpu]
    seconds = ms / 1000.0
    min_bytes = in_bytes + out_bytes
    flops = (kernel.get("tflops") or 0.0) * 1e12 * seconds
    peak_flops = spec["peak_tflops_bf16"] * 1e12
    peak_bw = spec["hbm_bandwidth_gbs"] * 1e9
    diag = {
        "min_hbm_traffic_mb": round(min_bytes / 1e6, 2),
        "achieved_hbm_gbs": round(min_bytes / seconds / 1e9, 1),
        "hbm_bandwidth_pct_of_peak": round(min_bytes / seconds / peak_bw * 100, 1),
        "mxu_pct_of_peak": kernel.get("utilization_pct"),
    }
    if flops > 0:
        intensity = flops / min_bytes
        ridge = peak_flops / peak_bw
        limit_seconds = max(flops / peak_flops, min_bytes / peak_bw)
        diag.update(
            arithmetic_intensity_flop_per_byte=round(intensity, 1),
            ridge_point_flop_per_byte=round(ridge, 1),
            workload_limit="memory-bound" if intensity < ridge else "compute-bound",
            pct_of_roofline_limit=round(limit_seconds / seconds * 100, 1),
        )
    else:
        diag.update(workload_limit="memory-bound (no FLOP count available)",
                    pct_of_roofline_limit=round(min_bytes / peak_bw / seconds * 100, 1))
    return diag


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
    if result.get("status") == "correct":
        diagnosis = _diagnosis(result, args.workload, args.tpu)
        if diagnosis:
            result["diagnosis"] = diagnosis
    if (HERE / "JAXBench").exists():  # inside an agent workspace
        _record(result, args.kernel)
    print(json.dumps(result, indent=2))
    sys.exit(0 if result["status"] == "correct" else 1)


if __name__ == "__main__":
    main()
