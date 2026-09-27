"""Screen JAXBench workloads (no agent) to find ones worth running experiments on.

    python scripts/screen_workloads.py                       # all workloads, on the TPU
    python scripts/screen_workloads.py --workloads 12p_RMSNorm,8p_GEMM

For each workload's baseline it measures the device time and reports:
  * pct_of_roofline_limit: how close the baseline already is to the chip's memory or compute limit
    (low = room for a better kernel; ~80+ = little to gain);
  * whether the workload is memory- or compute-bound;
  * output_max_abs and zeros_pass: the size of the reference output and whether an all-zeros output
    would pass the harness tolerance (atol = rtol = 1e-2), which makes correctness checking weak;
  * permutation_inputs: 2-D integer inputs that are permutations (page tables and the like), which
    a kernel could wrongly assume are fixed.
Each workload runs in its own process with a timeout, so one failure or hang does not stop the rest.
Results are printed as a table sorted by roofline efficiency and saved to results/screen_<tpu>.json.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "external" / "accelerator-agents" / "JAXBench" / "benchmark"
sys.path.insert(0, str(ROOT / "external" / "accelerator-agents"))


def worker(workload: str, tpu: str) -> dict:
    import jax
    import jax.numpy as jnp
    import numpy as np

    from JAXBench.harness.loader import load_module
    from JAXBench.harness.profiler import benchmark_fn
    from JAXBench.harness.runner import get_flop_count
    from JAXBench.harness.tpu_specs import TPU_SPECS

    spec = TPU_SPECS[tpu]
    mod = load_module(str(BENCH / workload / "baseline.py"), f"{workload}.screen")
    create = mod.create_inputs
    inputs = create(dtype=jnp.bfloat16) if "dtype" in create.__code__.co_varnames else create()
    inputs = inputs if isinstance(inputs, (list, tuple)) else (inputs,)
    skip_jit = bool(getattr(mod, "_skip_jit", False))

    bench = benchmark_fn(mod.workload, inputs, num_warmup=3, num_iters=20, skip_jit=skip_jit,
                         label=f"screen_{workload}")
    out = bench.pop("output")
    seconds = bench["median_ms"] / 1000.0
    flops = get_flop_count(mod, mod.workload, inputs, skip_jit)

    leaves = jax.tree_util.tree_leaves
    in_bytes = sum(x.size * x.dtype.itemsize for x in leaves(inputs))
    out_bytes = sum(np.asarray(x).size * np.asarray(x).dtype.itemsize for x in leaves(out))
    min_bytes = in_bytes + out_bytes
    peak_flops, peak_bw = spec["peak_tflops_bf16"] * 1e12, spec["hbm_bandwidth_gbs"] * 1e9
    limit_seconds = max(flops / peak_flops, min_bytes / peak_bw)
    intensity = flops / min_bytes if flops else 0.0

    ref = [np.asarray(x, dtype=np.float32) for x in leaves(out)]
    max_abs = max(float(np.max(np.abs(r))) for r in ref) if ref else 0.0
    perms = 0
    for x in inputs:
        arr = np.asarray(x) if hasattr(x, "dtype") and hasattr(x, "shape") else None
        if (arr is not None and arr.ndim == 2 and np.issubdtype(arr.dtype, np.integer)
                and np.array_equal(np.sort(arr.ravel()), np.arange(arr.size))):
            perms += 1
    return {
        "workload": workload,
        "baseline_ms": bench["median_ms"],
        "timing_method": bench["timing_method"],
        "tflops": round(flops / seconds / 1e12, 2) if flops else None,
        "min_hbm_gb": round(min_bytes / 1e9, 3),
        "limit": "compute" if flops / peak_flops > min_bytes / peak_bw else "memory",
        "arithmetic_intensity": round(intensity, 1),
        "pct_of_roofline_limit": round(limit_seconds / seconds * 100, 1),
        "output_max_abs": max_abs,
        "zeros_pass": bool(all(np.allclose(r, 0, atol=1e-2, rtol=1e-2) for r in ref)),
        "permutation_inputs": perms,
        "has_optimized_key": (BENCH / workload / "optimized.py").exists(),
        "skip_jit": skip_jit,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", help="comma-separated names (default: all)")
    ap.add_argument("--tpu", default="v5e")
    ap.add_argument("--timeout", type=int, default=900, help="seconds per workload")
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.worker:
        print("SCREEN_JSON " + json.dumps(worker(args.worker, args.tpu)))
        return

    names = (args.workloads.split(",") if args.workloads
             else sorted((p.name for p in BENCH.iterdir() if (p / "baseline.py").exists()),
                         key=lambda n: int("".join(c for c in n.split("_")[0] if c.isdigit()))))
    rows = []
    for name in names:
        try:
            proc = subprocess.run([sys.executable, __file__, "--worker", name, "--tpu", args.tpu],
                                  capture_output=True, text=True, errors="replace", timeout=args.timeout)
            line = next((l for l in reversed(proc.stdout.splitlines()) if l.startswith("SCREEN_JSON ")), None)
            row = json.loads(line[len("SCREEN_JSON "):]) if line else {
                "workload": name, "error": (proc.stdout + proc.stderr)[-300:].replace("\n", " ")}
        except subprocess.TimeoutExpired:
            row = {"workload": name, "error": f"timed out after {args.timeout}s"}
        rows.append(row)
        print(f"done {name}: " + (row.get("error") or f"{row['pct_of_roofline_limit']}% of {row['limit']} limit"), flush=True)

    out = ROOT / "results" / f"screen_{args.tpu}.json"
    out.write_text(json.dumps(rows, indent=2))
    ok = sorted((r for r in rows if "error" not in r), key=lambda r: r["pct_of_roofline_limit"])
    print(f"\n{'workload':<52}{'ms':>9}{'%roofline':>10}{'bound':>9}{'out max':>10}{'0s pass':>8}{'perm':>5}{'key':>4}")
    for r in ok:
        print(f"{r['workload']:<52}{r['baseline_ms']:>9.2f}{r['pct_of_roofline_limit']:>10}{r['limit']:>9}"
              f"{r['output_max_abs']:>10.3g}{str(r['zeros_pass']):>8}{r['permutation_inputs']:>5}"
              f"{'y' if r['has_optimized_key'] else '':>4}")
    for r in rows:
        if "error" in r:
            print(f"ERROR {r['workload']}: {r['error'][:160]}")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
