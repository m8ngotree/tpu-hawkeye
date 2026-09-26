"""Run every taxonomy cell's naive and optimized kernel and report status/timing (device-side when the profiler provides it).

    python scripts/verify_cells.py              # real hardware (TPU if present)
    python scripts/verify_cells.py --interpret  # CPU interpreter

A variant that fails to compile or run is reported as an error, not skipped.
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence profiler import warnings

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "accelerator-agents"))
CELLS = ROOT / "taxonomy" / "v5e"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_variant(path, name, iters):
    import jax

    try:
        from JAXBench.harness.profiler import benchmark_fn

        mod = load(path, name)
        inputs = mod.create_inputs()
        bench = benchmark_fn(mod.workload, inputs, num_warmup=5, num_iters=iters, label=name)
        out = np.asarray(bench["output"], dtype=np.float32)
        return {
            "status": "ok",
            "median_ms": bench["median_ms"],
            "timing_method": bench["timing_method"],
        }, out
    except Exception as e:
        return {"status": "error", "error": f"{type(e).__name__}: {str(e)[:300]}"}, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interpret", action="store_true")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()
    os.environ["PALLAS_INTERPRET"] = "1" if args.interpret else "0"

    import jax

    devices = str(jax.devices())
    print("devices:", devices)
    print("jax", jax.__version__)
    results = {"devices": devices, "interpret": args.interpret, "cells": {}}

    for cell in sorted(p for p in CELLS.iterdir() if p.is_dir()):
        entry = {}
        outs = {}
        for variant in ("naive", "optimized"):
            entry[variant], outs[variant] = run_variant(
                cell / f"{variant}_kernel.py", f"{cell.name}_{variant}", args.iters
            )
        if outs["naive"] is not None and outs["optimized"] is not None:
            entry["outputs_match"] = bool(
                np.allclose(outs["naive"], outs["optimized"], atol=1e-2, rtol=1e-2)
            )
        results["cells"][cell.name] = entry

    print(f"\n{'cell':<24}{'naive':>16}{'optimized':>16}{'speedup':>10}  match")
    for name, e in results["cells"].items():
        def fmt(v):
            return f"{v['median_ms'] * 1000:.1f} us" if v["status"] == "ok" else "ERROR"
        n, o = e["naive"], e["optimized"]
        sp = f"{n['median_ms'] / o['median_ms']:.2f}x" if n["status"] == o["status"] == "ok" else "-"
        print(f"{name:<24}{fmt(n):>16}{fmt(o):>16}{sp:>10}  {e.get('outputs_match', '-')}")
        for v in ("naive", "optimized"):
            if e[v]["status"] == "error":
                print(f"    {v}: {e[v]['error']}")

    out = ROOT / "results" / ("cell_verification_summary_%s.json" % ("cpu_interpret" if args.interpret else "hardware"))
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print("\nsaved", out)


if __name__ == "__main__":
    main()
