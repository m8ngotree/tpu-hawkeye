"""Look at what the JAX profiler records for a workload, to decide which performance
signals (achieved bandwidth, per-op device time) can be shown to the agent.

    python scripts/inspect_profile.py --workload 12p_RMSNorm
    python scripts/inspect_profile.py --workload 12p_RMSNorm --kernel path/to/kernel.py

Run on the TPU VM. Prints, for each variant (baseline, and the kernel if given):
  * XLA cost analysis (flops, bytes accessed) and the roofline numbers derived from it
  * the trace's event names grouped by process/thread, with total and mean duration
Everything is also written to profile_inspect.json for later reading.
"""

import argparse
import glob
import gzip
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "external" / "accelerator-agents"))

os.environ.setdefault("PALLAS_INTERPRET", "0")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from JAXBench.harness.loader import load_module  # noqa: E402
from JAXBench.harness.tpu_specs import TPU_SPECS  # noqa: E402

BENCH = REPO / "external" / "accelerator-agents" / "JAXBench" / "benchmark"


def make_inputs(mod):
    create = mod.create_inputs
    inputs = create(dtype=jnp.bfloat16) if "dtype" in create.__code__.co_varnames else create()
    return inputs if isinstance(inputs, (list, tuple)) else (inputs,)


def cost(fn, inputs):
    try:
        cost = jax.jit(fn).lower(*inputs).compile().cost_analysis()
        if isinstance(cost, list):
            cost = cost[0] if cost else {}
        return dict(cost)
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def trace_events(fn, inputs, iters=10):
    run = jax.jit(fn)
    for _ in range(3):
        jax.block_until_ready(run(*inputs))
    trace_dir = f"/tmp/inspect_profile_{os.getpid()}"
    shutil.rmtree(trace_dir, ignore_errors=True)
    os.makedirs(trace_dir)
    with jax.profiler.trace(trace_dir, create_perfetto_link=False, create_perfetto_trace=True):
        for _ in range(iters):
            jax.block_until_ready(run(*inputs))
    files = glob.glob(f"{trace_dir}/**/perfetto_trace.json.gz", recursive=True)
    if not files:
        return None, trace_dir
    with gzip.open(files[0], "rt") as f:
        data = json.load(f)
    return (data.get("traceEvents", data) if isinstance(data, dict) else data), trace_dir


def summarize(events, iters):
    names = {}
    for e in events:
        if isinstance(e, dict) and e.get("ph") == "M":
            args = e.get("args", {})
            if e.get("name") == "process_name":
                names[("p", e.get("pid"))] = args.get("name")
            elif e.get("name") == "thread_name":
                names[("t", e.get("pid"), e.get("tid"))] = args.get("name")
    groups = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    for e in events:
        if not isinstance(e, dict) or e.get("dur", 0) <= 0:
            continue
        proc = names.get(("p", e.get("pid")), str(e.get("pid")))
        thread = names.get(("t", e.get("pid"), e.get("tid")), str(e.get("tid")))
        rec = groups[f"{proc} / {thread}"][e.get("name", "")]
        rec[0] += 1
        rec[1] += e["dur"] / 1000.0
    out = {}
    for group, ops in groups.items():
        rows = sorted(ops.items(), key=lambda kv: -kv[1][1])[:12]
        out[group] = [
            {"name": n[:90], "count": c, "total_ms": round(t, 4), "mean_ms": round(t / c, 4)}
            for n, (c, t) in rows
        ]
    return out


def inspect(label, fn, inputs, spec):
    print(f"\n=== {label} ===")
    c = cost(fn, inputs)
    flops = c.get("flops", 0)
    nbytes = c.get("bytes accessed", 0)
    print("cost_analysis:", {k: v for k, v in c.items() if k in ("flops", "bytes accessed", "error")})
    events, trace_dir = trace_events(fn, inputs)
    summary = summarize(events, 10) if events else {}
    for group, rows in summary.items():
        print(f"\n[{group}]")
        for r in rows:
            print(f"  {r['total_ms']:9.4f} ms total  {r['count']:4d}x  {r['mean_ms']:8.4f} ms  {r['name']}")
    shutil.rmtree(trace_dir, ignore_errors=True)
    info = {"cost": c, "trace": summary}
    if flops and nbytes:
        info["arithmetic_intensity_flop_per_byte"] = round(flops / nbytes, 2)
        info["ridge_flop_per_byte"] = round(spec["peak_tflops_bf16"] * 1e12 / (spec["hbm_bandwidth_gbs"] * 1e9), 1)
        print(f"\narithmetic intensity {info['arithmetic_intensity_flop_per_byte']} flop/byte "
              f"(ridge {info['ridge_flop_per_byte']})")
    return info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workload", required=True)
    ap.add_argument("--kernel", type=Path)
    ap.add_argument("--tpu", default="v5e")
    ap.add_argument("--out", type=Path, default=Path("profile_inspect.json"))
    args = ap.parse_args()

    spec = TPU_SPECS[args.tpu]
    print("devices:", jax.devices())
    base = load_module(str(BENCH / args.workload / "baseline.py"), "baseline")
    inputs = make_inputs(base)
    report = {"baseline": inspect("baseline", base.workload, inputs, spec)}
    if args.kernel:
        kern = load_module(str(args.kernel), "kernel")
        report["kernel"] = inspect("kernel", kern.workload, inputs, spec)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
