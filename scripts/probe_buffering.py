"""Sweep block size and input buffer depth for a pipelined kernel, with a pure-copy body
(DMA-bound) and a small-arithmetic body (compute-bound), to find where deeper buffering
changes device time.

    python scripts/probe_buffering.py            # on a TPU
    python scripts/probe_buffering.py --interpret --rows 1024   # CPU smoke test

Results are printed and saved to results/probe_buffering.txt.
"""

import argparse
import contextlib
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # silence profiler import warnings

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "accelerator-agents"))


def copy_body(x_vmem, o_vmem):
    o_vmem[:, :] = x_vmem[:, :]


def scale_body(x_vmem, o_vmem):
    o_vmem[:, :] = x_vmem[:, :] * 2.0 + 0.5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interpret", action="store_true")
    ap.add_argument("--rows", type=int, default=16384)
    ap.add_argument("--cols", type=int, default=1024)
    args = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import tpu as pltpu
    from JAXBench.harness.profiler import benchmark_fn

    def build(block_m, bufs, body):
        def outer(x_hbm, o_hbm):
            pltpu.emit_pipeline(
                body,
                grid=(args.rows // block_m,),
                in_specs=[pl.BlockSpec((block_m, args.cols), lambda i: (i, 0),
                                       pipeline_mode=pl.Buffered(buffer_count=bufs))],
                out_specs=pl.BlockSpec((block_m, args.cols), lambda i: (i, 0)),
            )(x_hbm, o_hbm)

        def workload(X):
            ctx = contextlib.nullcontext()
            if jax.default_backend() != "tpu":
                ctx = jax.sharding.use_abstract_mesh(jax.sharding.AbstractMesh(
                    (), (), abstract_device=jax.sharding.AbstractDevice(
                        device_kind="TPU v5e", num_cores=1, platform="tpu")))
            with ctx:
                return pl.pallas_call(
                    outer,
                    in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
                    out_specs=pl.BlockSpec(memory_space=pl.ANY),
                    out_shape=jax.ShapeDtypeStruct((args.rows, args.cols), jnp.bfloat16),
                    interpret=args.interpret,
                )(X)
        return workload

    X = jax.random.normal(jax.random.key(0), (args.rows, args.cols), dtype=jnp.bfloat16)
    lines = [f"devices: {jax.devices()}  jax {jax.__version__}",
             f"array ({args.rows}, {args.cols}) bf16; median device time in us"]
    for name, body in (("copy (DMA-bound)", copy_body), ("scale (has arithmetic)", scale_body)):
        lines.append(f"\n{name}")
        lines.append(f"{'block_m':>8} {'blocks':>7} {'bufs=2':>10} {'bufs=3':>10} {'bufs=4':>10}")
        for block_m in (8, 16, 32, 64, 128, 256, 512):
            row = []
            for bufs in (2, 3, 4):
                try:
                    b = benchmark_fn(build(block_m, bufs, body), (X,), num_warmup=5,
                                     num_iters=30, label=f"p{block_m}_{bufs}")
                    row.append(f"{b['median_ms'] * 1000:.1f}")
                except Exception as e:
                    row.append("ERR")
                    lines.append(f"  block_m={block_m} bufs={bufs}: {type(e).__name__}: {str(e)[:150]}")
            lines.append(f"{block_m:>8} {args.rows // block_m:>7} {row[0]:>10} {row[1]:>10} {row[2]:>10}")

    text = "\n".join(lines)
    print(text)
    out = ROOT / "results" / "probe_buffering.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text(text + "\n")
    print("\nsaved", out)


if __name__ == "__main__":
    main()
