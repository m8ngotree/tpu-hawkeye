"""Sweep block size and input buffer depth for the 05_producer_consumer kernel to find
where deeper buffering changes the device time.

    python scripts/probe_buffering.py            # on a TPU
    python scripts/probe_buffering.py --interpret
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "accelerator-agents"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--interpret", action="store_true")
    ap.add_argument("--rows", type=int, default=16384)
    ap.add_argument("--cols", type=int, default=1024)
    args = ap.parse_args()
    os.environ["PALLAS_INTERPRET"] = "1" if args.interpret else "0"

    import jax
    from JAXBench.harness.profiler import benchmark_fn

    path = ROOT / "taxonomy" / "v5e" / "05_producer_consumer" / "optimized_kernel.py"
    spec = importlib.util.spec_from_file_location("cell05", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("devices:", jax.devices(), "jax", jax.__version__)
    print(f"array ({args.rows}, {args.cols}) bf16")
    print(f"{'block_m':>8} {'blocks':>7} {'bufs=2 (us)':>13} {'bufs=3 (us)':>13} {'bufs=4 (us)':>13}")
    for block_m in (8, 16, 32, 64, 128, 256, 512):
        row = []
        for bufs in (2, 3, 4):
            mod.CONFIG.update(M=args.rows, N=args.cols, block_m=block_m, in_buffer_count=bufs)
            try:
                inputs = mod.create_inputs()
                b = benchmark_fn(mod.workload, inputs, num_warmup=5, num_iters=30, label=f"p{block_m}_{bufs}")
                row.append(f"{b['median_ms'] * 1000:.1f}")
            except Exception as e:
                row.append("ERR")
                print(f"  block_m={block_m} bufs={bufs}: {type(e).__name__}: {str(e)[:150]}")
        print(f"{block_m:>8} {args.rows // block_m:>7} {row[0]:>13} {row[1]:>13} {row[2]:>13}")


if __name__ == "__main__":
    main()
