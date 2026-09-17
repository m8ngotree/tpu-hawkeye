"""Taxonomy cell 03_vectorized_vmem -- NAIVE variant.

Computes the exact same `y = x * SCALE + BIAS` as optimized_kernel.py, but via a
Python for-loop that reads and writes one lane (column) at a time:
`o_ref[:, j] = x_ref[:, j] * SCALE + BIAS` for each j. This unrolls into N separate
single-lane VMEM accesses at trace time -- each one issues at 1/128th of the VPU's
lane width, instead of one instruction operating on all 128 lanes at once. Distinct
from 02_vmem_tile_layout: that cell is about block/tile SHAPE (how work is divided
across grid steps); this cell is about whether a single block's own read/write
pattern uses the full vector width or falls back to scalar-per-lane access.

Same math, same output as optimized_kernel.py -- only the access pattern differs.
"""

import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

CONFIG = {
    "name": "vectorized_vmem_naive",
    "M": 128,
    "N": 128,
    "scale": 2.0,
    "bias": 0.5,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    M, N = CONFIG["M"], CONFIG["N"]
    X = jax.random.normal(key, (M, N), dtype=dtype)
    return (X,)


def _kernel(x_ref, o_ref):
    N = x_ref.shape[1]
    for j in range(N):
        o_ref[:, j] = x_ref[:, j] * CONFIG["scale"] + CONFIG["bias"]


def workload(X):
    interpret = os.environ.get("PALLAS_INTERPRET") == "1"
    M, N = CONFIG["M"], CONFIG["N"]
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.bfloat16),
        interpret=interpret,
    )(X)


def get_flops():
    M, N = CONFIG["M"], CONFIG["N"]
    return 2 * M * N


if __name__ == "__main__":
    import json
    import time

    import numpy as np

    inputs = create_inputs()
    fn = jax.jit(workload)
    out = fn(*inputs)
    out.block_until_ready()
    times = []
    for _ in range(20):
        t0 = time.perf_counter()
        out = fn(*inputs)
        out.block_until_ready()
        times.append(time.perf_counter() - t0)
    avg_s = float(np.mean(times))
    print(json.dumps({"name": CONFIG["name"], "time_ms": avg_s * 1000, "num_vmem_ops": CONFIG["N"]}))
