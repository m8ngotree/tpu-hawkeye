"""Taxonomy cell 06_lane_reduction -- NAIVE variant.

Row-sum of a 128x128 bf16 array (`y[i, 0] = sum_j x[i, j]`) via a Python for-loop that
reads one lane (column) at a time and accumulates in a running scalar-per-row sum --
128 separate single-lane reads plus 128 adds, instead of one reduction instruction.
Same failure mode as `03_vectorized_vmem`, specific to reductions: Mosaic has a
native cross-lane reduction op, but only if the code is written as one, not as a
manually unrolled accumulation loop.

Same math, same output as optimized_kernel.py -- only the reduction strategy differs.
"""

import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

CONFIG = {
    "name": "lane_reduction_naive",
    "M": 128,
    "N": 128,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    M, N = CONFIG["M"], CONFIG["N"]
    X = jax.random.normal(key, (M, N), dtype=dtype)
    return (X,)


def _kernel(x_ref, o_ref):
    N = x_ref.shape[1]
    acc = jnp.zeros((x_ref.shape[0], 1), dtype=jnp.float32)
    for j in range(N):
        acc = acc + x_ref[:, j : j + 1].astype(jnp.float32)
    o_ref[:, :] = acc


def workload(X):
    interpret = os.environ.get("PALLAS_INTERPRET") == "1"
    M = CONFIG["M"]
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((M, 1), jnp.float32),
        interpret=interpret,
    )(X)


def get_flops():
    M, N = CONFIG["M"], CONFIG["N"]
    return M * N  # one add per element


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
    print(json.dumps({"name": CONFIG["name"], "time_ms": avg_s * 1000, "num_reduce_ops": CONFIG["N"]}))
