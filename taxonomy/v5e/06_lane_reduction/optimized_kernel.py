"""Taxonomy cell 06_lane_reduction -- OPTIMIZED (expert) variant.

Identical row-sum, computed with `jnp.sum(x_ref[:, :], axis=1)` -- one native
cross-lane reduction instruction instead of naive_kernel.py's 128 single-lane
accumulation steps.
"""

import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

CONFIG = {
    "name": "lane_reduction_optimized",
    "M": 128,
    "N": 128,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    M, N = CONFIG["M"], CONFIG["N"]
    X = jax.random.normal(key, (M, N), dtype=dtype)
    return (X,)


def _kernel(x_ref, o_ref):
    o_ref[:, :] = jnp.sum(x_ref[:, :].astype(jnp.float32), axis=1, keepdims=True)


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
    return M * N


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
    print(json.dumps({"name": CONFIG["name"], "time_ms": avg_s * 1000, "num_reduce_ops": 1}))
