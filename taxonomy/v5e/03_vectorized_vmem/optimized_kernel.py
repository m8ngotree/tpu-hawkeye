"""Taxonomy cell 03_vectorized_vmem -- OPTIMIZED (expert) variant.

Identical math and output to naive_kernel.py: `o_ref[:, :] = x_ref[:, :] * SCALE +
BIAS` in one vectorized statement, operating on the full (M, N) block -- and
therefore all 128 lanes -- in a single VPU instruction, instead of naive_kernel.py's
N separate single-lane accesses.
"""

import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

CONFIG = {
    "name": "vectorized_vmem_optimized",
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
    o_ref[:, :] = x_ref[:, :] * CONFIG["scale"] + CONFIG["bias"]


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
    print(json.dumps({"name": CONFIG["name"], "time_ms": avg_s * 1000, "num_vmem_ops": 1}))
