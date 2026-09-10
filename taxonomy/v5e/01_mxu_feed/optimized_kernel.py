"""Taxonomy cell 01_mxu_feed -- OPTIMIZED (expert) variant.

Same math, same shapes as naive_kernel.py, but via jnp.dot inside the Pallas kernel
body with preferred_element_type=jnp.float32. Mosaic recognizes jnp.dot/dot_general
and lowers it onto the MXU (the systolic array) instead of the VPU -- this is the one
syntax change the whole cell exists to demonstrate. Everything else (shapes, dtypes,
grid) is held identical to naive_kernel.py so the delta is attributable to this one
change alone.

preferred_element_type=jnp.float32 matters: inputs are bf16, and without forcing a
float32 accumulator the MXU matmul accumulates in bf16, which is both less accurate
and not how the MXU's actual accumulation path works (see guide.md).
"""

import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

CONFIG = {
    "name": "mxu_feed_optimized",
    "M": 128,
    "K": 128,
    "N": 128,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    M, K, N = CONFIG["M"], CONFIG["K"], CONFIG["N"]
    A = jax.random.normal(k1, (M, K), dtype=dtype)
    B = jax.random.normal(k2, (K, N), dtype=dtype)
    return A, B


def _kernel(a_ref, b_ref, o_ref):
    a = a_ref[:, :]
    b = b_ref[:, :]
    o_ref[:, :] = jnp.dot(a, b, preferred_element_type=jnp.float32)


def workload(A, B):
    interpret = os.environ.get("PALLAS_INTERPRET") == "1"
    M, N = CONFIG["M"], CONFIG["N"]
    return pl.pallas_call(
        _kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        interpret=interpret,
    )(A, B)


def get_flops():
    M, K, N = CONFIG["M"], CONFIG["K"], CONFIG["N"]
    return 2 * M * K * N


if __name__ == "__main__":
    import json
    import time

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
    import numpy as np

    avg_s = float(np.mean(times))
    tflops = get_flops() / avg_s / 1e12
    print(json.dumps({"name": CONFIG["name"], "time_ms": avg_s * 1000, "tflops": tflops}))
