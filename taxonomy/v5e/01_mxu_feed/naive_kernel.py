"""Taxonomy cell 01_mxu_feed -- NAIVE variant.

Computes the exact same matmul as optimized_kernel.py, but WITHOUT jnp.dot or
jax.lax.dot_general: elementwise multiply + reduce instead. This is the mistake the
cell exists to catch -- Mosaic (Pallas's TPU compiler) recognizes jnp.dot/dot_general
as the pattern to lower onto the MXU (the systolic array); an equivalent
broadcast-multiply-then-sum does the same math but Mosaic has no reason to route it
through the MXU, so it runs entirely on the VPU instead. Numerically identical
output, dramatically lower throughput on real hardware.

Deliberately small (128x128x128) -- this is a unit test for one technique, not a
competitive kernel. See guide.md for why and what to do about it.
"""

import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

CONFIG = {
    "name": "mxu_feed_naive",
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
    a = a_ref[:, :].astype(jnp.float32)
    b = b_ref[:, :].astype(jnp.float32)
    # Same math as A @ B, but expressed as broadcast-multiply + sum instead of a
    # dot -- Mosaic won't recognize this as a matmul, so it never touches the MXU.
    acc = jnp.sum(a[:, :, None] * b[None, :, :], axis=1)
    o_ref[:, :] = acc.astype(o_ref.dtype)


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
