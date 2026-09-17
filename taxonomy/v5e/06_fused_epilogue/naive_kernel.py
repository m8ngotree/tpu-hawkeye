"""Taxonomy cell 06_fused_epilogue -- NAIVE variant.

Computes `relu(A @ B + bias)` as TWO separate pallas_call invocations: one for the
matmul, one for the bias-add + ReLU "epilogue." The matmul's (M, N) output `C` has to
be written to HBM after the first kernel and read back from HBM by the second --
a full extra HBM round trip for data that could have stayed in VMEM the whole time.

Same math, same output as optimized_kernel.py -- only whether the epilogue is fused
into the same kernel invocation differs.
"""

import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

CONFIG = {
    "name": "fused_epilogue_naive",
    "M": 128,
    "K": 128,
    "N": 128,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    M, K, N = CONFIG["M"], CONFIG["K"], CONFIG["N"]
    A = jax.random.normal(k1, (M, K), dtype=dtype)
    B = jax.random.normal(k2, (K, N), dtype=dtype)
    bias = jax.random.normal(k3, (N,), dtype=jnp.float32) * 0.1
    return A, B, bias


def _matmul_kernel(a_ref, b_ref, c_ref):
    c_ref[:, :] = jnp.dot(a_ref[:, :], b_ref[:, :], preferred_element_type=jnp.float32)


def _epilogue_kernel(c_ref, bias_ref, o_ref):
    o_ref[:, :] = jnp.maximum(c_ref[:, :] + bias_ref[:], 0.0)


def workload(A, B, bias):
    interpret = os.environ.get("PALLAS_INTERPRET") == "1"
    M, N = CONFIG["M"], CONFIG["N"]
    C = pl.pallas_call(
        _matmul_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        interpret=interpret,
    )(A, B)
    return pl.pallas_call(
        _epilogue_kernel,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        interpret=interpret,
    )(C, bias)


def get_flops():
    M, K, N = CONFIG["M"], CONFIG["K"], CONFIG["N"]
    return 2 * M * K * N + 2 * M * N  # matmul + (bias-add, relu)


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
    print(json.dumps({"name": CONFIG["name"], "time_ms": avg_s * 1000, "num_pallas_calls": 2}))
