"""Taxonomy cell 06_fused_epilogue -- OPTIMIZED (expert) variant.

Computes the identical `relu(A @ B + bias)` as ONE pallas_call: the bias add and ReLU
are applied right after the matmul, while each (256, 1024) result block is still in
VMEM, before anything is written to HBM. No intermediate array, no extra round trip.
"""

import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

CONFIG = {
    "name": "fused_epilogue_optimized",
    "M": 2048,
    "K": 1024,
    "N": 1024,
    "block_m": 256,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2, k3 = jax.random.split(key, 3)
    M, K, N = CONFIG["M"], CONFIG["K"], CONFIG["N"]
    A = jax.random.normal(k1, (M, K), dtype=dtype)
    B = jax.random.normal(k2, (K, N), dtype=dtype) * 0.03
    bias = jax.random.normal(k3, (1, N), dtype=jnp.float32) * 0.1
    return A, B, bias


def _fused_kernel(a_ref, b_ref, bias_ref, o_ref):
    c = jnp.dot(a_ref[:, :], b_ref[:, :], preferred_element_type=jnp.float32)
    o_ref[:, :] = jnp.maximum(c + bias_ref[:, :], 0.0)


def workload(A, B, bias):
    interpret = os.environ.get("PALLAS_INTERPRET") == "1"
    M, K, N, BM = CONFIG["M"], CONFIG["K"], CONFIG["N"], CONFIG["block_m"]
    return pl.pallas_call(
        _fused_kernel,
        grid=(M // BM,),
        in_specs=[
            pl.BlockSpec((BM, K), lambda i: (i, 0)),
            pl.BlockSpec((K, N), lambda i: (0, 0)),
            pl.BlockSpec((1, N), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((BM, N), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        interpret=interpret,
    )(A, B, bias)


def get_flops():
    M, K, N = CONFIG["M"], CONFIG["K"], CONFIG["N"]
    return 2 * M * K * N + 2 * M * N


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
    print(json.dumps({"name": CONFIG["name"], "time_ms": avg_s * 1000, "num_pallas_calls": 1}))
