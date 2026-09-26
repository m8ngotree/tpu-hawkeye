"""Taxonomy cell 06_fused_epilogue -- NAIVE variant.

Computes `relu(A @ B + bias)` as TWO pallas_calls: one for the matmul, one for the
bias add and ReLU. The (2048, 1024) float32 matmul result (8 MB) is written to HBM by
the first kernel and read back by the second, a round trip for data that could have
stayed in VMEM.

Same math, same output as optimized_kernel.py; only whether the epilogue is fused into
the same kernel differs.
"""

import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

CONFIG = {
    "name": "fused_epilogue_naive",
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


def _matmul_kernel(a_ref, b_ref, c_ref):
    c_ref[:, :] = jnp.dot(a_ref[:, :], b_ref[:, :], preferred_element_type=jnp.float32)


def _epilogue_kernel(c_ref, bias_ref, o_ref):
    o_ref[:, :] = jnp.maximum(c_ref[:, :] + bias_ref[:, :], 0.0)


def workload(A, B, bias):
    interpret = os.environ.get("PALLAS_INTERPRET") == "1"
    M, K, N, BM = CONFIG["M"], CONFIG["K"], CONFIG["N"], CONFIG["block_m"]
    C = pl.pallas_call(
        _matmul_kernel,
        grid=(M // BM,),
        in_specs=[
            pl.BlockSpec((BM, K), lambda i: (i, 0)),
            pl.BlockSpec((K, N), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((BM, N), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        interpret=interpret,
    )(A, B)
    return pl.pallas_call(
        _epilogue_kernel,
        grid=(M // BM,),
        in_specs=[
            pl.BlockSpec((BM, N), lambda i: (i, 0)),
            pl.BlockSpec((1, N), lambda i: (0, 0)),
        ],
        out_specs=pl.BlockSpec((BM, N), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        interpret=interpret,
    )(C, bias)


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
    print(json.dumps({"name": CONFIG["name"], "time_ms": avg_s * 1000, "num_pallas_calls": 2}))
