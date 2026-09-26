"""Taxonomy cell 08_grouped_matmul -- OPTIMIZED (expert) variant.

Identical grouped matmul to naive_kernel.py (same G=8 weight matrices, same per-block
group assignment) using `pltpu.PrefetchScalarGridSpec`. `group_id` is a SCALAR PREFETCH
operand: resident in SMEM before the pipeline starts and readable from `index_map`
functions, so each grid step DMAs only the (K, N) slice its block selects
(`group_id_ref[i]`). Only that slice is VMEM-resident, and only the 4 selected
matrices are ever read from HBM: O(K*N) VMEM per step and traffic proportional to the
matrices used, instead of naive_kernel.py's O(G*K*N) for all 8.
"""

import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    "name": "grouped_matmul_optimized",
    "G": 8,
    "K": 512,
    "N": 512,
    "block_m": 256,
    "num_blocks": 4,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    G, K, N = CONFIG["G"], CONFIG["K"], CONFIG["N"]
    BLOCK_M, num_blocks = CONFIG["block_m"], CONFIG["num_blocks"]
    M = BLOCK_M * num_blocks
    X = jax.random.normal(k1, (M, K), dtype=dtype)
    W = jax.random.normal(k2, (G, K, N), dtype=dtype) * 0.05
    # Each block selects one of the G weight matrices; only num_blocks < G are used.
    group_id = jnp.array([(2 * i) % G for i in range(num_blocks)], dtype=jnp.int32)
    return X, W, group_id


def _kernel(group_id_ref, x_ref, w_ref, o_ref):
    # group_id_ref is the scalar-prefetch ref, already consulted by index_map to pick
    # which weight block was DMA'd in; the body just uses whatever arrived.
    o_ref[:, :] = jnp.dot(x_ref[:, :], w_ref[0, :, :], preferred_element_type=jnp.float32)


def workload(X, W, group_id):
    interpret = os.environ.get("PALLAS_INTERPRET") == "1"
    K, N = CONFIG["K"], CONFIG["N"]
    BLOCK_M, num_blocks = CONFIG["block_m"], CONFIG["num_blocks"]
    M = BLOCK_M * num_blocks

    grid_spec = pltpu.PrefetchScalarGridSpec(
        num_scalar_prefetch=1,
        grid=(num_blocks,),
        in_specs=[
            pl.BlockSpec((BLOCK_M, K), lambda i, group_id_ref: (i, 0)),
            pl.BlockSpec((1, K, N), lambda i, group_id_ref: (group_id_ref[i], 0, 0)),
        ],
        out_specs=pl.BlockSpec((BLOCK_M, N), lambda i, group_id_ref: (i, 0)),
    )
    return pl.pallas_call(
        _kernel,
        grid_spec=grid_spec,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        interpret=interpret,
    )(group_id, X, W)


def get_flops():
    K, N = CONFIG["K"], CONFIG["N"]
    BLOCK_M, num_blocks = CONFIG["block_m"], CONFIG["num_blocks"]
    return 2 * (BLOCK_M * num_blocks) * K * N


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
    vmem_weight_elems = CONFIG["K"] * CONFIG["N"]
    print(json.dumps({"name": CONFIG["name"], "time_ms": avg_s * 1000, "vmem_resident_weight_elems": vmem_weight_elems}))
