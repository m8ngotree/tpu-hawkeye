"""Taxonomy cell 08_grouped_matmul -- NAIVE variant.

Grouped matmul: G=4 candidate weight matrices of shape (K, N); the input's M rows are
split into 8 contiguous blocks of BLOCK_M rows, and `group_id[i]` says which of the
G weight matrices block i's rows should be multiplied by -- a runtime-computed
selection, not a static one.

This naive version doesn't use scalar-prefetch at all: the ENTIRE (G, K, N) weight
tensor is kept VMEM-resident for every grid step (BlockSpec index_map always returns
(0, 0, 0)), and the per-block selection happens via a plain array index
(`w_all_ref[g]`) inside the kernel body. Correct, but VMEM footprint for the weight
tensor is O(G*K*N) -- every candidate, all the time -- when any single step only ever
needs one K*N slice. That doesn't scale: more candidates means more VMEM spent on
weights you're not using this step, regardless of how many you actually select.

See optimized_kernel.py for the scalar-prefetch alternative: O(K*N) VMEM per step
instead of O(G*K*N).
"""

import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    "name": "grouped_matmul_naive",
    "G": 4,
    "K": 64,
    "N": 64,
    "block_m": 16,
    "num_blocks": 8,
}


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    k1, k2 = jax.random.split(key)
    G, K, N = CONFIG["G"], CONFIG["K"], CONFIG["N"]
    BLOCK_M, num_blocks = CONFIG["block_m"], CONFIG["num_blocks"]
    M = BLOCK_M * num_blocks
    X = jax.random.normal(k1, (M, K), dtype=dtype)
    W = jax.random.normal(k2, (G, K, N), dtype=dtype) * 0.1
    # Round-robin group assignment per block -- exercises every group at least once.
    group_id = jnp.array([i % G for i in range(num_blocks)], dtype=jnp.int32)
    return X, W, group_id


def _kernel(x_ref, w_all_ref, group_id_ref, o_ref):
    i = pl.program_id(0)
    g = group_id_ref[i]
    w = w_all_ref[g]  # dynamic index into the fully VMEM-resident (G, K, N) tensor
    o_ref[:, :] = jnp.dot(x_ref[:, :], w, preferred_element_type=jnp.float32)


def workload(X, W, group_id):
    interpret = os.environ.get("PALLAS_INTERPRET") == "1"
    G, K, N = CONFIG["G"], CONFIG["K"], CONFIG["N"]
    BLOCK_M, num_blocks = CONFIG["block_m"], CONFIG["num_blocks"]
    M = BLOCK_M * num_blocks
    grid = (num_blocks,)
    in_specs = [
        pl.BlockSpec((BLOCK_M, K), lambda i: (i, 0)),
        pl.BlockSpec((G, K, N), lambda i: (0, 0, 0)),  # whole weight tensor, every step
        pl.BlockSpec(memory_space=pltpu.SMEM),
    ]
    out_spec = pl.BlockSpec((BLOCK_M, N), lambda i: (i, 0))
    return pl.pallas_call(
        _kernel,
        grid=grid,
        in_specs=in_specs,
        out_specs=out_spec,
        out_shape=jax.ShapeDtypeStruct((M, N), jnp.float32),
        interpret=interpret,
    )(X, W, group_id)


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
    vmem_weight_elems = CONFIG["G"] * CONFIG["K"] * CONFIG["N"]
    print(json.dumps({"name": CONFIG["name"], "time_ms": avg_s * 1000, "vmem_resident_weight_elems": vmem_weight_elems}))
