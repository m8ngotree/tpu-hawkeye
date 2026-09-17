"""Taxonomy cell 02_vmem_tile_layout -- OPTIMIZED (expert) variant.

Identical math and output shape to naive_kernel.py, only the block shape changes:
(32, 128) instead of (4, 64). (32, 128) satisfies Pallas TPU's documented block-shape
rule -- last two dimensions divisible by 8 and 128 respectively (32 is 4x8, 128 is
1x128) -- so Mosaic can load/store each block without internal padding, and there
are 16x fewer grid steps (16 vs. naive's 256) for the same total work.

See guide.md for the (8, 128) tiling rule (and a correction: an earlier version of
this cell wrongly claimed it was dtype-dependent -- it isn't, per Pallas's docs).
"""

import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

CONFIG = {
    "name": "vmem_tile_layout_optimized",
    "M": 256,
    "N": 256,
    "block_m": 32,
    "block_n": 128,
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
    BLOCK_M, BLOCK_N = CONFIG["block_m"], CONFIG["block_n"]
    grid = (M // BLOCK_M, N // BLOCK_N)
    return pl.pallas_call(
        _kernel,
        grid=grid,
        in_specs=[pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i, j))],
        out_specs=pl.BlockSpec((BLOCK_M, BLOCK_N), lambda i, j: (i, j)),
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
    print(json.dumps({"name": CONFIG["name"], "time_ms": avg_s * 1000, "num_grid_steps": (CONFIG["M"] // CONFIG["block_m"]) * (CONFIG["N"] // CONFIG["block_n"])}))
