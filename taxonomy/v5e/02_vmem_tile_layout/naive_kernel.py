"""Taxonomy cell 02_vmem_tile_layout -- NAIVE variant.

Elementwise `y = x * SCALE + BIAS` over a 1024x1024 bf16 array, gridded with the
smallest block Pallas TPU accepts, (8, 128). A block's last two dimensions must be
divisible by 8 and 128 (or equal the array's dimensions); shapes that violate this are
rejected at lowering with a ValueError. Legal but tiny blocks compile, and cost one
grid step (with its fixed DMA and loop overhead) per 8x128 tile: 1024 steps here,
versus 8 in optimized_kernel.py.

Same math and output shape as optimized_kernel.py; only the block shape differs.
"""

import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

CONFIG = {
    "name": "vmem_tile_layout_naive",
    "M": 1024,
    "N": 1024,
    "block_m": 8,
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
    return 2 * M * N  # one multiply + one add per element


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
