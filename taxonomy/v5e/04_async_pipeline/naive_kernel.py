"""Taxonomy cell 04_async_pipeline -- NAIVE variant.

Processes a (512, 128) bf16 array in 4 row-blocks via `pltpu.emit_pipeline` with
`no_pipelining=True`: each block's HBM->VMEM copy-in, compute, and VMEM->HBM copy-out
run synchronously, one block at a time, with no overlap between a block's DMA and the
compute of adjacent blocks.

Body, grid and BlockSpecs are identical to optimized_kernel.py; only the
`no_pipelining` flag differs.

When run without a TPU device, `emit_pipeline` requires the `pallas_call` to be
wrapped in `jax.sharding.use_abstract_mesh` with an `AbstractDevice` naming a TPU
generation (see `workload()` and guide.md).
"""

import contextlib
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    "name": "async_pipeline_naive",
    "M": 512,
    "N": 128,
    "block_m": 128,
    "scale": 2.0,
    "bias": 0.5,
}

# Names the target TPU generation so emit_pipeline works on a host without a TPU.
def _abstract_tpu_mesh():
    return jax.sharding.AbstractMesh(
        (),
        (),
        abstract_device=jax.sharding.AbstractDevice(device_kind="TPU v5e", num_cores=1, platform="tpu"),
    )


def create_inputs(dtype=jnp.bfloat16):
    key = jax.random.key(0)
    M, N = CONFIG["M"], CONFIG["N"]
    X = jax.random.normal(key, (M, N), dtype=dtype)
    return (X,)


def _outer_kernel(x_hbm_ref, o_hbm_ref):
    def body(x_vmem, o_vmem):
        o_vmem[:, :] = x_vmem[:, :] * CONFIG["scale"] + CONFIG["bias"]

    M, N, BLOCK_M = CONFIG["M"], CONFIG["N"], CONFIG["block_m"]
    pltpu.emit_pipeline(
        body,
        grid=(M // BLOCK_M,),
        in_specs=[pl.BlockSpec((BLOCK_M, N), lambda i: (i, 0))],
        out_specs=pl.BlockSpec((BLOCK_M, N), lambda i: (i, 0)),
        no_pipelining=True,  # <-- the ONE line that differs from optimized_kernel.py
    )(x_hbm_ref, o_hbm_ref)


def workload(X):
    interpret = os.environ.get("PALLAS_INTERPRET") == "1"
    M, N = CONFIG["M"], CONFIG["N"]
    mesh_ctx = (
        contextlib.nullcontext()
        if jax.default_backend() == "tpu"
        else jax.sharding.use_abstract_mesh(_abstract_tpu_mesh())
    )
    with mesh_ctx:
        return pl.pallas_call(
            _outer_kernel,
            in_specs=[pl.BlockSpec(memory_space=pl.ANY)],
            out_specs=pl.BlockSpec(memory_space=pl.ANY),
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
    print(json.dumps({"name": CONFIG["name"], "time_ms": avg_s * 1000, "no_pipelining": True}))
