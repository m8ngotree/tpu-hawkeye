"""Taxonomy cell 05_producer_consumer -- OPTIMIZED (expert) variant.

Body and grid identical to naive_kernel.py; the input BlockSpec uses
`pipeline_mode=pl.Buffered(buffer_count=3)` instead of 2 (triple buffering). The DMA
producer can run two blocks ahead of the compute consumer, absorbing a per-block
transfer that takes longer than a block's compute.

`buffer_count > 2` is supported on input buffered refs only, so `out_spec` is left at
its default in both variants.
"""

import contextlib
import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    "name": "producer_consumer_optimized",
    "M": 1024,
    "N": 128,
    "block_m": 128,
    "in_buffer_count": 3,
    "scale": 2.0,
    "bias": 0.5,
}

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
    in_spec = pl.BlockSpec(
        (BLOCK_M, N), lambda i: (i, 0),
        pipeline_mode=pl.Buffered(buffer_count=CONFIG["in_buffer_count"]),  # <-- the ONE setting that differs
    )
    out_spec = pl.BlockSpec((BLOCK_M, N), lambda i: (i, 0))

    pltpu.emit_pipeline(
        body,
        grid=(M // BLOCK_M,),
        in_specs=[in_spec],
        out_specs=out_spec,
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
    print(json.dumps({"name": CONFIG["name"], "time_ms": avg_s * 1000, "in_buffer_count": CONFIG["in_buffer_count"]}))
