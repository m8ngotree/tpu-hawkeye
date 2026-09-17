"""Taxonomy cell 04_async_pipeline -- OPTIMIZED (expert) variant.

Identical body/grid/specs to naive_kernel.py, with `no_pipelining=False` (the
default -- shown explicitly here for clarity). `emit_pipeline` automatically
double-buffers: while block i is being computed on, block i+1's HBM->VMEM copy-in is
already in flight, and block i-1's VMEM->HBM copy-out overlaps with block i's
compute. Same math, same output as naive_kernel.py -- the DMA and compute for
adjacent blocks overlap instead of running fully sequentially.
"""

import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    "name": "async_pipeline_optimized",
    "M": 512,
    "N": 128,
    "block_m": 128,
    "scale": 2.0,
    "bias": 0.5,
}

_ABSTRACT_TPU_V5E = jax.sharding.AbstractMesh(
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
        no_pipelining=False,  # <-- the ONE line that differs from naive_kernel.py
    )(x_hbm_ref, o_hbm_ref)


def workload(X):
    interpret = os.environ.get("PALLAS_INTERPRET") == "1"
    M, N = CONFIG["M"], CONFIG["N"]
    with jax.sharding.use_abstract_mesh(_ABSTRACT_TPU_V5E):
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
    print(json.dumps({"name": CONFIG["name"], "time_ms": avg_s * 1000, "no_pipelining": False}))
