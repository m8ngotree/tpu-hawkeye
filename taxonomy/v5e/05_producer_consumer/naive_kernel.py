"""Taxonomy cell 05_producer_consumer -- NAIVE variant.

Same elementwise y = x*scale+bias over an (1024, 128) bf16 array, processed in 8
row-blocks via pltpu.emit_pipeline (pipelining ON, unlike 04_async_pipeline's
naive variant which had it off). The difference here is buffering DEPTH: the input
BlockSpec uses `pipeline_mode=pl.Buffered(buffer_count=2)` -- standard double
buffering, meaning the DMA "producer" can have at most ONE block prefetched ahead of
the compute "consumer."

This is a different knob from 04_async_pipeline's no_pipelining flag (that cell was
overlap on/off; this cell is how FAR ahead the producer can run once overlap is on).
With only 2 buffers, if a single block's DMA transfer takes longer than a single
block's compute, the pipeline still stalls waiting on that transfer -- there's no
room to prefetch a second block ahead to absorb the extra latency.

Identical body/grid otherwise to optimized_kernel.py -- only the buffer_count differs.
"""

import os

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

CONFIG = {
    "name": "producer_consumer_naive",
    "M": 1024,
    "N": 128,
    "block_m": 128,
    "in_buffer_count": 2,
    "scale": 2.0,
    "bias": 0.5,
}

# See 04_async_pipeline/guide.md -- emit_pipeline queries real TPU tiling info even
# under interpret=True and fails on a non-TPU host without this.
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
    print(json.dumps({"name": CONFIG["name"], "time_ms": avg_s * 1000, "in_buffer_count": CONFIG["in_buffer_count"]}))
