# Async Pipeline (HBM<->VMEM DMA)

## Overview

Each grid step requires its input block to be copied HBM -> VMEM before compute and
its output block copied VMEM -> HBM afterwards. Executed synchronously, the steps are
copy-in, wait, compute, copy-out, wait, per block. With pipelining, the copy-in for
block i+1 is issued while block i computes, and the copy-out of block i-1 completes
in the background, hiding DMA time behind compute time. This is the TPU analogue of
asynchronous copy pipelining on GPUs (`cp.async`, TMA).

## Rule

`pltpu.emit_pipeline(body, grid=..., in_specs=..., out_specs=..., no_pipelining=...)`
manages the overlap. The per-block `body` is written once; `no_pipelining=True` runs
it fully synchronously, and `no_pipelining=False` (the default) runs it with
automatic double-buffered overlap.

`naive_kernel.py` and `optimized_kernel.py` share the same body, grid and
`BlockSpec`s; only the `no_pipelining` flag differs. `emit_pipeline` is generally
preferable to hand-written `pltpu.make_async_copy` plus semaphores unless finer
control is required.

## Buffer depth

`pl.BlockSpec(..., pipeline_mode=pl.Buffered(buffer_count=N))` sets the number of
buffers for an operand (default 2). For streaming elementwise and copy kernels on v5e,
buffer counts of 2, 3 and 4 produced the same device time; block size
(`02_vmem_tile_layout`) had the dominant effect. `buffer_count > 2` is supported on
inputs only, not outputs.

## Running without a TPU device

`emit_pipeline` queries TPU device information (tiling factors, generation) even
when `interpret=True`. On a host without a TPU it raises
`ValueError: Unsupported TPU device kind: cpu`. Wrap the `pallas_call` in an abstract
mesh that names the target generation:

```python
abstract_mesh = jax.sharding.AbstractMesh(
    (), (),
    abstract_device=jax.sharding.AbstractDevice(
        device_kind="TPU v5e", num_cores=1, platform="tpu"),
)
with jax.sharding.use_abstract_mesh(abstract_mesh):
    ...  # pallas_call
```

## Diagnosis

`workload_limit` reported as memory-bound with `hbm_bandwidth_pct_of_peak` well below
100 (time that does not scale with FLOP count, on large tensors with low arithmetic
intensity) suggests DMA stalls.
If the kernel is a matmul not reaching the MXU, see `01_mxu_feed`. If DMA is
overlapped but block shapes are undersized, see `02_vmem_tile_layout`.
