# Async Pipeline (HBM<->VMEM DMA)

## What this is

Every grid step of a Pallas kernel needs its input block copied HBM -> VMEM before
compute and its output block copied VMEM -> HBM after. Done naively, that's
synchronous: copy in, wait, compute, copy out, wait, move to the next block. Done
well, the copy-in for block i+1 starts while block i is still computing, and the
copy-out for block i-1 finishes in the background while block i computes -- DMA
transfer time gets hidden behind compute time instead of adding to it. This is the
direct TPU analogue of GPU's `cp.async`/TMA pipelining (Hawkeye's Async Pipeline row).

## The rule

`pltpu.emit_pipeline(body, grid=..., in_specs=..., out_specs=..., no_pipelining=...)`
manages this for you -- you write the per-block `body` once, and the same code
either runs fully synchronously (`no_pipelining=True`) or with automatic
double-buffered overlap (`no_pipelining=False`, the default). Compare
`naive_kernel.py` and `optimized_kernel.py`: identical `body`, identical grid,
identical `BlockSpec`s -- the only difference is that one flag. This is a much
smaller surface area to get right than hand-rolling `pltpu.make_async_copy` +
semaphores yourself, and is the better default unless you need finer control than
`emit_pipeline` exposes.

## The part that isn't in any Pallas tutorial we found

`emit_pipeline` queries real TPU hardware info (tiling factors, device generation)
to decide how to lay out its double buffers -- and it does this **even under
`interpret=True`**, which means it fails on a non-TPU host (a laptop, a Kaggle CPU
session) with `ValueError: Unsupported TPU device kind: cpu`, even though
`interpret=True` is supposed to mean "don't need a TPU." The fix: wrap the
`pallas_call` in `jax.sharding.use_abstract_mesh(...)` with a
`jax.sharding.AbstractMesh` whose `abstract_device` names the TPU generation you're
targeting:

```python
abstract_mesh = jax.sharding.AbstractMesh(
    (), (),
    abstract_device=jax.sharding.AbstractDevice(
        device_kind="TPU v5e", num_cores=1, platform="tpu"),
)
with jax.sharding.use_abstract_mesh(abstract_mesh):
    ...  # pallas_call goes here
```

We found this by reading the exception's own message and then
`jax/_src/tpu_info.py`'s source -- it's not documented anywhere we could find in the
Pallas docs themselves. If you're composing this cell into a workload kernel and hit
the same `ValueError`, this is almost certainly why.

## When to reach for this vs. a neighboring cell

If `eval.py` (on real TPU, not interpret mode) shows correct output but the kernel
spends a lot of time that doesn't track with FLOP count -- especially for a
memory-heavy workload (large tensors, low arithmetic intensity) -- suspect DMA
stalls and check this cell. If the workload is instead compute-bound on a matmul
that isn't hitting the MXU, that's `01_mxu_feed`. If DMA transfers are happening but
the *shape* of each block wastes VMEM, that's `02_vmem_tile_layout`.

## Status

Correctness verified locally via `interpret=True` (CPU, no TPU, using the
abstract-mesh workaround above) -- both variants match a plain elementwise reference
within JAXBench's tolerance (atol=rtol=1e-2). Unlike `02_vmem_tile_layout` and
`03_vectorized_vmem`, there is **no exact code-level fact** that proves the
performance difference here (no grid-step or op count to point at) -- the actual
DMA-overlap benefit is a real-hardware runtime behavior that has **not** been
measured on v5e yet. That needs a Kaggle TPU session with `interpret=False` and,
ideally, a real profiler trace once `agent/profiler.py` exists.
