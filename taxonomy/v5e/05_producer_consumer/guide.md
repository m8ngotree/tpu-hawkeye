# Producer/Consumer (buffering depth)

## What this is

Once DMA/compute overlap is enabled (`04_async_pipeline`), there's a second question:
how far can the DMA "producer" get ahead of the compute "consumer"? With standard
double buffering (2 buffers), at most one block can be in flight being fetched while
one is being consumed -- if a single block's HBM->VMEM transfer takes longer than a
single block's compute, the pipeline stalls waiting on that transfer anyway, even
though overlap is "on." Triple (or deeper) buffering gives the producer more slack:
it can fetch block i+2 while block i is still computing, absorbing a DMA that's
slower than compute without stalling the consumer.

This is Hawkeye's GPU "Producer/Consumer" row, which is about warp specialization
(dedicating specific warps to issuing loads vs. computing) -- a mechanism TPU doesn't
have (no warps). The translation kept here is the part of the idea that *does* carry
over: tuning how far the producer is allowed to run ahead of the consumer, via
buffer count rather than warp roles.

## The rule

`pl.BlockSpec(..., pipeline_mode=pl.Buffered(buffer_count=N))` on an *input* spec
controls this. Compare `naive_kernel.py` (`buffer_count=2`) against
`optimized_kernel.py` (`buffer_count=3`) -- identical body, identical grid, only the
buffer count differs. This only matters when DMA latency per block exceeds compute
time per block; if compute is the bottleneck already, deeper buffering buys nothing
(you're not DMA-bound to begin with -- check `04_async_pipeline` and
`01_mxu_feed`/`03_vectorized_vmem` first).

## How this differs from `04_async_pipeline`

Easy to conflate -- both are about the same underlying `emit_pipeline` mechanism, at
different levels:

- `04_async_pipeline` is on/off: does overlap happen at all (`no_pipelining` flag)?
- `05_producer_consumer` (this cell) is depth: *given* overlap is on, how many steps
  ahead can the producer run (`buffer_count`)?

A kernel with pipelining correctly enabled can still stall if its buffer count is too
shallow for its DMA/compute ratio -- that's what this cell catches that `04` doesn't.

## Known limitation (found by testing, not documented)

`buffer_count > 2` only works on **input** buffered refs in this jax/jaxlib version --
setting it on an output `BlockSpec` raises
`NotImplementedError: Buffer count >2 not supported for output buffered refs`. Both
files here leave `out_spec` at its default for this reason. If you hit this same
error composing a real kernel, this is why -- it's not a bug in your code.

## Status

Correctness verified locally via `interpret=True` (CPU, with the `04_async_pipeline`
abstract-mesh workaround) -- both variants match a plain elementwise reference
within JAXBench's tolerance (atol=rtol=1e-2). Like `04_async_pipeline`, there is no
exact code-level fact proving the throughput delta -- it's a genuine runtime behavior
that only manifests under a specific DMA/compute ratio, and has **not** been measured
on real v5e hardware. This cell in particular may need a deliberately
DMA-latency-heavy workload (larger blocks, smaller per-block compute) to actually
exercise the difference -- the current problem size was chosen for a fast local
interpret-mode check, not to guarantee the real-hardware gap is visible.
