# Producer/Consumer (Buffering Depth)

## Overview

With DMA/compute overlap enabled (`04_async_pipeline`), the remaining parameter is
how far the DMA "producer" may run ahead of the compute "consumer". With double
buffering (2 buffers) at most one block is in flight while another is consumed; if
one block's HBM->VMEM transfer takes longer than one block's compute, the pipeline
stalls even though overlap is on. Triple or deeper buffering lets the producer fetch
block i+2 while block i computes, absorbing transfers slower than compute.

TPU has no warp specialization, so producer/consumer depth is controlled through
buffer count rather than dedicated warp roles.

## Rule

Set `pipeline_mode=pl.Buffered(buffer_count=N)` on an input `BlockSpec`. The default
`buffer_count` is 2 for all inputs and outputs.

`naive_kernel.py` uses `buffer_count=2`; `optimized_kernel.py` uses `buffer_count=3`.
Body and grid are identical. Deeper buffering only helps when per-block DMA latency
exceeds per-block compute time; if compute is the bottleneck it has no effect.

`pl.Buffered(buffer_count=N, use_lookahead=True)` additionally allows the pipeline to
begin fetching a future block as soon as a buffer slot frees, regardless of how many
iterations ahead that block is, instead of prefetching only one step ahead.

## Constraints

- `buffer_count > 2` is supported on input buffered refs only; on an output
  `BlockSpec` it raises
  `NotImplementedError: Buffer count >2 not supported for output buffered refs`.
  Both variants leave the output spec at its default.
- `emit_pipeline` requires the abstract-mesh wrapper described in
  `04_async_pipeline` when run on a host without a TPU.

## Relation to `04_async_pipeline`

`04_async_pipeline` toggles overlap on or off (`no_pipelining`). This cell tunes
prefetch depth once overlap is on (`buffer_count`).
