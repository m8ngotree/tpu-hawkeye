# VMEM Tile Layout

## What this is

TPU VMEM (and the vector registers that feed the VPU/MXU) physically store arrays in
fixed-size tiles, not as a flat buffer. The tile shape is dtype-dependent: **(8, 128)**
for fp32 (8 sublanes x 128 lanes), **(16, 128)** for bf16 (16 sublanes -- two bf16
elements pack into each physical fp32 sublane -- x 128 lanes, lane width is fixed at
128 regardless of dtype). When a Pallas `BlockSpec`'s block shape isn't a multiple of
this tile in its last two dimensions, Mosaic still makes it work -- it pads each block
up to the next tile boundary -- but that padding is wasted VMEM capacity and wasted
DMA/compute cycles on lanes that don't hold real data. It's also invisible unless you
know to look for it: the kernel still runs, still produces correct output, just slower
and with more grid steps than the logical problem size would suggest.

## The rule

Choose block shapes whose last two dimensions are whole-number multiples of the
dtype's native tile -- (8,128)/(16,128)/etc. -- not just "small enough to fit VMEM."
Compare `naive_kernel.py` (block `(4, 64)` -- neither dimension is a clean multiple of
bf16's `(16, 128)` tile) against `optimized_kernel.py` (block `(32, 128)` -- 2x the
sublane dimension, exactly 1x the lane dimension). Identical math, identical output;
the only difference is block shape, and it produces a 16x difference in grid steps
(256 vs. 16) for the same total array size -- see each file's `__main__` block, which
reports `num_grid_steps` directly, no profiler needed to observe this part.

## Why this is dtype-dependent

The 128-lane width is fixed, but the sublane count that packs into one physical row
scales with how many elements of that dtype fit in a 32-bit physical sublane: 1 for
fp32 (tile = 8x128), 2 for bf16 (tile = 16x128), 4 for int8/fp8 (tile = 32x128). A
block shape tuned for fp32 will silently be sub-optimal if you switch the same kernel
to bf16 without revisiting the tile shape -- this is a common way a working kernel
gets slower after a "harmless" dtype change.

## When to reach for this vs. a neighboring cell

If `eval.py` reports correct output and the kernel is running many more grid steps
than the problem size seems to justify (compare against `M // block_m * N // block_n`
by hand), check block shape alignment here first. If grid-step count looks reasonable
but you're compute-bound on a matmul specifically, that's `01_mxu_feed`'s territory.
If you're bound on HBM<->VMEM transfer time rather than tile padding, that's
`04_async_pipeline`.

## Status

Correctness verified locally via `interpret=True` (CPU, no TPU) -- both kernels match
a plain elementwise reference within JAXBench's own tolerance (atol=rtol=1e-2). The
grid-step counts (256 vs. 16) are exact and verifiable from the code alone, not a
hardware claim. The actual wall-clock/throughput delta this tiling choice produces on
real silicon has **not** been measured on v5e hardware yet -- that needs a Kaggle TPU
session comparing both kernels with `interpret=False`.
