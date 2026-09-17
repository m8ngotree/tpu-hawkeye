# VMEM Tile Layout

**Correction (read against official docs after initially writing this cell):** an
earlier version of this guide claimed the minimum block-shape tile was
dtype-dependent -- (8,128) for fp32, (16,128) for bf16, with bf16 packing two values
per physical sublane. That was wrong. [Pallas's own TPU docs](https://docs.jax.dev/en/latest/pallas/tpu/details.html)
state the rule plainly: *"the last two dimensions of your block shape must be
divisible by 8 and 128 respectively, or be equal to the respective dimensions of the
overall array"* -- **(8, 128), uniformly, regardless of dtype.** No dtype-specific
packing rule for block shapes is documented anywhere we could find. The cell's
underlying example still demonstrates a real thing correctly (see below) -- only the
explanation of *why* needed fixing.

## What this is

TPU VMEM (and the vector registers that feed the VPU/MXU) physically store arrays in
fixed-size tiles, not as a flat buffer. Per the official docs, the minimum tile for a
block's last two dimensions is **(8, 128)** -- 8 sublanes x 128 lanes -- and this
applies uniformly across dtypes at the `BlockSpec` level. When a block shape isn't a
multiple of this tile, Mosaic still makes it work -- it pads each block up to the
next tile boundary -- but that padding is wasted VMEM capacity and wasted DMA/compute
cycles on lanes that don't hold real data. It's also invisible unless you know to
look for it: the kernel still runs, still produces correct output, just slower and
with more grid steps than the logical problem size would suggest.

## The rule

Choose block shapes whose last two dimensions are whole-number multiples of **(8,
128)** -- not just "small enough to fit VMEM." Compare `naive_kernel.py` (block `(4,
64)` -- 4 isn't a multiple of 8, 64 isn't a multiple of 128) against
`optimized_kernel.py` (block `(32, 128)` -- 4x and 1x respectively). Identical math,
identical output; the only difference is block shape, and it produces a 16x
difference in grid steps (256 vs. 16) for the same total array size -- see each
file's `__main__` block, which reports `num_grid_steps` directly, no profiler needed
to observe this part.

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
hardware claim. The (8,128) rule itself is now confirmed against Pallas's official
TPU docs (see correction note above) -- the actual wall-clock/throughput delta this
tiling choice produces on real silicon still has **not** been measured on v5e
hardware yet -- that needs a Kaggle TPU session comparing both kernels with
`interpret=False`.
