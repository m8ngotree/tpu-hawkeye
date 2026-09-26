# VMEM Tile Layout

## Overview

TPU vector registers and VMEM store arrays in fixed-size tiles. The Pallas TPU block
shape rule is that the last two dimensions of a block must be divisible by 8 and 128
respectively, or equal the full array dimensions; this is independent of dtype and is
enforced at lowering, so a violating block shape raises a `ValueError` instead of
running slowly.

Among legal shapes, block size still matters. Every grid step carries fixed overhead
(DMA issue and wait, loop bookkeeping), so a kernel that uses the smallest legal block
spends most of its time on that overhead. Larger blocks amortize it, up to the VMEM
capacity available for the double-buffered input and output blocks.

## Rule

Use block shapes that are multiples of (8, 128) in the last two dimensions, and make
them as large as VMEM allows.

`naive_kernel.py` uses an `(8, 128)` block, the smallest legal shape;
`optimized_kernel.py` uses `(128, 1024)`. Math and output are identical. For the same
1024x1024 array the grid has 1024 steps versus 8. Each file's `__main__` block reports
`num_grid_steps`.

## Diagnosis

A grid step count much larger than the problem size warrants
(`M // block_m * N // block_n`) indicates undersized blocks; it shows up as
`pct_of_roofline_limit` well below 100 regardless of whether `workload_limit` is
compute- or memory-bound. If the step count is
reasonable and the kernel is a compute-bound matmul, see `01_mxu_feed`. If it is bound
on HBM<->VMEM transfer time rather than per-step overhead, see `04_async_pipeline`.
