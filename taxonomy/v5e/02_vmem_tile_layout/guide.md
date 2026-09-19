# VMEM Tile Layout

## Overview

TPU vector registers and VMEM store arrays in fixed-size tiles. The Pallas TPU block
shape rule is that the last two dimensions of a block must be divisible by 8 and 128
respectively (or equal the full array dimensions), independent of dtype. A block that
does not satisfy this is padded up to the next tile boundary by Mosaic: the kernel
still runs and produces correct output, but padded lanes consume VMEM capacity and
DMA/compute cycles without holding data, and undersized blocks multiply the number
of grid steps.

## Rule

Choose block shapes whose last two dimensions are whole multiples of (8, 128).

`naive_kernel.py` uses a `(4, 64)` block (4 is not a multiple of 8; 64 is not a
multiple of 128); `optimized_kernel.py` uses `(32, 128)`. Math and output are
identical. For the same 256x256 array the grid has 256 steps versus 16. Each file's
`__main__` block reports `num_grid_steps`.

## Diagnosis

A grid step count much larger than the problem size warrants
(`M // block_m * N // block_n`) indicates poorly chosen block shapes. If the step
count is reasonable and the kernel is a compute-bound matmul, see `01_mxu_feed`. If
it is bound on HBM<->VMEM transfer time rather than tile padding, see
`04_async_pipeline`.
