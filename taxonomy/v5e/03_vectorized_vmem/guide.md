# Vectorized VMEM Load/Store

## Overview

The VPU (vector unit) operates on all 128 lanes of a VMEM row in a single
instruction. Code that accesses a Ref one lane, or one small slice, at a time (a
Python loop over the lane dimension, per-element indexing) produces many narrow
reads and writes instead of one wide one. The result is still correct, but per-lane
parallelism goes unused and throughput drops.

## Rule

Write whole-block vector expressions (`ref[:, :] = ...`, or slices spanning full
lane-width chunks) instead of looping over individual lanes or elements in Python.

`naive_kernel.py` issues `for j in range(N): o_ref[:, j] = ...`, 128 single-lane
statements; `optimized_kernel.py` issues `o_ref[:, :] = ...`, one statement over the
whole block. Each file's `__main__` block reports `num_vmem_ops` (128 versus 1).

## Relation to `02_vmem_tile_layout`

`02_vmem_tile_layout` concerns how the grid divides an array into blocks (BlockSpec
shape versus hardware tile size). This cell concerns how the code inside one block
reads and writes it: whole-vector versus element-by-element. The two are
independent; a well-tiled block can still be processed with a scalar loop.

## Diagnosis

Low throughput on an elementwise kernel with no matmul: check the kernel body for
Python loops over an array dimension. For matmul-shaped work see `01_mxu_feed`; for
reductions along an axis see `07_lane_reduction`.
