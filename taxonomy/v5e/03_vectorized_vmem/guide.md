# Vectorized VMEM Load/Store

## What this is

The VPU (vector unit) operates on all 128 lanes of a VMEM row in one instruction.
Code that accesses a Ref one lane (or one small slice) at a time -- a Python loop
over the lane dimension, dynamic per-element indexing, anything that produces N
separate small reads/writes instead of one wide one -- still produces correct output,
but at a fraction of the achievable throughput, because the VPU's per-lane
parallelism goes unused. Like `01_mxu_feed`, this is a silent failure mode: nothing
errors, the kernel just runs far slower than the array size suggests it should.

## The rule

Write whole-block vector expressions (`ref[:, :] = ...`, or slices that span full
lane-width chunks) instead of looping over individual lanes/elements in Python.
Compare `naive_kernel.py` (`for j in range(N): o_ref[:, j] = ...` -- 128 single-lane
statements) against `optimized_kernel.py` (`o_ref[:, :] = ...` -- one statement over
the whole 128-lane block). Both files' `__main__` blocks report `num_vmem_ops`
directly (128 vs. 1) -- this is a fact about the code, verifiable without a profiler
or real hardware, unlike this cell's actual throughput delta (see Status).

## How this differs from `02_vmem_tile_layout`

Easy to conflate these two -- they're both "VMEM access patterns," but at different
granularities:

- `02_vmem_tile_layout` is about how the **grid divides an array into blocks**
  (BlockSpec shape vs. the hardware's tile size).
- `03_vectorized_vmem` (this cell) is about how **one block's own code** reads/writes
  it -- whole-vector vs. element-by-element, independent of how that block was sized.

A kernel can get one of these right and the other wrong: a perfectly-tiled block can
still be processed with a scalar Python loop inside the kernel body, and vice versa.

## When to reach for this vs. a neighboring cell

If `eval.py` reports correct output but throughput is low on something elementwise
(no matmul involved), check whether the kernel body has any Python loop over an array
dimension -- that's this cell's territory. If the slow part is a matmul specifically,
that's `01_mxu_feed`. If it's a full-array reduction (sum/max/mean along an axis),
that's `07_lane_reduction` instead -- reductions have their own vectorization
concerns beyond plain elementwise ops.

## Status

Correctness verified locally via `interpret=True` (CPU, no TPU) -- both kernels match
a plain elementwise reference within JAXBench's tolerance (atol=rtol=1e-2). The
`num_vmem_ops` counts (128 vs. 1) are exact, code-level facts, not a hardware claim.
The actual throughput delta this produces on real silicon has **not** been measured
on v5e hardware yet -- needs a Kaggle TPU session with `interpret=False`.
