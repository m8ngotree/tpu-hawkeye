# Lane Reduction

## Overview

Combining values across a dimension into fewer outputs (sum, max, mean) requires
reducing across lanes, a different hardware operation from an elementwise transform.
Mosaic provides a native cross-lane reduction path for `jnp.sum`, `jnp.max`, and
related functions. A Python loop that accumulates one lane at a time produces the
same result but bypasses that path.

## Rule

Use `jnp.sum(x, axis=...)` (or `jnp.max`, `jnp.mean`, ...) over the whole array or
block instead of looping over the reduction axis.

`naive_kernel.py` runs `for j in range(N): acc = acc + x_ref[:, j]`, 128 single-lane
accumulation steps; `optimized_kernel.py` runs `jnp.sum(x_ref[:, :], axis=1)`. Each
file's `__main__` block reports `num_reduce_ops` (128 versus 1).

## Running (cumulative) reductions

`jnp.cumsum(x, axis=...)` computes a value per position that accumulates everything
up to that position, rather than one final value. The same principle applies: use the
vectorized primitive rather than a manual loop of running sums. `jnp.cumsum` lowers
inside a Pallas kernel like `jnp.sum` does.

## Diagnosis

Low throughput on a kernel that reduces along an axis: check whether the reduction is
written as a loop. For matmul-shaped work see `01_mxu_feed`; for non-reducing
elementwise work see `03_vectorized_vmem`.
