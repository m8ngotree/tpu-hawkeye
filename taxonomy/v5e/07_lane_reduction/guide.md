# Lane Reduction

## What this is

Summing (or maxing, or otherwise reducing) across a dimension is common enough
(softmax, RMSNorm, cross-entropy, pooling) that it's worth its own cell, distinct
from `03_vectorized_vmem`'s general elementwise case: a reduction has to combine
values across lanes into fewer outputs, which is a different hardware operation
(and a different failure mode) than a plain elementwise vectorized op. Mosaic has a
native cross-lane reduction path for `jnp.sum`/`jnp.max`/etc. -- a Python loop that
manually accumulates one lane at a time gets correct output but bypasses it entirely.

## The rule

Use `jnp.sum(x, axis=...)` (or `jnp.max`, `jnp.mean`, etc.) over the whole
array/block, not a Python loop over the reduction axis. Compare `naive_kernel.py`
(`for j in range(N): acc = acc + x_ref[:, j]` -- 128 single-lane accumulation steps)
against `optimized_kernel.py` (`jnp.sum(x_ref[:, :], axis=1)` -- one call). Both
files' `__main__` blocks report `num_reduce_ops` directly (128 vs. 1).

## Prefix scans (cumulative reductions)

A related but distinct pattern: some workloads need a *running* reduction rather
than one final value per row -- e.g. RetNet/Mamba2-style linear-attention variants
build their decay mask from `jnp.cumsum(log_a, axis=-1)` (JAXBench's
`15p_RetNet_Retention`/`16p_Mamba2_SSD` baselines both do this). The same principle
applies: `jnp.cumsum` maps to a native scan primitive, a manual loop computing
running sums one step at a time does not. Verified locally that `jnp.cumsum` works
correctly inside a Pallas kernel under `interpret=True` (same tolerance as the plain
reduction case above) -- no naive/optimized pair built for this specifically, since
it's the same underlying lesson as the row's main cell with a different JAX
primitive (`jnp.cumsum` instead of `jnp.sum`), not a new technique.

## When to reach for this vs. a neighboring cell

If `eval.py` reports correct output but low throughput on a workload with a
`sum`/`max`/`mean`/`cumsum` along some axis (softmax, norm, pooling, a decay mask),
check whether that reduction is written as a loop. If the slow part is a matmul
instead, that's `01_mxu_feed`. If it's a *non-reducing* elementwise op, that's
`03_vectorized_vmem`.

## Status

Correctness verified locally via `interpret=True` (CPU, no TPU) -- both variants
match a plain `jnp.sum` reference within JAXBench's tolerance (atol=rtol=1e-2), and
the `jnp.cumsum` claim above was independently verified the same way. `num_reduce_ops`
is exact from the code. The actual throughput delta on real silicon has **not** been
measured on v5e hardware yet.
