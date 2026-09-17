# Lane Reduction

## What this is

Combining values across a dimension into fewer outputs (a sum, a max, a mean) is a
different hardware operation than a plain elementwise op -- it has to reduce across
lanes, not just transform each one independently -- and Mosaic has a native
cross-lane reduction path for it. A Python loop that manually accumulates one lane
at a time gets correct output but bypasses that native path entirely, same failure
mode as `03_vectorized_vmem` but specific to reductions.

## The rule

Use `jnp.sum(x, axis=...)` (or `jnp.max`, `jnp.mean`, etc.) over the whole
array/block, not a Python loop over the reduction axis. Compare `naive_kernel.py`
(`for j in range(N): acc = acc + x_ref[:, j]` -- 128 single-lane accumulation steps)
against `optimized_kernel.py` (`jnp.sum(x_ref[:, :], axis=1)` -- one call). Both
files' `__main__` blocks report `num_reduce_ops` directly (128 vs. 1).

## Running (cumulative) reductions

A related but distinct pattern: some workloads need a *running* reduction -- a
value per position along an axis that accumulates everything up to that point --
rather than one final value per row. `jnp.cumsum(x, axis=...)` is the vectorized
way to write this; the same principle as the row's main cell applies: it maps to a
native scan primitive, a manual loop computing running sums one step at a time does
not. Verified locally that `jnp.cumsum` works correctly inside a Pallas kernel under
`interpret=True` (same tolerance as the plain reduction case above) -- no
naive/optimized pair built for this specifically, since it's the same underlying
lesson with a different JAX primitive (`jnp.cumsum` instead of `jnp.sum`), not a new
technique.

## When to reach for this vs. a neighboring cell

If `eval.py` reports correct output but low throughput on a workload that reduces
along some axis (a sum, max, mean, or running/cumulative variant), check whether
that reduction is written as a loop. If the slow part is a matmul instead, that's
`01_mxu_feed`. If it's a *non-reducing* elementwise op, that's `03_vectorized_vmem`.

## Status

Correctness verified locally via `interpret=True` (CPU, no TPU) -- both variants
match a plain `jnp.sum` reference within JAXBench's tolerance (atol=rtol=1e-2), and
the `jnp.cumsum` claim above was independently verified the same way. `num_reduce_ops`
is exact from the code. The actual throughput delta on real silicon has **not** been
measured on v5e hardware yet.
