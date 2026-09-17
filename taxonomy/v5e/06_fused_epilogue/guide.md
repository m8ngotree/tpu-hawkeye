# Fused Epilogue

## What this is

Splitting a computation into separate kernels is the natural way to write it
incrementally -- one kernel per logical op, matching how you'd write it in plain
JAX. But every kernel boundary is an HBM round trip: the previous kernel's output
has to be written to HBM, and the next kernel has to read it back, even if nothing
outside the two kernels ever needed that intermediate value. For a "final op" like a
bias-add + activation that immediately follows a matmul, that round trip is pure
overhead -- the data was already sitting in VMEM right after the matmul finished.

This is most of what JAXBench's ~33 "k-series" fused-operator workloads are testing
(`Gemm_Add_ReLU`, `Conv2d_GroupNorm_Tanh_HardSwish_ResidualAdd_LogSumExp`, etc.) --
whether an agent fuses the whole chain into one kernel or leaves it as separate ops.

## The rule

If op B always immediately follows op A on A's full output, with nothing else
consuming A's output in between, put both in the same kernel body instead of two
`pallas_call`s (or a `pallas_call` followed by a bare `jnp` op). Compare
`naive_kernel.py` (`_matmul_kernel` then `_epilogue_kernel`, two separate
`pallas_call`s, with `C` written to and read from HBM in between) against
`optimized_kernel.py` (one `_fused_kernel` that computes the matmul and applies the
bias+ReLU to the same VMEM-resident result before ever writing it out). Both files'
`__main__` blocks report `num_pallas_calls` directly (2 vs. 1) -- an exact,
code-level fact, not a hardware claim.

## When this doesn't apply

Fusing only pays off when the intermediate really is dead outside the fused region.
If you need the pre-activation matmul output for something else too (a residual
connection consumed elsewhere, a debugging hook, gradient computation in a training
step), materializing it isn't waste -- it's a real dependency, and this cell doesn't
apply.

## When to reach for this vs. a neighboring cell

If `eval.py` reports correct output on a workload that's visibly a chain of ops
(matmul + activation + norm, or conv + norm + residual), and the kernel file has more
than one `pallas_call` where the intermediate isn't reused elsewhere, check this cell.
If the issue is that the matmul itself is slow regardless of fusion, that's
`01_mxu_feed`'s territory first -- fuse only once the core compute is already correct.

## Status

Correctness verified locally via `interpret=True` (CPU, no TPU) -- both variants
match a plain `relu(A@B + bias)` reference within JAXBench's tolerance
(atol=rtol=1e-2). `num_pallas_calls` is exact from the code. The actual HBM-traffic /
wall-clock savings from fusion has **not** been measured on real v5e hardware yet.
