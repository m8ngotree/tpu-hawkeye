# MXU Feed

## What this is

The MXU (matrix unit -- TPU's systolic array) is where the vast majority of a TPU's
peak FLOPS lives. A Pallas kernel that never issues an MXU instruction can still be
numerically correct and still compile and run -- it just runs the matmul on the VPU
(vector unit) instead, which is dramatically slower for anything matmul-shaped. This
is the single most common way a "correct but slow" kernel happens, and it's silent:
nothing errors, nothing warns you, the kernel just runs at a fraction of achievable
throughput.

## The rule

Inside a Pallas TPU kernel body, use `jnp.dot` or `jax.lax.dot_general` for any
matrix contraction you want on the MXU. Mosaic (Pallas's TPU compiler) pattern-matches
these specific primitives and lowers them to MXU instructions. Anything
mathematically equivalent but expressed a different way -- broadcast-multiply-then-
sum, an explicit accumulation loop, `jnp.einsum` in some forms -- does NOT get this
treatment. Compare `naive_kernel.py` (broadcast + `jnp.sum`) against
`optimized_kernel.py` (`jnp.dot`): identical math, identical output, only one of them
touches the MXU.

## `preferred_element_type` matters

Inputs here are bf16. `jnp.dot(a, b, preferred_element_type=jnp.float32)` tells
Mosaic to accumulate in float32 even though the operands are bf16 -- this matches how
the MXU actually accumulates (bf16 x bf16 -> fp32 partial sums), and without it
either the accumulation happens in lower precision (accuracy loss on anything with a
long reduction dimension) or Mosaic may refuse to lower it onto the MXU at all,
depending on version. Set this explicitly rather than relying on a default.

## Tile-size gotcha (not yet exercised by this cell's small 128x128x128 case)

Pallas TPU's documented block-shape rule (see `02_vmem_tile_layout/guide.md`) is
that a block's last two dimensions must be divisible by 8 and 128 respectively --
uniform across dtypes, not dtype-dependent (an earlier version of this guide
wrongly claimed a dtype-dependent minimum; corrected after checking Pallas's
official TPU docs). Reduction (K) and output tile dimensions that violate this force
Mosaic to pad or fall back to slower paths. This cell uses exactly 128x128x128 so the
tiling question doesn't come up -- if you're adapting this pattern to a real
workload with awkward shapes, that's `03_vectorized_vmem`'s and
`02_vmem_tile_layout`'s territory, not this cell's. Separately, the official Pallas
matmul tutorial shows that even a correct MXU-feeding matmul stays far under peak if
its blocks are too small relative to the overall problem -- 128x128x128 blocks in a
4096-cubed problem measure only ~6% utilization there, vs. ~78-91% with 512x1024x1024
blocks. This cell's lesson (issue `jnp.dot`, not a manual loop) and that lesson
(size blocks large enough to be compute-bound) are both real and both needed --
neither one implies the other.

## When to reach for this vs. a neighboring cell

If `eval.py` reports correct output but low `utilization_pct` on something that's
matmul-shaped (GEMM, attention's QK^T/softmax@V, any linear layer), check this cell
first -- it's the most common root cause. If utilization is already reasonable but
you're DMA-bound (profiler shows time waiting on HBM->VMEM transfers, not compute),
that's `04_async_pipeline` instead.

## Status

Correctness verified locally via `interpret=True` (CPU, no TPU) -- both kernels
produce bitwise-identical output to a plain `jnp.dot` reference. The actual MXU-vs-VPU
throughput claim (the whole point of this cell) has **not** been verified on real v5e
hardware yet -- that needs a Kaggle TPU session running both kernels with
`interpret=False` and comparing `utilization_pct`. Do that before trusting this cell's
`expected_optimized` claim in config.json.
