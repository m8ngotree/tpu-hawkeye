# Fused Epilogue

## Overview

Every kernel boundary is an HBM round trip: a kernel's output is written to HBM and
the next kernel reads it back, even if nothing else consumes that intermediate. For an
op that immediately follows a matmul (bias add, activation), the intermediate was
already resident in VMEM when the matmul finished, so the round trip is pure
overhead. Chains of a matmul or convolution followed by elementwise or reduction ops
benefit most from running as a single kernel.

## Rule

If op B consumes only op A's full output, place both in the same kernel body rather
than in two `pallas_call`s (or a `pallas_call` followed by a bare `jnp` op).

`naive_kernel.py` runs `_matmul_kernel` and `_epilogue_kernel` as two
`pallas_call`s, so the matmul result is written to and read from HBM between them.
`optimized_kernel.py` computes the matmul and applies bias and ReLU in one
`_fused_kernel`, before anything is written out. Each file's `__main__` block reports
`num_pallas_calls` (2 versus 1).

## When fusion does not apply

Fusion only helps when the intermediate is dead outside the fused region. If the
unfused value is needed elsewhere (a residual connection consumed later, an auxiliary
output), it must be materialized and fusing it away is not possible.

## Diagnosis

A kernel file with several `pallas_call`s (or `pallas_call`s interleaved with `jnp`
ops) whose intermediates are not reused is a candidate. Intermediates written to and
read back from HBM are not counted in `min_hbm_traffic_mb`, so they appear as
`pct_of_roofline_limit` well below 100. If the matmul itself is slow
independent of fusion, address `01_mxu_feed` first.
