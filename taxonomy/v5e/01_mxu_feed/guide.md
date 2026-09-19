# MXU Feed

## Overview

The MXU (matrix unit, a systolic array) provides the bulk of a TPU's peak FLOPS. A
Pallas kernel that never issues an MXU instruction still compiles and produces
correct results, but executes the matmul on the VPU (vector unit), which is
dramatically slower for matmul-shaped work. Nothing errors or warns; the kernel
simply runs at a fraction of achievable throughput.

## Rule

Inside a Pallas TPU kernel body, express matrix contractions with `jnp.dot` or
`jax.lax.dot_general`. Mosaic (the Pallas TPU compiler) lowers these primitives to MXU
instructions. Mathematically equivalent formulations (broadcast-multiply followed by
a sum, an explicit accumulation loop) are not recognized as matmuls and execute on
the VPU.

`naive_kernel.py` computes the product as a broadcast multiply plus `jnp.sum`;
`optimized_kernel.py` uses `jnp.dot`. The math and output are identical; only the
second uses the MXU.

## Accumulation dtype

For bf16 operands, pass `preferred_element_type=jnp.float32`. The MXU accumulates
bf16 x bf16 products in float32; requesting a float32 result keeps the accumulation
in that precision and defers any downcast to the final store. Set it explicitly
rather than relying on the default.

## Block sizing

A block's last two dimensions must be divisible by 8 and 128 respectively (see
`02_vmem_tile_layout`). Beyond legality, block size determines whether the matmul is
compute-bound: the Pallas TPU matmul tutorial reports, for bf16 on v5e at
4096x4096x4096, roughly 6% utilization with 128x128x128 blocks versus 78.5% with
512x1024x1024 blocks (XLA: 84.3%), and 91.5% at 8192x8192x8192 (XLA: 92.3%). Use the
largest blocks that fit VMEM while retaining enough grid steps to pipeline.

This cell's example is a single 128x128x128 problem and demonstrates only the
`jnp.dot` versus manual-formulation distinction, not block sizing.

## Diagnosis

Low reported utilization on matmul-shaped work with correct output points here first.
If utilization is reasonable but the kernel spends its time waiting on HBM->VMEM
transfers, see `04_async_pipeline`.
