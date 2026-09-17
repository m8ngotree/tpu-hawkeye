# Grouped Matmul (scalar-prefetch dynamic indexing)

## What this is

Some workloads don't have a single static matmul shape -- which weight matrix (or
KV-cache page, or expert) a given block of rows needs depends on data computed
earlier (a router's top-k choice, a page table). MoE layers are the clearest case:
after tokens are routed to experts and sorted by assignment, each contiguous group
of rows needs a *different* weight matrix, and which one isn't known until runtime.
Paged attention has the same shape: which KV-cache pages a query needs depends on a
page-index table, not a fixed offset.

This row didn't come from Hawkeye's original GPU taxonomy -- it was added after
reading all 50 JAXBench workloads and finding that 5 of them (`6p`/`7p` paged
attention, `10p`/`11p`/`14p` MoE-family) share this exact pattern and that none of
the other 7 rows teach it. See `taxonomy/README.md`'s "Row provenance" section for
the full reasoning.

## The rule

`pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=N, ...)` lets you pass small
integer arrays (like a per-block group-id, or a page-index table) as **scalar
prefetch** operands -- resident in SMEM before the pipeline starts, and readable from
inside a `BlockSpec`'s `index_map` function. That means which block of a larger
tensor gets DMA'd in for a given grid step can be **data-dependent**, decided by the
prefetched array, instead of requiring the whole tensor to be VMEM-resident so you
can index into it after the fact.

Compare `naive_kernel.py` (loads the entire `(G, K, N)` weight tensor into VMEM every
step, indexes into it with a plain array read inside the kernel body) against
`optimized_kernel.py` (`group_id` as a scalar-prefetch operand; the weight
`BlockSpec`'s `index_map` reads `group_id_ref[i]` to fetch only that step's expert's
`(K, N)` slice). Both files' `__main__` blocks report `vmem_resident_weight_elems`
directly -- `G*K*N` vs. `K*N`. This cell uses a small `G=4` to keep the example fast
to verify locally, but the naive approach's cost scales with `G`; the optimized
approach's doesn't.

## Calling convention (verified by testing, not copied from a doc)

- The kernel body function receives the scalar-prefetch ref(s) **first**, before the
  regular block refs: `def kernel(group_id_ref, x_ref, w_ref, o_ref): ...`
- Every `index_map` in `in_specs`/`out_specs` must accept the scalar-prefetch ref(s)
  as trailing arguments, even the ones that don't use them:
  `lambda i, group_id_ref: (i, 0)`
- The scalar-prefetch array itself is passed as the **first** positional argument to
  the compiled `pallas_call`, before the regular inputs:
  `pallas_call(kernel, grid_spec=grid_spec, ...)(group_id, X, W)`

This worked on the first attempt when tested against this project's jax/jaxlib
version, but was not cross-checked against Pallas's own official documentation or
examples -- treat it as verified-by-experiment, not verified-by-source.

## When to reach for this vs. a neighboring cell

If a workload's shape involves routing, grouping, or gathering by an index computed
earlier in the same kernel/model (MoE dispatch, paged KV-cache, any "which weight
matrix" decision that isn't known until runtime), this is the cell to consult before
`01_mxu_feed` -- getting the matmul itself onto the MXU doesn't help if you're paying
for `G`x too much VMEM traffic on the weight side first.

## Status

Correctness verified locally via `interpret=True` (CPU, no TPU) -- both variants
match a plain per-block reference matmul within JAXBench's tolerance
(atol=rtol=1e-2). `vmem_resident_weight_elems` is exact from the code/config. The
actual throughput/VMEM-pressure benefit on real silicon, and at a realistic `G`
(JAXBench's `11p_Megablox_GMM` uses G=128), has **not** been measured on v5e hardware
yet.
