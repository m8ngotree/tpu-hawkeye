# Grouped Matmul (scalar-prefetch dynamic indexing)

## What this is

Some workloads don't have a single static matmul shape: which of several weight
matrices a given block of rows should be multiplied by is decided by data computed
earlier in the pipeline (an index or assignment array), not fixed at compile time.
The naive way to handle this is to keep every candidate weight matrix VMEM-resident
and select the right one with a plain array index inside the kernel body -- correct,
but VMEM cost scales with how many candidates there are, even though any single grid
step only ever needs one.

This row didn't come from Hawkeye's original GPU taxonomy -- it was added after
finding that this specific pattern (fetch depends on a runtime-computed index, not a
static offset) recurs and that none of the other 7 rows teach it.

## The rule

`pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=N, ...)` lets you pass small
integer arrays (a per-block group-id, any small index array) as **scalar prefetch**
operands -- resident in SMEM before the pipeline starts, and readable from inside a
`BlockSpec`'s `index_map` function. That means which block of a larger tensor gets
DMA'd in for a given grid step can be **data-dependent**, decided by the prefetched
array, instead of requiring the whole tensor to be VMEM-resident so you can index
into it after the fact.

Compare `naive_kernel.py` (loads the entire `(G, K, N)` weight tensor into VMEM every
step, indexes into it with a plain array read inside the kernel body) against
`optimized_kernel.py` (`group_id` as a scalar-prefetch operand; the weight
`BlockSpec`'s `index_map` reads `group_id_ref[i]` to fetch only that step's `(K, N)`
slice). Both files' `__main__` blocks report `vmem_resident_weight_elems` directly --
`G*K*N` vs. `K*N`. This cell uses a small `G=4` to keep the example fast to verify
locally, but the naive approach's cost scales with `G`; the optimized approach's
doesn't.

## Calling convention (verified by testing, then confirmed against official docs)

- The kernel body function receives the scalar-prefetch ref(s) **first**, before the
  regular block refs: `def kernel(group_id_ref, x_ref, w_ref, o_ref): ...`
- Every `index_map` in `in_specs`/`out_specs` must accept the scalar-prefetch ref(s)
  as trailing arguments, even the ones that don't use them:
  `lambda i, group_id_ref: (i, 0)`
- The scalar-prefetch array itself is passed as the **first** positional argument to
  the compiled `pallas_call`, before the regular inputs:
  `pallas_call(kernel, grid_spec=grid_spec, ...)(group_id, X, W)`

This worked on the first attempt when tested against this project's jax/jaxlib
version, and matches the official
["Scalar Prefetch and Block-Sparse Computation" guide](https://docs.jax.dev/en/latest/pallas/tpu/sparse.html)
exactly: *"the user-defined kernel expects prefetch Refs to come before the input
Refs... additionally, the scratch refs come after the output Refs"* and *"each
BlockSpec's index_map now expects the prefetch Refs to come after the grid
indices."* Full order: `kernel(*prefetch_refs, *input_refs, *output_refs,
*scratch_refs)`.

## Related pattern from the same official guide, not built here

The same guide also documents skipping computation entirely for blocks a
scalar-prefetch mask says are irrelevant: wrap the compute in `pl.when(condition)`,
and multiply an index_map's fetch index by the mask so a skipped block doesn't even
get DMA'd (`k_fetch = (block_mask[i, j] != 0) * k`). This is a real, related
technique -- for cases with structural sparsity in which blocks matter at all -- but
distinct enough from this cell's "which of several tensors to fetch" question (this
is "whether to fetch/compute at all") that it's noted here rather than folded in or
given its own row.

## When to reach for this vs. a neighboring cell

If a workload's shape involves selecting which of several weight tensors (or other
data blocks) to use per grid step, based on an index computed earlier rather than a
fixed offset, this is the cell to consult before `01_mxu_feed` -- getting the matmul
itself onto the MXU doesn't help if you're paying for `G`x too much VMEM traffic on
the weight side first.

## Status

Correctness verified locally via `interpret=True` (CPU, no TPU) -- both variants
match a plain per-block reference matmul within JAXBench's tolerance
(atol=rtol=1e-2). `vmem_resident_weight_elems` is exact from the code/config. The
actual throughput/VMEM-pressure benefit on real silicon, and at a much larger group
count than this cell's illustrative `G=4`, has **not** been measured on v5e hardware
yet.

## Note on eval-set separation

This cell (like every taxonomy cell) is deliberately generic and doesn't name any
specific benchmark task, model, or architecture -- everything in `taxonomy/v5e/*/`
gets copied verbatim into the workspace of the agent being *evaluated*, so naming
which exact tasks need this technique, or which model architectures typically
exhibit it, would leak hints that could bias what the agent looks for or reveal
information specific to the eval set. This cell describes only the hardware/Pallas
mechanism (data-dependent block selection via scalar prefetch) -- see
`taxonomy/README.md`'s "Cell contents" section for the general rule this follows.
