# Grouped Matmul (Scalar-Prefetch Dynamic Indexing)

## Overview

Some kernels do not have a single static weight operand: which of several weight
matrices a block of rows is multiplied by is determined by a runtime index array
rather than fixed at compile time. Keeping every candidate weight matrix
VMEM-resident and selecting with an array index inside the kernel body is correct,
but VMEM cost scales with the number of candidates even though each grid step needs
only one.

## Rule

`pltpu.PrefetchScalarGridSpec(num_scalar_prefetch=N, ...)` passes small integer arrays
(a per-block group id, or any index array) as scalar-prefetch operands. They are
resident in SMEM before the pipeline starts and are readable inside each
`BlockSpec`'s `index_map`, so which block of a larger tensor is DMA'd for a given grid
step can depend on the prefetched data instead of requiring the whole tensor in
VMEM.

`naive_kernel.py` keeps the entire `(G, K, N)` weight tensor VMEM-resident for every
step and indexes it in the body. `optimized_kernel.py` passes `group_id` as a scalar
prefetch operand and has the weight `BlockSpec`'s `index_map` return
`group_id_ref[i]`, so only one `(K, N)` slice is resident per step. Each file's
`__main__` block reports `vmem_resident_weight_elems`: `G*K*N` versus `K*N`. The
naive footprint grows with `G`; the optimized footprint does not.

## Calling convention

Order of arguments with `num_scalar_prefetch=n`:

- Kernel body: `kernel(*prefetch_refs, *input_refs, *output_refs, *scratch_refs)`
- Each `index_map`: `index_map(*grid_indices, *prefetch_refs)`; every `BlockSpec`'s
  `index_map` must accept the prefetch refs, including those that do not use them.
- Call: the first `n` positional arguments to the compiled `pallas_call` are the
  scalar-prefetch operands, given before the regular inputs, and `grid_spec=` carries
  the `PrefetchScalarGridSpec`.

```python
grid_spec = pltpu.PrefetchScalarGridSpec(
    num_scalar_prefetch=1,
    grid=(num_blocks,),
    in_specs=[
        pl.BlockSpec((BLOCK_M, K), lambda i, group_id_ref: (i, 0)),
        pl.BlockSpec((1, K, N), lambda i, group_id_ref: (group_id_ref[i], 0, 0)),
    ],
    out_specs=pl.BlockSpec((BLOCK_M, N), lambda i, group_id_ref: (i, 0)),
)
pl.pallas_call(kernel, grid_spec=grid_spec, ...)(group_id, X, W)
```

## Related: skipping blocks

The same mechanism supports skipping blocks entirely: wrap the compute in
`pl.when(condition)` and multiply an `index_map` fetch index by a prefetched mask
(`k_fetch = (block_mask[i, j] != 0) * k`) so skipped blocks are not DMA'd either. This
answers "whether to fetch and compute at all", whereas the cell above answers "which
of several tensors to fetch".

## Diagnosis

Use this cell when a per-step operand is selected by an index computed earlier rather
than by a fixed offset. Correct MXU usage (`01_mxu_feed`) does not help if the weight
side carries `G` times the necessary VMEM traffic.
