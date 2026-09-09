# Taxonomy schema

Rows = **optimization pattern**. Columns = **kernel category** (v5e is the only
hardware column for now; a hardware generation becomes a *second* axis once we add
v5p/v6e -- see below).

## Rows (patterns) to fill in, roughly in build order

1. `memory_pipelining` -- HBM<->VMEM double/multi-buffered DMA (`pltpu.emit_pipeline`,
   `make_async_copy`), hiding memory latency behind compute.
2. `mxu_tiling` -- tile shapes as multiples of the dtype-dependent minimum
   (8x128 fp32, 16x128 bf16, ...), avoiding relayouts that stall the MXU.
3. `vpu_vectorization` -- keeping elementwise/reduction ops on the lane dimension,
   avoiding scalar-core fallback.
4. `grid_blockspec_design` -- `BlockSpec`/`index_map` choices, revisiting, prefetch
   depth vs. VMEM budget.
5. `precision_casting` -- fp32 vs bf16 vs int8, where TPU actually benefits vs. where
   it just adds cast overhead.
6. `ici_collectives` -- all-reduce/all-gather across the 8 chips on a v5e-8 slice,
   relevant for sharded MoE/attention kernels.

## Columns (kernel categories)

Match JAXBench's own split so cells map directly onto eval workloads:
`attention`, `moe_routing`, `matmul_fusion`, `normalization`.

## Cell contents (`patterns/<pattern>/<category>/`)

- `naive.py` -- correct but unoptimized Pallas kernel for this pattern x category
- `expert.py` -- hand-optimized version. Prefer copying from JAXBench's 8 hand-optimized
  priority-kernel Pallas variants where the category overlaps, rather than writing from
  scratch.
- `metrics.yaml` -- the profiler signature that should change between naive and expert:
  ```yaml
  bottleneck_naive: dma_bound       # or mxu_bound, vmem_spill, scalar_fallback
  bottleneck_expert: mxu_bound
  mxu_utilization: {naive: 0.15, expert: 0.75}
  roofline_regime: memory_bound      # arithmetic intensity vs v5e's 197 TFLOPS / 819 GB/s ridge point
  ```
- `protocol.md` -- what the pattern is, why it helps, when an agent should reach for it
  (the profiler symptom that should trigger retrieval of this cell).

## Adding a generation column later

When v5p/v6e come online, don't duplicate the pattern rows. Add
`patterns/<pattern>/<category>/<generation>/` only where the expert kernel or metrics
actually differ by generation (e.g. v6e's higher bandwidth changes the roofline ridge
point, tile minimums may differ) -- the `protocol.md` stays generation-agnostic unless
the *pattern itself* doesn't apply on the new chip.
