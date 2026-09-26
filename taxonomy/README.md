# Taxonomy schema

Follows Hawkeye's Table 3 / Appendix C.2 shape (rows = recurring optimization
strategies, columns = hardware generations), but not its row count -- **10 wasn't a
target to hit, it's just what fell out of the GPU study.** We trimmed to 7 rows that
matter for the core (single-chip) JAXBench workloads, then added one back after
actually reading all 50 JAXBench workloads and finding a real gap (`08_grouped_matmul`
-- see "Row provenance" below for the full trace of what got cut, added, and why.
**We only have one column right now (`v5e/`)** -- a new generation later (`v5p/`,
`v6e/`, ...) means adding a sibling directory with the same row names, not
redesigning anything.

Each cell is a **generic, workload-agnostic** illustration of one technique -- not a
JAXBench kernel. It exists so an agent that has never seen this TPU generation can
learn the syntax and the profiler signature for one optimization in isolation, then
compose several cells together itself when it goes to optimize a real workload.

## The 8 rows, translated from Hawkeye's GPU taxonomy to Pallas/Mosaic/TPU

Directory names are final -- use these exactly when writing cells, they're what the
agent will `ls`/`cat` to browse the taxonomy (see "How the agent finds a cell" below).

| # | Directory (`v5e/...`) | Hawkeye (GPU) row | What it demonstrates |
|---|---|---|---|
| 1 | `01_mxu_feed` | MMA Unit | A matmul that actually lowers to the systolic array (MXU) vs. falling back to VPU emulation -- operand dtype/shape requirements |
| 2 | `02_vmem_tile_layout` | Shared Memory Layout | Block shapes: last two dims must be divisible by (8, 128) (violations are rejected at lowering); among legal shapes, larger blocks amortize per-step overhead |
| 3 | `03_vectorized_vmem` | Vectorized Memory | Lane-aligned loads so the VPU doesn't fall back to scalar-core ops |
| 4 | `04_async_pipeline` | Async Pipeline | `pltpu.emit_pipeline` / `make_async_copy`, multi-stage buffering -- direct analogue of TMA/cp.async |
| 5 | `05_producer_consumer` | Producer/Consumer | Multi-stage buffering depth (2-stage vs. 3+-stage prefetch) -- how far ahead the DMA "producer" can run of the compute "consumer," distinct from `04_async_pipeline`'s on/off overlap toggle |
| 6 | `06_fused_epilogue` | Epilogue Pipeline | Fusing bias/activation/norm into the same kernel instead of separate ops (fewer HBM round trips) |
| 7 | `07_lane_reduction` | Warp/Wave Reduction | Reductions that map to native cross-lane ops instead of naive loops, including cumulative reductions (`jnp.cumsum`) |
| 8 | `08_grouped_matmul` | *(no direct Hawkeye row -- see provenance below)* | Per-step operand selected by a runtime index array via scalar prefetch (`PrefetchScalarGridSpec`), instead of keeping every candidate VMEM-resident |

## Row provenance

Cut two Hawkeye rows, folded one in, and added one that Hawkeye doesn't have at all
(found by actually reading all 50 JAXBench workloads, not guessed in advance):

- **Quantized Precision** (Hawkeye's int8/fp8 cascade row) -- dropped for now. Most
  JAXBench workloads run bf16; quantization is a real technique but a secondary
  concern for a first taxonomy-vs-no-taxonomy result. Add back as `09_precision_cascade`
  if/when low-precision workloads become a focus.
- **Multi-Unit Coordination** (ICI collectives across the 8 v5e-8 chips) -- dropped.
  Only matters for sharded multi-chip workloads, which is a small slice of JAXBench's
  50 (mostly single-op, single-chip) tasks. Add back as `09_ici_collective` if a
  sharded workload actually needs it.
- **Persistent Scheduling** -- folded into `05_producer_consumer` rather than kept
  separate. On v5e (no megacore split, unlike v4/v5p), persistent-grid scheduling and
  producer/consumer overlap are close enough in practice that a separate cell would
  mostly repeat the same DMA-prefetch content.
- **`08_grouped_matmul` -- added.** Read all 50 JAXBench workloads (not just the
  17 "priority" ones) to check whether the 7-row set actually covers them. It mostly
  does -- standard/GQA/MLA/Flex/Sparse attention, GEMM, SwiGLU, RMSNorm, Triangle
  Multiplication, and the ~33 fused elementwise/GEMM/conv chains all map cleanly onto
  MXU Feed + Fused Epilogue + Lane Reduction + the pipelining rows. But 5 workloads --
  `6p_Paged_Attention`, `7p_Ragged_Paged_Attention`, `10p_Sparse_MoE`,
  `11p_Megablox_GMM`, `14p_Ragged_Dot` -- share a pattern none of the 7 rows teach:
  matmuls (or gathers feeding a matmul) whose group/segment boundaries are
  data-dependent rather than static. `11p_Megablox_GMM`'s baseline literally uses
  `jax.lax.scan` + `dynamic_slice`/`dynamic_update_slice` per expert group; the paged
  attention workloads gather KV pages by an index table and scatter-add the output.
  This is a different technique from tiling a dense matmul, not a variant of one, so
  it earns its own row rather than a note inside `01_mxu_feed`.
- **Convolutions -- explicitly NOT added, despite being real.** 8 of 50 workloads
  (16%) use `lax.conv_general_dilated` fused with norm/activation. A hand-Pallas conv
  kernel needs im2col-style patch extraction before it can feed the MXU, which is a
  distinct technique from `01_mxu_feed`'s plain-matmul case. Deferred because XLA's
  native conv lowering on TPU is already reasonable -- the agent can legitimately
  leave convolutions to XLA rather than hand-write Pallas convs, so this gap doesn't
  block a correct/adequate result the way grouped_matmul's did. Revisit if
  convolution-heavy workloads become a specific focus.

This list isn't sacred either -- if writing/testing these 8 surfaces a bottleneck the
agent can identify but has no matching cell for, add a row then, grounded in a real
gap instead of a guess.

## How the agent finds a cell (no separate retriever)

There's deliberately no embedding search or hardcoded lookup table mapping bottleneck
-> cell. The agent's workspace just has a `taxonomy/` directory with these names
in it (see [agent/workspace.py](../agent/workspace.py)), and `taxonomy/` itself is
only *mentioned*, not pasted, in the initial task prompt (see
[agent/README.md](../agent/README.md)) -- the agent has to actively `ls`/`cat` its
way in via its own tools when it wants to. Two things make that tractable at only a
handful of cells: the directory names are self-describing, and every cell's `config.json` names
the exact profiler counter it targets, which the agent can match against whatever
`eval.py` just reported back to it. This only scales because the taxonomy stays
minimal -- it would fall apart as a retrieval strategy against a large raw doc corpus,
which is exactly the point Hawkeye makes about curated-minimal beating
comprehensive-raw (see Section 2.2 of the paper).

## Cell contents (`v5e/<NN>_<row_name>/`)

Matching Hawkeye's four-artifact shape (Appendix C.2) exactly, just renamed for
Python/Pallas instead of CUDA:

- `naive_kernel.py` -- deliberately unoptimized `workload(*inputs)`, so the profiler
  counter this cell targets reads near zero. Establishes the floor the technique has
  to beat.
- `optimized_kernel.py` -- the one hand-written expert example. Callable directly, or
  meant to be read as a syntax reference and composed into a larger kernel.
- `config.json` -- machine-readable: which profiler counter proves this technique
  fired, and which direction is "better." e.g.:
  ```json
  {
    "cell_name": "async_pipeline",
    "task_description": "Double-buffered HBM->VMEM DMA on v5e",
    "profiling": {
      "metrics": [{
        "name": "dma_wait_fraction",
        "description": "fraction of step time spent waiting on HBM->VMEM DMA",
        "direction": "lower_is_better"
      }]
    }
  }
  ```
- `guide.md` -- prose protocol for anything the code + counter don't convey by
  themselves (a buffering/phase protocol, a gotcha, when to reach for this vs. a
  neighboring row).

**Content rules for all four files.** Everything under `taxonomy/v5e/<cell>/` is
copied verbatim into the workspace of the agent being evaluated (see
`agent/workspace.py`), so cell content must:

- describe hardware and Pallas/Mosaic mechanics only (what the hardware does, what the
  compiler recognizes, what the API requires), with generic examples built from
  primitive operations such as elementwise ops, matmuls and reductions;
- not name any benchmark task, workload, model, or neural-network architecture, and
  not include project-internal commentary (revision history, corrections,
  verification status);
- read as formal reference documentation.

Workload-specific reasoning (why a row exists, which tasks motivated it) belongs in
this file, which sits outside `taxonomy/v5e/` and is never copied into an agent
workspace (see "Row provenance" above).

## Why cells are separate from JAXBench workload kernels

The agent being evaluated (Section 2.4, Fig. 3) never edits taxonomy cells -- it reads
them as reference material, then writes and iterates on its own kernel file for
whatever JAXBench workload it's assigned, using `evaluate_kernel` (compile +
correctness + benchmark, see [agent/runner.py](../agent/runner.py)) to check its work
and profiler counters to decide what to try next. `kernel_pool/` (not yet built)
accumulates the agent's own correct kernels from earlier workloads on the same
architecture, so later workloads can reuse compositions it already found -- this is
the cross-workload reuse Hawkeye describes in Appendix E.3.1.

## Status

All 8 cells are written; each has been checked for numerical correctness against a
reference under `interpret=True` on CPU. Throughput claims have not yet been measured
on v5e hardware.
