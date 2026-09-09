# Taxonomy schema

Follows Hawkeye's Table 3 / Appendix C.2 exactly: rows are recurring optimization
strategies, columns are hardware generations. **We only have one column right now
(`v5e/`)** -- a new generation later (`v5p/`, `v6e/`, ...) means adding a sibling
directory with the same 10 row names, not redesigning anything.

Each cell is a **generic, workload-agnostic** illustration of one technique -- not a
JAXBench kernel. It exists so an agent that has never seen this TPU generation can
learn the syntax and the profiler signature for one optimization in isolation, then
compose several cells together itself when it goes to optimize a real workload.

## The 10 rows, translated from Hawkeye's GPU taxonomy to Pallas/Mosaic/TPU

Directory names are final -- use these exactly when writing cells, they're what the
agent will `ls`/`cat` to browse the taxonomy (see "How the agent finds a cell" below).

| # | Directory (`v5e/...`) | Hawkeye (GPU) row | What it demonstrates |
|---|---|---|---|
| 1 | `01_mxu_feed` | MMA Unit | A matmul that actually lowers to the systolic array (MXU) vs. falling back to VPU emulation -- operand dtype/shape requirements |
| 2 | `02_precision_cascade` | Quantized Precision | bf16 -> int8 quantized matmul path (v5e's MXU supports int8, not fp8/fp4 like newer GPUs -- that's a real hardware difference from Hawkeye's GPUs) |
| 3 | `03_vmem_tile_layout` | Shared Memory Layout | Block/tile shapes that avoid relayouts (multiples of the dtype-dependent minimum, e.g. 8x128 fp32 / 16x128 bf16) |
| 4 | `04_vectorized_vmem` | Vectorized Memory | Lane-aligned loads so the VPU doesn't fall back to scalar-core ops |
| 5 | `05_async_pipeline` | Async Pipeline | `pltpu.emit_pipeline` / `make_async_copy`, multi-stage buffering -- direct analogue of TMA/cp.async |
| 6 | `06_producer_consumer` | Producer/Consumer | Prefetching the next grid step's block while computing the current one |
| 7 | `07_fused_epilogue` | Epilogue Pipeline | Fusing bias/activation/norm into the same kernel instead of separate ops (fewer HBM round trips) |
| 8 | `08_lane_reduction` | Warp/Wave Reduction | Reductions that map to native cross-lane ops instead of naive loops |
| 9 | `09_ici_collective` | Multi-Unit Coordination | Cross-chip all-reduce/all-gather on the 8-chip v5e-8 slice (sharded MoE/attention) |
| 10 | `10_persistent_scheduling` | Persistent Scheduling | Minimizing relaunch/pipeline-drain overhead across grid steps |

## How the agent finds a cell (no separate retriever)

There's deliberately no embedding search or hardcoded lookup table mapping bottleneck
-> cell. The agent's workspace just has a `taxonomy/` directory with these 10 names
in it (see [agent/workspace.py](../agent/workspace.py)), and `taxonomy/` itself is
only *mentioned*, not pasted, in the initial task prompt (see
[agent/README.md](../agent/README.md)) -- the agent has to actively `ls`/`cat` its
way in via its own tools when it wants to. Two things make that tractable at only 10
cells: the directory names are self-describing, and every cell's `config.json` names
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

Empty. Writing these 10 cells for v5e is the next real chunk of work -- see the repo
README for sequencing.
