# tpu-hawkeye

A Hawkeye-style, hardware-aware kernel optimization agent for TPUs (Pallas/Mosaic),
evaluated against [JAXBench](external/accelerator-agents/JAXBench). Named after
[Hawkeye](https://www.alphaxiv.org/abs/2608.hawkeye-hardware-aware-gpu-kernel-optimizationv1),
which does the same thing for GPUs (CUDA/Triton on Ampere/Hopper/Blackwell/MI350).

## Idea

Hawkeye's core insight isn't the agent loop (edit -> run -> profile -> diagnose -> patch
is standard by now) -- it's the **taxonomy**: a small, hand-curated matrix of
(optimization pattern x target architecture) cells, each with a naive kernel, an
expert-optimized kernel, the profiler metrics that distinguish them, and a prose
explanation of *why*. Grounding the agent in ~10 unit tests per architecture beats
grounding it in raw vendor docs or a pile of production kernels.

This project builds that taxonomy for TPUs instead of GPUs, and measures whether an
agent armed with it beats a base LLM (no taxonomy, just docs/RAG) on JAXBench.

## Compute plan

- **Now**: Kaggle TPU v5e-8 (free, 20 hrs/week). All taxonomy cells are built and
  validated here first. Local development against `interpret=True` (CPU) so TPU
  hours are spent on real profiling/benchmark runs, not debugging.
- **Later**: GCP TPU VMs for v5p / v6e (Trillium) / v7 (Ironwood) once the v5e column
  is solid, per [Cloud TPU pricing](https://cloud.google.com/tpu/pricing). Each new
  generation is a new *column* added to the taxonomy -- the row (pattern) definitions
  should stay generation-agnostic where possible so porting is "re-profile and adjust
  the cell," not "start over."

## Repo layout

```
taxonomy/            the 2D matrix: pattern (row) x kernel-category (column)
  patterns/<pattern>/<category>/
    naive.py            unoptimized Pallas kernel
    expert.py            hand-optimized reference kernel
    metrics.yaml          expected profiler signature (MXU util, VMEM occ, DMA-bound?, roofline regime)
    protocol.md            prose: what the pattern is, why it helps, when to reach for it
  README.md               taxonomy schema + how to add a cell

agent/                the optimization loop
  editor.py               applies a patch to a candidate kernel
  runner.py               compiles + executes on TPU, numerical validation
  profiler.py              wraps JAX/XLA profiler -> XProf/Perfetto trace parsing
  diagnose.py              trace -> bottleneck classification (compute/memory/DMA-bound, roofline)
  retriever.py              bottleneck + kernel category -> taxonomy cell lookup
  loop.py                   orchestrates the above, calls the LLM to propose the patch

eval/                 wiring into JAXBench's harness
  run_agent_eval.py         drive agent over JAXBench workloads, collect speedup vs base-LLM
  baselines/                 base-LLM-only runs (no taxonomy) for comparison

external/
  accelerator-agents/       submodule (sparse-checked to JAXBench/) -- github.com/m8ngotree/accelerator-agents

notebooks/            Kaggle-runnable notebooks (bundle deps; Kaggle sessions are ephemeral)
results/              JSON/CSV run outputs (gitignored except summaries)
```

## Status

Scaffolding only. See [taxonomy/README.md](taxonomy/README.md) for the schema and
[agent/README.md](agent/README.md) for the loop design before filling in cells.
