# tpu-hawkeye

A TPU/Pallas port of [Hawkeye](docs/papers/hawkeye-hardware-aware-gpu-kernel-optimization.pdf)
(Tschand, Ramakrishnan, et al.) -- grounding an autonomous LLM coding agent in a
minimal, hand-authored taxonomy of hardware-aware optimizations, and measuring the
speedup that taxonomy buys over the same agent without it, on
[JAXBench](external/accelerator-agents/JAXBench).

## The actual shape of this (see agent/README.md for the full correction)

Two separate things write code here:

1. **The taxonomy** -- 10 small, generic, hand-written unit-test cells (one per
   recurring optimization technique: MXU feed, DMA pipelining, quantization, etc.),
   per TPU generation. We write these. See [taxonomy/README.md](taxonomy/README.md).
2. **The coding agent** -- [OpenHands](https://docs.openhands.dev) (same backend the
   Hawkeye paper itself used), given a generated workspace (task prompt, `eval.py`,
   optionally `taxonomy/` + `kernel_pool/`) autonomously writes and iterates on
   kernels for JAXBench's 50 real workloads, in its own bash-driven loop (profile ->
   diagnose -> consult taxonomy -> edit -> re-evaluate). We don't write these kernels
   -- the agent does, at evaluation time. See [agent/README.md](agent/README.md).

The research question: run the same agent twice per workload, once with the taxonomy
in its context and once without (same tools minus taxonomy access, same turn budget).
The speedup delta is the result.

## Compute plan

- **Now**: Kaggle TPU v5e-8 (free, 20 hrs/week). Taxonomy cells and kernel-plumbing
  are built and validated here first. Local development uses Pallas's
  `interpret=True` CPU mode so TPU hours go to real profiling/eval runs, not
  debugging syntax errors.
- **Later**: GCP TPU VMs for v5p / v6e (Trillium) / v7 (Ironwood), per
  [Cloud TPU pricing](https://cloud.google.com/tpu/pricing) -- compute isn't the
  constraint long-term. Each new generation is a new sibling directory under
  `taxonomy/` (`v5p/`, `v6e/`, ...) with the same 10 row names, matching how Hawkeye
  adds a GPU architecture as one new column.

## Repo layout

```
taxonomy/
  v5e/<NN>_<row_name>/        10 cells for the v5e generation
    naive_kernel.py             deliberately unoptimized -- establishes the floor
    optimized_kernel.py           the one hand-written expert example for this technique
    config.json                    which profiler counter proves the technique fired
    guide.md                        protocol notes the code/counter alone don't convey
  README.md                     schema + the GPU-row -> TPU-row translation table

agent/
  runner.py                    tool: evaluate_kernel (compile, correctness, benchmark) -- done
  profiler.py                  tool: deeper profiler counters -- stub
  tools.py                     tools: read_taxonomy_cell, kernel_pool_read/write -- done
  harness.py                   wraps the tools for an LLM API, drives the turn loop -- not built
                                 (LLM/agent-SDK choice not yet made)

eval/
  run_agent_eval.py            sweep the harness over JAXBench workloads, taxonomy vs. no-taxonomy
  smoke_test.py + smoke_kernels/  plumbing check for runner.py (no TPU needed) -- passing

external/
  accelerator-agents/          submodule (sparse-checked to JAXBench/) -- github.com/m8ngotree/accelerator-agents

docs/papers/                  reference papers (Hawkeye PDF)
notebooks/                    Kaggle-runnable notebooks (bundle deps; Kaggle sessions are ephemeral)
results/                      run outputs + kernel_pool/ (gitignored except summaries)
```

## Status

`runner.py` is implemented and verified end-to-end on CPU (see `eval/smoke_test.py`).
Everything else is scaffolding. Next real chunk of work: write the 10 v5e taxonomy
cells, then build `agent/harness.py` once the LLM backend is chosen.
