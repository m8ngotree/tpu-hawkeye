# tpu-hawkeye

A TPU/Pallas port of [Hawkeye](docs/papers/hawkeye-hardware-aware-gpu-kernel-optimization.pdf)
(Tschand, Ramakrishnan, et al.) -- grounding an autonomous LLM coding agent in a
minimal, hand-authored taxonomy of hardware-aware optimizations, and measuring the
speedup that taxonomy buys over the same agent without it, on
[JAXBench](external/accelerator-agents/JAXBench).

## The actual shape of this (see agent/README.md for the full correction)

Two separate things write code here:

1. **The taxonomy** -- 7 small, generic, hand-written unit-test cells (one per
   recurring optimization technique: MXU feed, DMA pipelining, fused epilogue, etc.),
   per TPU generation. We write these. See [taxonomy/README.md](taxonomy/README.md).
2. **The coding agent** -- a small hand-rolled tool-use loop (not the paper's own
   OpenHands harness -- see [agent/README.md](agent/README.md) for why), model-agnostic via any
   OpenAI-compatible API (DeepSeek by default, for cost). Given a generated workspace
   (task prompt, `eval.py`, optionally `taxonomy/` + `kernel_pool/`) it autonomously
   writes and iterates on kernels for JAXBench's 50 real workloads, in its own
   tool-driven loop (profile -> diagnose -> consult taxonomy -> edit -> re-evaluate).
   We don't write these kernels -- the agent does, at evaluation time.

The research question: run the same agent twice per workload, once with the taxonomy
in its context and once without (same tools minus taxonomy access, same turn budget).
The speedup delta is the result.

## Compute plan

- **Development**: a laptop, using Pallas's `interpret=True` CPU mode, so TPU time
  goes to real profiling and eval runs rather than debugging syntax errors.
- **Experiments**: a rented Cloud TPU VM on Google Cloud (a single v5e chip is enough:
  every JAXBench workload runs on one device). Step-by-step commands are in
  [docs/running_on_tpu.md](docs/running_on_tpu.md).
- **Later generations** (v5p, v6e, ...): each is a new sibling directory under
  `taxonomy/` (`v5p/`, `v6e/`, ...) with the same row names, matching how Hawkeye
  adds a GPU architecture as one new column.

## Repo layout

```
taxonomy/
  v5e/<NN>_<row_name>/        7 cells for the v5e generation (adapted from Hawkeye's 10 GPU rows -- see taxonomy/README.md)
    naive_kernel.py             deliberately unoptimized -- establishes the floor
    optimized_kernel.py           the one hand-written expert example for this technique
    config.json                    which profiler counter proves the technique fired
    guide.md                        protocol notes the code/counter alone don't convey
  README.md                     schema + the GPU-row -> TPU-row translation table

agent/
  runner.py                    tool: evaluate_kernel (compile, correctness, benchmark) -- done
  tools_exec.py                 the 3 tools given to the agent LLM: read_file, write_file, run_bash -- done
  harness.py                    the tool-use loop itself (OpenAI-compatible API, e.g. DeepSeek) -- done
  workspace.py                   builds the per-run agent workspace (task prompt, eval.py, taxonomy/, kernel_pool/) -- done
  profiler.py                    tool: deeper profiler counters -- stub
  tools.py                       our-side helpers: read_taxonomy_cell, kernel_pool_read/write -- done

eval/
  run_agent_eval.py            run the agent per workload and condition (taxonomy / none), re-score the final kernel -- done
  eval.py                       CLI the agent runs: `python eval.py --workload X --kernel kernel.py` -- done, verified
  smoke_test.py + smoke_kernels/  plumbing check for runner.py (no TPU needed) -- passing

external/
  accelerator-agents/          submodule (sparse-checked to JAXBench/) -- github.com/m8ngotree/accelerator-agents

docs/papers/                  reference papers (Hawkeye PDF)
scripts/                      verify_cells.py (run every cell on the current backend), tpu_vm_setup.sh
results/                      run outputs + kernel_pool/ (gitignored except summaries)
```

## Status

The taxonomy (7 cells), the agent harness, the evaluation CLI, the runner and the analysis
scripts are implemented and have been exercised on a real v5e chip. Only preliminary runs
exist so far; the full taxonomy-vs-no-taxonomy sweep is pending. For a complete
walkthrough of every part of the codebase, read
[docs/codebase_overview.md](docs/codebase_overview.md).
