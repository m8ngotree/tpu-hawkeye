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
2. **The coding agent** -- a small hand-rolled tool-use loop (not the paper's own
   OpenHands harness -- see [agent/README.md](agent/README.md) for why: OpenHands
   needs Docker, Kaggle notebooks don't have it), model-agnostic via any
   OpenAI-compatible API (DeepSeek by default, for cost). Given a generated workspace
   (task prompt, `eval.py`, optionally `taxonomy/` + `kernel_pool/`) it autonomously
   writes and iterates on kernels for JAXBench's 50 real workloads, in its own
   tool-driven loop (profile -> diagnose -> consult taxonomy -> edit -> re-evaluate).
   We don't write these kernels -- the agent does, at evaluation time.

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
  `taxonomy/` (`v5p/`, `v6e/`, ...) with the same row names, matching how Hawkeye
  adds a GPU architecture as one new column.

## Repo layout

```
taxonomy/
  v5e/<NN>_<row_name>/        8 cells for the v5e generation (trimmed+adjusted from Hawkeye's 10 GPU rows -- see taxonomy/README.md)
    naive_kernel.py             deliberately unoptimized -- establishes the floor
    optimized_kernel.py           the one hand-written expert example for this technique
    config.json                    which profiler counter proves the technique fired
    guide.md                        protocol notes the code/counter alone don't convey
  README.md                     schema + the GPU-row -> TPU-row translation table

agent/
  runner.py                    tool: evaluate_kernel (compile, correctness, benchmark) -- done
  tools_exec.py                 the 3 tools given to the agent LLM: read_file, write_file, run_bash -- done
  harness.py                    the tool-use loop itself (OpenAI-compatible API, e.g. DeepSeek) -- done, unverified against a real API call
  workspace.py                   builds the per-run agent workspace (task prompt, eval.py, taxonomy/, kernel_pool/) -- done
  profiler.py                    tool: deeper profiler counters -- stub
  tools.py                       our-side helpers: read_taxonomy_cell, kernel_pool_read/write -- done

eval/
  run_agent_eval.py            sweep the harness over JAXBench workloads, taxonomy vs. no-taxonomy -- stub
  eval.py                       CLI the agent runs: `python eval.py --workload X --kernel kernel.py` -- done, verified
  smoke_test.py + smoke_kernels/  plumbing check for runner.py (no TPU needed) -- passing

external/
  accelerator-agents/          submodule (sparse-checked to JAXBench/) -- github.com/m8ngotree/accelerator-agents

docs/papers/                  reference papers (Hawkeye PDF)
notebooks/                    Kaggle-runnable notebooks (bundle deps; Kaggle sessions are ephemeral)
results/                      run outputs + kernel_pool/ (gitignored except summaries)
```

## Status

`runner.py`, `eval.py`, `tools_exec.py`, and `workspace.py` are implemented and
verified end-to-end on CPU. `harness.py` is implemented but not yet exercised against
a real LLM API call. Taxonomy: `01_mxu_feed` through `05_producer_consumer` written
and correctness-verified (CPU/interpret mode only, not yet on real TPU hardware); 3
cells remain. Next real chunk of work: the rest of the taxonomy, then smoke-test
`harness.py` against one cheap agent turn before running a full sweep.
