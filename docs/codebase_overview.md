# tpu-hawkeye: complete codebase overview

This document explains the whole project for someone who has never seen it: what it is
trying to show, the background you need, every directory and file, how one experiment
run flows from start to finish, how results are scored, what has been measured so far,
and what is unfinished. Read it top to bottom once; afterwards use the table of contents
as a reference.

1. [What this project is](#1-what-this-project-is)
2. [Background concepts](#2-background-concepts)
3. [The big picture](#3-the-big-picture)
4. [Repository map](#4-repository-map)
5. [The taxonomy](#5-the-taxonomy)
6. [The agent](#6-the-agent)
7. [Evaluation and scoring](#7-evaluation-and-scoring)
8. [Experiment design and integrity](#8-experiment-design-and-integrity)
9. [Scripts](#9-scripts)
10. [Running things](#10-running-things)
11. [Analysing a run](#11-analysing-a-run)
12. [Status, results so far, limitations](#12-status-results-so-far-limitations)
13. [Glossary](#13-glossary)

---

## 1. What this project is

**The idea, in one paragraph.** Large language models can write GPU/TPU kernels (small,
highly optimized programs), but they do it badly on hardware they have seen little of,
because they do not know the hardware's specific tricks. The paper *Hawkeye* (Tschand,
Ramakrishnan, et al.; PDF in `docs/papers/`) showed that giving a coding agent a small,
hand-written, well-organized reference collection of "hardware-aware optimizations"
(a *taxonomy*) makes it write much faster GPU kernels. This project ports that idea from
NVIDIA/AMD GPUs to **Google TPU v5e** and the **Pallas** kernel language.

**The research question.** Take one autonomous LLM coding agent. Run it on the same
optimization tasks twice: once with the taxonomy available, once without. Same model, same
tools, same number of turns. How much faster are the kernels it produces with the
taxonomy? That difference is the result. (The paper already argues the taxonomy is a good
form of context; this project only measures *taxonomy vs. no taxonomy* on TPU.)

**Two separate things write code here. Do not confuse them.**

| | Who writes it | When | What it is |
|---|---|---|---|
| The taxonomy (`taxonomy/`) | Us, by hand | Before experiments | 7 small generic example "cells", one per optimization technique |
| Kernels for benchmark workloads | The LLM agent, autonomously | During an experiment run | The thing being measured |

We never hand-write kernels for the benchmark workloads. The only hand-written kernels are
the seven small generic ones inside the taxonomy, which demonstrate one technique each.

---

## 2. Background concepts

You need these terms to read the rest.

**TPU v5e.** A Google AI accelerator chip. This project uses a single chip (Cloud name:
`v5litepod-1`; JAX reports the device as `TPU v5 lite`). Peak throughput: 197 TFLOPS for
bfloat16 matrix math; memory bandwidth: 819 GB/s.

**The parts of the chip that matter.**
- **HBM**: the large, slow-ish main memory where tensors live.
- **VMEM**: small, fast on-chip scratchpad memory. Kernels compute on data in VMEM.
- **MXU** (matrix unit): a systolic array that does matrix multiplication at very high
  speed. It provides almost all of the chip's peak FLOPS.
- **VPU** (vector unit): does elementwise and reduction work, far slower than the MXU for
  matmuls.
- Data must be copied HBM to VMEM before compute, and back afterward (DMA transfers).
  Good kernels overlap these copies with compute.

**JAX / XLA.** JAX is a Python array library. `jax.jit` compiles functions with the XLA
compiler. Ordinary JAX code is compiled to reasonable TPU code automatically; that is the
"baseline" that hand-written kernels try to beat.

**Pallas.** A JAX extension for writing custom kernels. You write a *kernel body* that
operates on small blocks in VMEM, and `pl.pallas_call` runs it over a *grid* of blocks.
`BlockSpec` says which block of each array a grid step sees. On TPU, Pallas is compiled by
**Mosaic**. Two Pallas facts used throughout:
- A `BlockSpec`'s `index_map` returns *block indices*, not element offsets.
- On TPU, a block's last two dimensions must each be divisible by 8 and 128 respectively
  (or equal the full array dimension); violations are rejected, not padded.

**Roofline.** A workload is either *memory-bound* (limited by how fast bytes move from
HBM) or *compute-bound* (limited by how fast the MXU can do FLOPs). Its arithmetic
intensity (FLOPs per byte) compared with the chip's ridge point (peak FLOPS divided by peak
bandwidth; about 240 flop/byte on v5e) says which. A kernel's distance from its limit tells
you how much room is left.

**JAXBench.** A benchmark of 50 JAX/TPU workloads (`external/accelerator-agents/JAXBench`,
a git submodule): 17 "priority" workloads (`1p`..`17p`: attention variants, GEMM, RMSNorm,
mixture-of-experts, etc.) and 33 KernelBench-style ones (`18k`..`50k`: fused matmul/conv
chains). Each workload directory has a `baseline.py` (plain JAX reference with
`create_inputs()` and `workload(*inputs)`); some also have an `optimized.py` (a hand-tuned
Pallas answer key, which we deliberately hide from the agent). JAXBench also supplies the
evaluation harness (`evaluate_kernel`): checks correctness (tolerance 1e-2) and times
kernels with the device profiler. Speedup means `baseline_time / kernel_time`.

**Hawkeye.** The paper. A taxonomy is a 2-D table: rows are optimization techniques,
columns are hardware. Each cell holds a naive kernel, an optimized kernel, a config naming
the profiler metric that proves the technique worked, and a guide. An agent runs a closed
loop: evaluate, profile, diagnose the bottleneck, consult the matching cell, edit,
re-evaluate. The agent chooses the cell itself; nothing in code picks it.

---

## 3. The big picture

```
                    (one experiment run = one workload, one condition, one repeat)

  build_workspace()              run_agent()                       scoring
  ┌──────────────────┐      ┌─────────────────────────┐      ┌────────────────────────┐
  │ private folder:  │      │ LLM loop, up to N turns │      │ take best_kernel.py    │
  │  task_prompt.md  │      │  reads files            │      │ (else kernel.py)       │
  │  baseline.py     │ ───► │  writes kernel.py       │ ───► │ re-run with repo       │
  │  kernel.py       │      │  runs eval.py (TPU)     │      │ JAXBench harness       │
  │  eval.py         │      │  reads the diagnosis    │      │ -> result.json         │
  │  JAXBench copy   │      │  (taxonomy: reads cells)│      │ status, speedup, tokens│
  │  [taxonomy/]     │      └─────────────────────────┘      └────────────────────────┘
  └──────────────────┘             trajectory.jsonl
     [taxonomy/] only in the "taxonomy" condition; everything else identical.
```

The entry point for experiments is `eval/run_agent_eval.py`, which loops this over
workloads × conditions × repeats and writes results to `results/`.

---

## 4. Repository map

```
tpu-hawkeye/
├── README.md                     short project intro
├── pyproject.toml                Python package metadata / dependencies
├── taxonomy/
│   ├── README.md                 INTERNAL design notes (not shown to the agent)
│   └── v5e/                      one directory per TPU generation
│       └── NN_<row_name>/        7 cells: naive_kernel.py, optimized_kernel.py,
│                                 config.json, guide.md
├── agent/
│   ├── harness.py                the LLM tool-use loop
│   ├── tools_exec.py             the 3 tools the LLM can call (read/write/run_bash)
│   ├── workspace.py              builds the per-run workspace + task prompt
│   ├── runner.py                 wraps JAXBench evaluate_kernel for our own scoring
│   ├── tools.py                  helper functions for us (taxonomy/pool access)
│   ├── profiler.py               stub, unused
│   └── README.md                 older design notes (partly outdated, see §12)
├── eval/
│   ├── eval.py                   CLI the AGENT runs to test its kernel
│   ├── run_agent_eval.py         CLI WE run to execute a full experiment
│   ├── smoke_test.py, smoke_kernels/   plumbing check without a TPU or LLM
│   └── baselines/README.md
├── scripts/                      utilities (see §9)
├── docs/
│   ├── codebase_overview.md      this file
│   ├── running_on_tpu.md         Google Cloud TPU runbook
│   └── papers/                   the Hawkeye PDF
├── external/accelerator-agents/  git submodule containing JAXBench
└── results/                      run outputs (raw runs are git-ignored;
                                  files named *summary*.json are kept)
```

---

## 5. The taxonomy

### 5.1 Structure

`taxonomy/v5e/` is the single "column" (v5e). Each of the 7 subdirectories is a **cell**
(one optimization technique) with four files:

| File | Purpose |
|---|---|
| `naive_kernel.py` | Deliberately unoptimized version, so the cell's metric reads at its floor |
| `optimized_kernel.py` | The hand-written expert example; also serves as a syntax reference |
| `config.json` | Machine-readable: which measurement proves the technique worked, and which direction is better |
| `guide.md` | Formal explanation: rule, hardware reason, gotchas, and a **Diagnosis** section saying when to use this cell |

A new TPU generation (v5p, v6e, ...) means adding a sibling directory with the same row
names; nothing else changes.

### 5.2 The seven cells

| Cell | What it teaches | Naive vs. optimized |
|---|---|---|
| `01_mxu_feed` | A matmul must be written with `jnp.dot` / `dot_general` to reach the MXU; equivalent math (broadcast-multiply and sum) silently runs on the slow VPU. Use `preferred_element_type=jnp.float32` for bf16. | multiply+sum vs `jnp.dot` |
| `02_vmem_tile_layout` | Legal block shapes (last two dims divisible by (8,128)); among legal shapes, larger blocks cut per-grid-step overhead. | (8,128) blocks vs (128,1024) blocks on a 1024×1024 bf16 array |
| `03_vectorized_vmem` | Operate on a whole VMEM block at once; per-lane Python loops fall back to slow scalar-style code. | per-lane loop vs whole-block op |
| `04_async_pipeline` | Overlap HBM↔VMEM DMA with compute using `pltpu.emit_pipeline` (multi-stage buffering). Also records that buffer depth 2/3/4 made no difference on v5e. | pipelining off vs on |
| `05_fused_epilogue` | Fuse bias/activation into the matmul kernel so the intermediate never round-trips through HBM. | two `pallas_call`s vs one, computing relu(A@B+bias) at 2048×1024×1024 |
| `06_lane_reduction` | Reductions should use native cross-lane ops (`jnp.sum(axis=...)`), keeping the result 2-D (`keepdims=True`) because Mosaic cannot compile 1-D results. | 128 single-lane adds vs one `jnp.sum` |
| `07_grouped_matmul` | Selecting a per-step operand by a runtime index array using scalar prefetch (`PrefetchScalarGridSpec`), instead of keeping every candidate resident in VMEM. | whole-tensor VMEM-resident vs prefetch-indexed |

All seven were compiled and run on a real v5e chip, verified numerically against their
naive counterparts, and showed a measurable device-time gain at the chosen problem size.
Reproduce with `scripts/verify_cells.py`.

### 5.3 Content rules (important)

Everything in `taxonomy/v5e/` is copied verbatim into the agent's workspace, so it must:
- describe hardware and Pallas/Mosaic mechanics only, using generic examples built from
  primitive operations (elementwise ops, matmuls, reductions);
- **not** name any benchmark workload, model, or neural-network architecture;
- **not** contain project commentary (revision history, corrections, "verified locally"
  notes);
- read as formal reference documentation.

The reason: JAXBench is the evaluation set. Naming its workloads or architectures in the
taxonomy would leak the test into the training material. Workload-specific reasoning (why
a row exists, which workloads motivated it) lives in `taxonomy/README.md`, which is *not*
copied to workspaces.

### 5.4 How the agent uses a cell (no retriever)

There is no search index or lookup table. The agent's workspace contains a `taxonomy/`
directory; the task prompt only *mentions* it. The agent lists it and reads files with its
own tools when it decides to. Each guide's **Diagnosis** section states, in terms of the
fields of `eval.py`'s `diagnosis` block (see §7.3), when that cell applies. Example:
`04_async_pipeline` applies when `workload_limit` is memory-bound and
`hbm_bandwidth_pct_of_peak` is well below 100. The model matches its observed numbers to
that text.

### 5.5 How the rows were chosen

Started from Hawkeye's 10 GPU rows. Kept and translated MMA unit, shared-memory layout,
vectorized memory, async pipeline, epilogue, and warp reduction. Dropped: quantized
precision (workloads are mostly bf16), multi-unit coordination (multi-chip only),
producer/consumer and persistent scheduling (measured on v5e: no effect to demonstrate).
Added `07_grouped_matmul` after reading all 50 JAXBench workloads and finding several
whose group boundaries are data-dependent. Convolutions were deliberately not added; XLA's
convolution lowering is already reasonable. Full trace in `taxonomy/README.md`.

---

## 6. The agent

### 6.1 Design

A small hand-written loop (not the paper's OpenHands framework): no Docker, fully readable,
and identical across both conditions, which is what matters for the comparison. It talks to
any OpenAI-compatible chat API. We use DeepSeek (`deepseek-v4-pro`, `deepseek-flash`) for
cost; the API key is read from the environment variable `LLM_API_KEY`.

### 6.2 The workspace (`agent/workspace.py`)

`build_workspace()` creates a private folder for each run (default under
`~/hawkeye_work/<tag>/<condition>/<workload>_r<rep>`, outside the repo):

```
task_prompt.md      the task text (see 6.3)
baseline.py         the reference workload (the problem definition)
kernel.py           what the agent edits; starts as a copy of baseline.py
eval.py             the evaluation CLI (see §7)
JAXBench/           a private copy of the harness + ONLY this workload's baseline
taxonomy/           the 7 cells (taxonomy condition only)
run_config.json     which settings this run used
```

The workspace is self-contained: it contains no reference to the repo, other workloads, or
the hand-tuned `optimized.py` answer keys.

### 6.3 The task prompt

`task_prompt.md` is a fixed template, filled with a problem-type string, the TPU generation
and the workload name. It states: the score is TFLOPS relative to baseline and correctness
comes first; what `baseline.py`, `kernel.py`, and `eval.py` are; what the `diagnosis` block
contains; that `eval.py` remembers the fastest correct kernel; and a workflow (evaluate,
identify the limiter, make one targeted change, re-evaluate, stop when out of ideas). In the
taxonomy condition it adds one short paragraph describing the `taxonomy/` folder. That
paragraph is the *only* difference between conditions.

There is **no natural-language description of the workload**. The agent must read
`baseline.py` to learn the math, shapes, and dtypes.

### 6.4 The three tools (`agent/tools_exec.py`)

- `read_file(path)`: returns file contents (paths clamped to the workspace).
- `write_file(path, content)`: overwrites or creates a file.
- `run_bash(command)`: runs a shell command in the workspace with a 300 s timeout. Output is
  clipped to 8000 characters (first 3000, an "omitted" marker, then the last 5000) so long
  tracebacks and compiler logs do not flood the context.

`run_bash` rejects commands that mention locations outside the workspace (parent
directories, `~`, `/home`, the repo name, and similar). This is a pattern guard, not a real
sandbox: code the agent writes could still open arbitrary paths. Each rejection is counted
(`guard_rejections`) so runs can be audited. The real safety boundary is the disposable VM.

### 6.5 The loop (`agent/harness.py`)

1. Messages = system prompt + `task_prompt.md`.
2. Send messages and tool schemas to the model.
3. If the reply has tool calls, execute each, append results as tool messages, repeat.
4. Stop when the model replies without tool calls or after `max_turns`.

Every turn, the **entire** message list is re-sent. So files the agent has read stay in its
context for the rest of the run (and are paid for again as input tokens each turn); nothing
is deduplicated or summarized. Nothing persists between runs; the model learns only within
a run's context.

Each trajectory line records the turn, role, content, tool calls or outputs, and now also
timestamps, API seconds, per-turn prompt/completion tokens, and tool seconds. The run
result carries the totals.

---

## 7. Evaluation and scoring

### 7.1 What the agent runs: `eval/eval.py`

`python eval.py --workload <name> --kernel kernel.py --tpu v5e [--interpret]`. Inside a
workspace it uses the workspace's private JAXBench copy. It calls `evaluate_kernel`, which:
1. loads the baseline and the candidate;
2. checks the candidate's output against the baseline's (atol = rtol = 1e-2);
3. times both with the device profiler (parses `jit_*` device events from a Perfetto trace
   captured by `jax.profiler`);
4. returns status (`correct`, `incorrect`, `compile_error`, `runtime_error`, ...), median
   times, TFLOPS, utilization, and speedup.

`--interpret` runs Pallas on CPU (correctness only, no TPU time). On TPU always pass
`--tpu v5e`: JAXBench's auto-detection does not recognize `TPU v5 lite`.

### 7.2 Best-kernel tracking

After each evaluation inside a workspace, `eval.py` appends a line to `eval_log.jsonl`
(status, speedup, and `is_reference`, true if the kernel is identical to the baseline copy).
If the result is correct and the fastest so far, the kernel is saved as `best_kernel.py`
with `best_score.json`. So a failed later experiment never loses earlier progress, and the
score is "best correct kernel the agent ever evaluated", as in the paper.

### 7.3 The `diagnosis` block

For a correct kernel, the JSON ends with a roofline view. Minimum HBM traffic is the
workload's inputs plus its output, each moved once (computed from shapes, not from XLA's
cost analysis, which reports meaningless traffic for Pallas kernels).

| Field | Meaning |
|---|---|
| `min_hbm_traffic_mb` | inputs + output size |
| `achieved_hbm_gbs`, `hbm_bandwidth_pct_of_peak` | that traffic divided by kernel time, against 819 GB/s |
| `mxu_pct_of_peak` | achieved TFLOPS against 197 |
| `arithmetic_intensity_flop_per_byte`, `ridge_point_flop_per_byte` | which side of the roofline the workload sits on |
| `workload_limit` | "memory-bound" or "compute-bound" |
| `pct_of_roofline_limit` | how close the kernel is to that limit; 100 means the chip cannot go faster |

Example: a memory-bound kernel at `pct_of_roofline_limit: 80` has at most ~20% left to gain.
The block says *how far* from the limit and *which* limit, not *why*. It is the same in both
conditions; the taxonomy's job is to help the agent act on it.

### 7.4 Independent final scoring (`eval/run_agent_eval.py`)

The agent's own claims are never trusted. After a run ends, the runner re-scores
`best_kernel.py` (or `kernel.py` if none) itself using `agent/runner.py` (wrapping the
repo's JAXBench harness, 5 warmup and 50 timed iterations) and writes `result.json`.

Key fields: `workload`, `condition`, `rep`, `tag`, `turns_used`, `stopped_reason`
(`done`, `max_turns`, `interrupted`, `error`), `scored_file`, `status`, `correct`,
`speedup_vs_baseline`, `kernel_median_ms`, `baseline_median_ms`, `error`,
`guard_rejections`, token and time totals, and the agent-side counts
`agent_eval_calls`, `agent_own_kernel_evals`, `agent_own_kernels_correct`,
`agent_own_best_speedup` (the agent's own kernels only, excluding evaluations of the
untouched baseline copy).

### 7.5 Runner behavior

Arguments: `--workloads` (space-separated names), `--conditions taxonomy none`, `--reps`,
`--max-turns`, `--model`, `--base-url`, `--api-key-env`, `--tag`, `--work-dir`, `--quiet`,
`--dry-run`, `--interpret`, `--force`. Runs that already have a `result.json` are skipped
(except `harness_error` ones, which are retried), so an interrupted or preempted sweep can
be resumed by re-running the same command. Each finished run is saved to
`results/runs/<tag>/<condition>/<workload>_r<rep>/`:

```
result.json  trajectory.jsonl  kernel.py  best_kernel.py  eval_log.jsonl  task_prompt.md
```

and a summary to `results/<tag>_summary.json`.

Terminology: a **run** is one attempt (one workload, one condition, one repeat); a **rep**
is a repeat of the same setup, needed because the LLM is nondeterministic.

---

## 8. Experiment design and integrity

**Conditions.** `taxonomy` and `none`. Everything else is held equal: model, tools, prompt
(minus the taxonomy paragraph), turn budget, evaluation, scoring.

**Metric.** Speedup of the best correct kernel over the JAXBench baseline, re-scored
independently. Also report how often a correct kernel was produced at all, and cost (tokens).

**Safeguards already in place.**
- Self-contained workspaces; only the assigned workload's baseline is present.
- Answer keys (`optimized.py`) excluded.
- Pattern guard plus rejection audit.
- Independent re-scoring.
- `is_reference` separates the agent's own kernels from the copied baseline.
- Resumable runner.
- Taxonomy content rules (no benchmark leakage).

**Known limitations.**
- The guard is not a true sandbox.
- Each `baseline.py` includes a `CONFIG` dict that can name the source model (for example a
  named LLM's operator dimensions). The agent can read it in both conditions; it does not
  bias the comparison but may help the model recall known designs.
- The diagnosis block is thinner than Hawkeye's profiler counters (TPU exposes no stall or
  conflict counters for Pallas kernels: the profiler reports the kernel as one opaque
  custom call with zero hardware metrics).
- Results depend on the model; a weaker model may exaggerate the taxonomy's benefit.
- Few reps per cell means noisy comparisons; use several reps and several workloads.

---

## 9. Scripts

| Script | Purpose |
|---|---|
| `scripts/tpu_vm_setup.sh` | Run once on a fresh TPU VM: clones the repo, creates `.venv-tpu`, installs `jax[tpu]==0.6.2`, `openai`, `numpy`, prints the TPU devices |
| `scripts/verify_cells.py` | Runs every taxonomy cell's naive and optimized kernel with device timing (`--interpret` for CPU) and writes a summary JSON |
| `scripts/probe_buffering.py` | Sweeps block size and buffer depth (the measurement behind the async-pipeline finding) |
| `scripts/inspect_profile.py` | Prints XLA cost analysis and profiler trace events for a workload/kernel; used to decide which signals the agent can see |
| `scripts/show_trajectory.py` | Prints a saved trajectory compactly, turn by turn |
| `scripts/analyze_taxonomy_use.py` | For saved runs: which cell files were opened and when, evaluations after the first read, and which cell-specific code constructs show up in the kernels the agent wrote |
| `scripts/score_workspace.py` | Scores an interrupted workspace and saves it like a finished run (`stopped_reason: interrupted`) |

---

## 10. Running things

**On a laptop (no TPU).** Create a venv, install the package, then check plumbing with
`python eval/smoke_test.py` and `python scripts/verify_cells.py --interpret`. Pallas
`interpret=True` runs kernels on CPU, so correctness can be checked without spending TPU
time. CPU timings are meaningless.

**On a TPU VM.** Full step-by-step (quota, creating a `v5litepod-1`, SSH, setup, deleting)
is in `docs/running_on_tpu.md`. Summary:

```bash
# laptop
gcloud compute tpus tpu-vm create hawkeye --zone=us-west4-a \
  --accelerator-type=v5litepod-1 --version=v2-alpha-tpuv5-lite --spot
gcloud compute tpus tpu-vm ssh hawkeye --zone=us-west4-a

# on the VM (once)
bash <(curl -s https://raw.githubusercontent.com/m8ngotree/tpu-hawkeye/main/scripts/tpu_vm_setup.sh)
cd ~/tpu-hawkeye && source .venv-tpu/bin/activate
read -s -p "API key: " K; echo "export LLM_API_KEY=$K" > ~/.llm_key; chmod 600 ~/.llm_key; unset K

# verify the taxonomy on hardware, then run an experiment
python scripts/verify_cells.py
source ~/.llm_key
python eval/run_agent_eval.py --workloads 8p_GEMM --conditions taxonomy none --reps 1 \
  --max-turns 48 --model deepseek-v4-pro --tag mytest
```

Use `tmux` for long runs so a dropped connection does not kill them. Copy `results/` off
the VM before deleting it (`gcloud compute tpus tpu-vm scp --recurse ...`). **Delete the VM
when finished; it bills by the hour**, then confirm with `gcloud compute tpus tpu-vm list`.

**Practical notes.** Spot TPUs can be preempted (the resumable runner handles it). Google
Cloud in some organizations blocks external IPs by policy; see `docs/running_on_tpu.md`.
Never paste an API key into chat, logs, or the repo; if one leaks, revoke it.

---

## 11. Analysing a run

Per run folder under `results/runs/<tag>/<condition>/<workload>_r<rep>/`:
- `result.json`: outcome, independent score, token and time counts.
- `trajectory.jsonl`: every turn (what it read, what it wrote, tool outputs, errors).
- `eval_log.jsonl`: the sequence of evaluations (progress over time; failures).
- `best_kernel.py`: the scored kernel; diff it against `baseline.py`.
- `kernel.py`: where the agent ended.

Questions and where to look:
- Did it beat the baseline? `result.json`.
- How much effort was wasted? `agent_own_kernels_correct / agent_own_kernel_evals`.
- What errors cost it turns? Tool outputs in `trajectory.jsonl`.
- Did it use the taxonomy, and did it help? `scripts/analyze_taxonomy_use.py`, then compare
  with the same workload's `none` run.
- How close is the kernel to the ceiling? The `diagnosis` block in its last `eval.py` output.

Only the best and final kernels are saved; earlier attempts are recoverable from the
`write_file` arguments in the trajectory.

---

## 12. Status, results so far, limitations

**Built and working.** All 7 cells verified on real v5e. Agent harness, workspace builder,
tools, evaluation CLI with best-kernel tracking and roofline diagnosis, independent scoring,
resumable runner with progress printing, token/time accounting, and analysis scripts.
Runs execute on a rented Cloud TPU v5e VM.

**Results so far (preliminary).** Smoke and pilot runs on a few workloads confirmed the
pipeline end to end. One taxonomy-condition run (RMSNorm, `deepseek-v4-pro`, 48 turns,
interrupted) scored a correct 1.43× kernel, with 9 of 14 of the agent's own kernels
correct; the diagnosis shows it at about 80% of the memory roofline. The matching
no-taxonomy run at the same budget does **not** exist yet, so there is currently **no
evidence about whether the taxonomy helps**. That comparison, over several workloads with
several repeats, is the pending experiment.

**Not built / open items.**
- The full sweep (workloads × both conditions × multiple repeats).
- `kernel_pool/` (cross-workload reuse of the agent's own kernels): the workspace code
  supports it but it is disabled in runs.
- `agent/profiler.py` is an unused stub.
- Only the v5e column exists; no v5p/v6e columns.
- No hardware counters beyond the roofline diagnosis.
- `agent/README.md` and the top-level `README.md` contain some outdated status notes
  (for example calling completed pieces "stubs"); this document is the current reference.

**Related decisions to remember.** Taxonomy content is formal reference only (no workload
names). The comparison is taxonomy vs. none, not vs. other context types. Both conditions
get the same diagnosis block.

---

## 13. Glossary

- **Cell**: one taxonomy entry (one technique on one hardware generation): naive kernel,
  optimized kernel, config, guide.
- **Condition**: `taxonomy` (agent has the cells) or `none` (it does not).
- **Rep**: a repeat of the same workload and condition.
- **Run**: one agent attempt at one workload; scored once.
- **Turn**: one model call plus the tool calls it makes.
- **Workspace**: the private folder an agent works in.
- **Trajectory**: the log of every turn in a run.
- **Speedup**: baseline time divided by kernel time (>1 is faster).
- **Best correct kernel**: the fastest kernel that passed the correctness check.
- **HBM / VMEM / MXU / VPU**: main memory / on-chip scratchpad / matrix unit / vector unit.
- **BlockSpec, grid, index_map**: Pallas's way of splitting arrays into blocks and mapping
  grid steps to blocks.
- **Mosaic**: the compiler that lowers Pallas kernels to TPU code.
- **Roofline**: performance limit set by either memory bandwidth or peak compute.
- **Interpret mode**: running Pallas on CPU for correctness checks only.
