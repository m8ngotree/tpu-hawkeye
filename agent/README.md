# Agent

**Correction from an earlier version of this README:** this is not a scripted
profile -> classify-bottleneck -> retrieve-cell -> patch pipeline. That was modeling
the wrong thing. In Hawkeye, an LLM coding agent (given tools + a short system
prompt) does the profiling, the bottleneck diagnosis, and the taxonomy lookup itself,
as reasoning steps in its own tool-use loop (paper Section 2.4, Appendix E.1, Figure
47). This codebase's job is to give it the right tools and stay out of the way -- not
to pre-decide what it should conclude.

## Backend: OpenHands

This matches what the Hawkeye paper itself used (Appendix G.1, "MSWEA" harness) --
`pip install openhands` (needs Python 3.12+, needs Docker for its sandboxed runtime,
needs an LLM API key configured separately). Important architectural consequence:
**OpenHands drives the agent through bash commands in a sandboxed workspace, not
function-calling tools.** So the "tools" below are exposed as plain files/scripts the
agent can `cat` and `python` from a shell, not as an API tool schema.

**Model choice is independent of this decision.** OpenHands routes LLM calls through
LiteLLM, so any backbone works, including cost-cheaper OSS models like DeepSeek --
set `LLM_MODEL="deepseek/deepseek-chat"` (or whichever DeepSeek endpoint/model you
want) and `LLM_API_KEY` before running `openhands`. This also means later comparing
multiple backbone models (as the paper does in Figure 2a's Pass@1 sweep) is just
re-running with a different `LLM_MODEL`, not a harness change.

## Pieces

1. **`agent/runner.py`** (done, verified) -- `run_kernel()`: wraps JAXBench's
   `evaluate_kernel` (compile, correctness, benchmark). Exposed to the OpenHands agent
   as **`eval/eval.py`**, a CLI wrapper (`python eval.py --workload X --kernel kernel.py`)
   -- this is the file the agent actually runs from its shell. Matches Fig. 48's
   `eval.py` role exactly.

2. **`agent/workspace.py`** (done) -- `build_workspace()` builds the directory
   OpenHands operates in for one run, matching Hawkeye's Figure 22 tree: `kernel.py`
   (the agent edits this), `eval.py`, `task_prompt.md` (the task, modeled on Fig. 48's
   skeleton), and conditionally `taxonomy/` + `kernel_pool/`. `use_taxonomy=False`
   drops the `taxonomy/` directory entirely -- that's the baseline condition (Table 4).

3. **`agent/profiler.py`** (stub) -- deeper counters (DMA-wait fraction, VMEM
   occupancy) beyond what `eval.py` already reports (TFLOPS, median_ms,
   utilization_pct). Needed once a taxonomy cell's `config.json` names a counter that
   `evaluate_kernel` doesn't already surface. Would also need to become a CLI script
   under the workspace, same pattern as `eval.py`.

4. **`agent/tools.py`** -- `read_taxonomy_cell`/`kernel_pool_read`/`kernel_pool_write`
   as plain Python. Under OpenHands these aren't called directly -- the agent just
   reads the copied `taxonomy/`/`kernel_pool/` files in its workspace with `cat`/shell
   globbing. This module is for *us*: `eval/run_agent_eval.py` uses
   `kernel_pool_write` to persist a correct kernel back into the pool after a run, so
   the next workload's workspace includes it.

5. **`agent/harness.py`** (scaffolded, **CLI flags unverified**) -- `run_agent()`
   shells out to `openhands --headless --json -f task_prompt.md` inside a built
   workspace, capturing every event to `trajectory.jsonl`. The exact flags (workspace
   mounting, max-iterations, JSON event schema) are from OpenHands' docs, not a real
   run -- verify with `openhands --help` once it's installed where you'll actually run
   it (Kaggle/GCP), and fix up `harness.py` if anything differs.

6. **`eval/run_agent_eval.py`** (stub) -- loop `build_workspace()` + `run_agent()`
   over all JAXBench workloads, twice per workload (`use_taxonomy=True/False`), save
   both result sets. The speedup delta between them is the actual research result.

## What this project does NOT build

Kernels for JAXBench workloads themselves (attention, GEMM, RMSNorm, ...) -- those
are written by the coding agent at evaluation time, autonomously, using the workspace
above. We don't hand-write them. The only kernels we hand-write are the 10 small
generic ones inside `taxonomy/v5e/*/` (see [taxonomy/README.md](../taxonomy/README.md)),
which are illustrations of single techniques, not competitive implementations of any
benchmark task.

## Setup still needed before a real run

- `pip install openhands` in the environment that will actually run agent turns
  (Kaggle notebook or GCP VM -- not required for taxonomy-writing or plumbing work,
  which stay on CPU/interpret mode)
- Docker available there (OpenHands' sandboxed runtime needs it)
- An LLM API key configured for whichever model drives the agent
- Verify `agent/harness.py`'s CLI flags against `openhands --help`
