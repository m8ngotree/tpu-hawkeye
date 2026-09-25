# Agent

**Correction from an earlier version of this README:** this is not a scripted
profile -> classify-bottleneck -> retrieve-cell -> patch pipeline. That was modeling
the wrong thing. In Hawkeye, an LLM coding agent (given tools + a short system
prompt) does the profiling, the bottleneck diagnosis, and the taxonomy lookup itself,
as reasoning steps in its own tool-use loop (paper Section 2.4, Appendix E.1, Figure
47). This codebase's job is to give it the right tools and stay out of the way -- not
to pre-decide what it should conclude.

## Backend: hand-rolled tool-use loop (not OpenHands)

An earlier version of this project picked OpenHands (matching the paper's own
harness, Appendix G.1). Reconsidered: OpenHands' default sandbox needs Docker, and
its headless-mode docs had real gaps (workspace dir, max-iterations, JSON event
schema) when checked. Given the goal here -- taxonomy vs. no-taxonomy, same harness
held constant across both -- doesn't require matching the paper's exact framework,
just holding *our* harness constant, a small hand-rolled loop is a better fit: no
Docker dependency, no framework internals to debug, fully readable in one sitting.

**Tradeoff to know about:** `agent/tools_exec.py`'s `run_bash` is not sandboxed
beyond staying inside the workspace directory (no seccomp, no container, no resource
limits) -- see that module's docstring. This relies on the *outer* environment (a
disposable cloud VM being the actual safety boundary. Don't
run this against a machine you care about.

**Model-agnostic via any OpenAI-compatible chat-completions API** -- DeepSeek's API
is OpenAI-SDK compatible (`base_url="https://api.deepseek.com"`,
`model="deepseek-chat"`), so does OpenAI itself, and most local servers (vLLM,
Ollama). Comparing multiple backbone models later (as the paper does in Figure 2a's
Pass@1 sweep) is just re-running `run_agent()` with a different `model`/`base_url`.

## Pieces

1. **`agent/runner.py`** (done, verified) -- `run_kernel()`: wraps JAXBench's
   `evaluate_kernel` (compile, correctness, benchmark). Exposed to the agent as
   **`eval/eval.py`**, a CLI wrapper (`python eval.py --workload X --kernel kernel.py`)
   -- one of the commands the agent can invoke via its `run_bash` tool.

2. **`agent/workspace.py`** (done) -- `build_workspace()` builds the directory the
   agent operates in for one run, matching Hawkeye's Figure 22 tree: `kernel.py`
   (the agent edits this), `eval.py`, `task_prompt.md` (the task, modeled on Fig. 48's
   skeleton), and conditionally `taxonomy/` + `kernel_pool/`. `use_taxonomy=False`
   drops the `taxonomy/` directory entirely -- that's the baseline condition (Table 4).

3. **`agent/tools_exec.py`** (done, tested) -- the three tools actually exposed to
   the LLM: `read_file`, `write_file`, `run_bash`, all clamped to the workspace root.

4. **`agent/harness.py`** (done, untested against a real API call) -- `run_agent()`:
   the tool-use loop itself. Sends the system prompt + `task_prompt.md`, calls the
   model with the three tool schemas, executes whichever tool it calls via
   `tools_exec.py`, feeds the result back, repeats until the model stops calling
   tools or `max_turns` is hit. Logs every turn to `trajectory.jsonl` (mirrors
   Hawkeye's own trajectory log, Appendix G.4). Needs `LLM_API_KEY` set in the
   environment before running.

5. **`agent/profiler.py`** (stub) -- deeper counters (DMA-wait fraction, VMEM
   occupancy) beyond what `eval.py` already reports (TFLOPS, median_ms,
   utilization_pct). Needed once a taxonomy cell's `config.json` names a counter
   `evaluate_kernel` doesn't already surface. Would become a CLI script the agent can
   run via `run_bash`, same pattern as `eval.py`.

6. **`agent/tools.py`** -- `read_taxonomy_cell`/`kernel_pool_read`/`kernel_pool_write`
   as plain Python, for *us* (not the agent -- it reads the copied
   `taxonomy/`/`kernel_pool/` files directly via its own `read_file` tool).
   `eval/run_agent_eval.py` uses `kernel_pool_write` to persist a correct kernel back
   into the pool after a run, so the next workload's workspace includes it.

7. **`eval/run_agent_eval.py`** (stub) -- loop `build_workspace()` + `run_agent()`
   over all JAXBench workloads, twice per workload (`use_taxonomy=True/False`), save
   both result sets. The speedup delta between them is the actual research result.

## What this project does NOT build

Kernels for JAXBench workloads themselves (attention, GEMM, RMSNorm, ...) -- those
are written by the coding agent at evaluation time, autonomously, using the workspace
above. We don't hand-write them. The only kernels we hand-write are the 8 small
generic ones inside `taxonomy/v5e/*/` (see [taxonomy/README.md](../taxonomy/README.md)),
which are illustrations of single techniques, not competitive implementations of any
benchmark task.

## Setup still needed before a real run

- `pip install openai` (already in `pyproject.toml`) -- used purely as an
  OpenAI-compatible HTTP client, works against DeepSeek/OpenAI/local servers alike
- `LLM_API_KEY` set in the environment for whichever provider you point `run_agent()` at
- `agent/harness.py` has not been exercised against a real API call yet -- worth a
  cheap one-turn smoke test before trusting it for a full run
