# Agent

**Correction from an earlier version of this README:** this is not a scripted
profile -> classify-bottleneck -> retrieve-cell -> patch pipeline. That was modeling
the wrong thing. In Hawkeye, an LLM coding agent (given tools + a short system
prompt) does the profiling, the bottleneck diagnosis, and the taxonomy lookup itself,
as reasoning steps in its own tool-use loop (paper Section 2.4, Appendix E.1, Figure
47). This codebase's job is to give it the right tools and stay out of the way -- not
to pre-decide what it should conclude.

## What this project builds

1. **The tools** ([tools.py](tools.py), [runner.py](runner.py), [profiler.py](profiler.py)):
   - `evaluate_kernel` (runner.py, done) -- compile, check correctness, benchmark
     against a JAXBench workload
   - `profile_kernel` (profiler.py, stub) -- deeper counters a taxonomy cell's
     `config.json` names (DMA-wait fraction, VMEM occupancy, ...)
   - `read_taxonomy_cell` / `list_taxonomy_cells` (tools.py) -- read one row's
     naive/optimized kernel + config + guide for a TPU generation
   - `kernel_pool_read` / `kernel_pool_write` (tools.py) -- reuse kernels the agent
     already got correct for other workloads on the same generation

2. **The harness** (`harness.py`, not yet built) -- wraps the tools above as callable
   tools for an LLM API, sends the system prompt (see `system_prompt.md` below), and
   drives the turn loop for one (workload, taxonomy-condition) run, logging every
   tool call/result like Hawkeye's `trajectory.jsonl` (Appendix G.4).

   **Not yet decided: which LLM/agent SDK drives this.** Needs your call before
   building it -- see the chat for options.

3. **The eval sweep** ([eval/run_agent_eval.py](../eval/run_agent_eval.py), stub) --
   runs the harness over all JAXBench workloads twice: once with `taxonomy/`
   available (the condition under test), once with it swapped out (the "base LLM"
   baseline -- same tools minus `read_taxonomy_cell`/`list_taxonomy_cells`, same turn
   budget, matching Hawkeye's controlled baseline comparison in Table 4). The
   speedup delta between the two runs is the actual research result.

## What this project does NOT build

Kernels for JAXBench workloads themselves (attention, GEMM, RMSNorm, ...) -- those
are written by the coding agent at evaluation time, autonomously, using the tools
above. We don't hand-write them. The only kernels we hand-write are the 10 small
generic ones inside `taxonomy/v5e/*/` (see [taxonomy/README.md](../taxonomy/README.md)),
which are illustrations of single techniques, not competitive implementations of any
benchmark task.
