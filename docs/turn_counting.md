# Turn counting

How an agent run is budgeted, exactly what does and does not use up the budget, and how
that compares with the Hawkeye paper. Code: `_productive_actions` and the loop in
`agent/harness.py`; command-line flags in `eval/run_agent_eval.py`.

## Definitions

- **Raw turn**: one model call (one request to the LLM API). A single raw turn may contain
  several tool calls, because the model can request more than one at a time (for example
  three `read_file` calls in its first message).
- **Productive turn**: a raw turn that contains at least one productive action (below).
  A turn with several productive actions still counts once.
- **Budget**: the run stops when it has used `--max-productive-turns` productive turns
  (default 50).
- **Raw cap**: a safety limit, `--max-raw-turns` (default 200), on model calls, so a run that
  spends its turns on free actions cannot go on forever.
- A run also stops when the model answers without any tool call (`done`).

`stopped_reason` in `result.json` records which limit ended the run:
`done`, `max_productive_turns`, or `max_raw_turns`.

## What counts as a productive action

| Action | Productive? | Notes |
|---|---|---|
| `write_file` to `kernel.py` | **yes** | A kernel edit. Writes to any other path do not count. |
| `run_eval` | **yes** | An evaluation. Counts whether it succeeds, fails to compile, is incorrect, or is rejected for not using Pallas. |
| `read_file` on a file under `taxonomy/` (`.py`, `.md`, `.json`) | **yes** | Every read counts, **repeats included**. |
| `read_file` on anything else (`baseline.py`, `eval.py`, the JAXBench copy, ...) | no | Free. |
| `list_files` (any directory, including `taxonomy/`) | no | Free. |
| A call to a tool that does not exist | no | Free. |
| A call that fails | as above | Productivity is decided from the call's name and arguments before it runs, so a `read_file` of a non-existent path under `taxonomy/`, or a `write_file` to `kernel.py` that errors, still counts. |
| A model reply with no tool call | no | It ends the run. |

There is no shell, so there are no other kinds of action.

### Consequences worth knowing

- The taxonomy condition **pays budget for consulting the taxonomy**; the no-taxonomy
  condition has nothing to read. This is intended and matches the paper: consulting a cell
  costs a productive turn.
- Reading several taxonomy files in the **same** raw turn costs one productive turn, not
  one per file. Reading them one per turn costs one each.
- Writing `kernel.py` and calling `run_eval` in the same raw turn costs one productive
  turn; in two consecutive turns, two.
- Free actions (listing, reading the problem, reading the harness) are limited only by the
  raw cap.

## What Hawkeye does (from the paper)

Sources: Section 3 ("Agent baselines"), the Figure 5 caption, Appendix D.3 and D.4
(trajectory descriptions), and Appendix G.1 (run configuration).

1. **Budget**: "All agents have a budget of 100 productive turns of kernel edits,
   evaluations, and taxonomy or deduplicated kernel-pool reads." Figure 5 words the same
   thing as "kernel edits, evaluations, and abstraction reads".
2. **Productive versus raw**: trajectories are reported with both counts, for example
   "328 raw turns, 100 productive" and "354 raw turns" for another run. Raw turns include
   everything; only productive turns use the budget.
3. **Reads count**: in the raw-documentation comparison, an agent that re-issued the same
   search "across 7 distinct turns ... each costing a productive turn". So repeated
   documentation reads are charged; the paper says only *kernel-pool* reads are
   deduplicated.
4. **Failed evaluations**: the paper does not say they are excluded; its table of
   inflection turns includes "debug turns where the kernel still failed". We count every
   evaluation.
5. **Tools**: one of the paper's two harnesses is an API agent with dedicated tools
   (`evaluate_kernel`, `read_header`, `read_validation_test`, `query_replay`) and no general
   shell; the other is a bash-only OpenHands agent. Our tool set follows the first.
6. **Other settings**: `max_turns = 100`, `max_tokens = 16,384` per reply, a 900 s
   evaluation timeout, and three to five runs per configuration.

## Comparison

| Aspect | Hawkeye | Ours | Same? |
|---|---|---|---|
| Budget unit | productive turns | productive turns | yes |
| Kernel edits count | yes | yes (`write_file` on `kernel.py`) | yes |
| Evaluations count, including failures | yes | yes | yes |
| Taxonomy / unit-test reads count | yes | yes (every `read_file` under `taxonomy/`) | yes |
| Repeated taxonomy reads | count | count | yes |
| Kernel-pool reads deduplicated | yes | not applicable (no kernel pool in use) | n/a |
| Other exploration | free | free | yes |
| Raw turns tracked separately | yes | yes (`turns_used`) | yes |
| Budget size | 100 | 50 by default (configurable) | **no, by choice** |
| Raw-turn safety cap | not stated | 200 | our addition |
| General shell available | OpenHands harness only; the API harness has none | none | matches the API harness |

### Uncertainties (things the paper does not settle)

- **What exactly is "one turn" in their API agent?** We take it to be one model response.
  If their harness counted each tool call separately, our budget is looser than theirs for
  the same nominal number.
- **Edit and evaluation together.** In an API harness with `evaluate_kernel(source)`, an
  edit and an evaluation may be a single call. Our `write_file` plus `run_eval` are two
  calls that share a productive turn only if issued in the same response. This makes our
  budget slightly tighter for agents that write and evaluate in separate turns.
- **The `max_turns = 100` line** in the paper's appendix may refer to raw turns for the
  OpenHands harness (one run is reported as hitting "MaxIterationsReached"). We cannot tell
  which counts it applies to, so we follow the main text (100 productive).
- **Which reads are "taxonomy reads" for them** (`read_validation_test` on a whole cell
  versus single files). We charge each file read.

## Choosing a budget

Fewer turns than the paper's 100 is acceptable for our purposes; the trade-off is that
taxonomy reads take a larger share of a smaller budget, so a 50-turn run leaves the
taxonomy agent fewer turns for kernel edits than a 100-turn run would. Raise
`--max-productive-turns` (and `--max-raw-turns`, roughly four times as large) to move toward
the paper's setting. Token cost grows faster than linearly with turns, because every turn
re-sends the whole conversation.

## Reading the numbers

Per run, `result.json` has `turns_used` (raw), `productive_turns`, `stopped_reason`, and
`eval_history`, in which each evaluation records the raw turn and the productive turns used
so far. `python scripts/analyze_run.py <run dir>` prints them. Each `tool` line in
`trajectory.jsonl` carries `turn_productive`, the turn-level flag (all tool calls in a turn
share it).
