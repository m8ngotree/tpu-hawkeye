# Agent loop

Mirrors Hawkeye's closed loop, retargeted at Pallas/TPU:

```
loop.py
  1. runner.py     compile + run candidate kernel (interpret=True locally, real TPU for
                    validation/benchmark runs), check numerics against JAXBench baseline
  2. profiler.py   wrap the JAX/XLA profiler, parse the resulting Perfetto/XProf trace
  3. diagnose.py   trace -> bottleneck label (mxu_bound / dma_bound / vmem_spill /
                    scalar_fallback) via roofline + the raw utilization numbers
  4. retriever.py  (bottleneck, kernel category) -> matching taxonomy/patterns/ cell
  5. editor.py     LLM proposes a patch, grounded in the retrieved cell's protocol.md
                    + expert.py as few-shot context
  6. back to 1, until no improving edit is found or the iteration budget is spent
```

## Baseline to compare against

`eval/baselines/` runs the *same* JAXBench workloads through steps 1-3 + a plain LLM
patch proposal with no taxonomy retrieval (docs/RAG only, matching what Hawkeye calls
out as the weaker baseline). The whole point of this project is the speedup delta
between this and the taxonomy-grounded loop above.

## Not yet implemented

Every module here is a stub. Fill in `runner.py` first (it's the only one that needs
a live TPU to test), then `profiler.py`/`diagnose.py` against a couple of manually-run
traces before wiring up `retriever.py`/`editor.py`.
