# Kaggle notebooks

Kaggle TPU v5e-8 sessions are ephemeral (no persistent disk between sessions beyond
Kaggle Datasets/outputs), so each notebook here should:

1. `pip install` this repo + `external/accelerator-agents/JAXBench` deps at the top
2. Pull taxonomy cells + agent code from this repo (clone at session start, or attach
   as a Kaggle Dataset if you want to avoid a git clone every session)
3. Write `results/*.json` to a Kaggle Dataset output (or push back to this repo) before
   the session ends -- don't let a run's output live only in the ephemeral session.

Nothing here yet; add one notebook per experiment (e.g. `01_correctness_smoke_test.ipynb`,
`02_taxonomy_cell_profiling.ipynb`, `03_agent_eval_run.ipynb`) rather than one big notebook.
