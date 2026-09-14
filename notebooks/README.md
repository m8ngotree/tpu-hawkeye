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

## Kaggle setup (first cells of any notebook)

The notebook itself should stay thin -- it clones and installs the repo, then shells
out to scripts that already exist here. Don't reimplement logic in notebook cells.

```python
# Cell 1 -- clone + install (every session, nothing persists)
!git clone --recurse-submodules https://github.com/m8ngotree/tpu-hawkeye.git
%cd tpu-hawkeye
!pip install -e .

# Cell 2 -- verify JAX actually sees the TPU before anything else. If this doesn't
# show 8 devices, that's the first thing to debug -- not our code.
import jax
print(jax.devices())

# Cell 3 -- plumbing check, now on real hardware instead of CPU
!python eval/smoke_test.py

# Cell 4 -- evaluate a kernel for real (drop --interpret to use the TPU)
!python eval/eval.py --workload 12p_RMSNorm --kernel eval/smoke_kernels/rmsnorm_naive.py --tpu v5e

# Cell 5 -- LLM API key via Kaggle Secrets (Add-ons > Secrets), never hardcoded
from kaggle_secrets import UserSecretsClient
import os
os.environ["LLM_API_KEY"] = UserSecretsClient().get_secret("LLM_API_KEY")
```

Not yet verified: whether Kaggle's TPU image needs anything beyond what's in
`pyproject.toml` for `jax.devices()` to see the TPU. Cell 2 is the check.
