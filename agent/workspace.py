"""Build a per-run workspace directory for the agent (agent/harness.py) -- matches
Hawkeye's workspace tree (paper Figure 22, page 23) and MSWEA task-prompt skeleton
(Figure 48, page 45), adapted for JAXBench/Pallas instead of CUDA.

The agent's run_bash/read_file/write_file tools (agent/tools_exec.py) operate purely
on files inside this directory, so everything it needs -- the prompt, the eval
script, the taxonomy cells, the kernel pool -- has to exist as real files here.
"""

import json
import shutil
from pathlib import Path
from string import Template

REPO_ROOT = Path(__file__).parent.parent
TAXONOMY_ROOT = REPO_ROOT / "taxonomy"
EVAL_PY = REPO_ROOT / "eval" / "eval.py"
JAXBENCH_BENCHMARKS = REPO_ROOT / "external" / "accelerator-agents" / "JAXBench" / "benchmark"
KERNEL_POOL_ROOT = REPO_ROOT / "results" / "kernel_pool"

TASK_PROMPT_TEMPLATE = Template(
    """\
# Task: Write an optimized $problem_type kernel for TPU $generation

Score = TFLOPS / baseline_tflops (higher is better). Must pass correctness first.

## Workspace
- `baseline.py` -- the reference implementation: `create_inputs()` builds the inputs and
  `workload(*inputs)` is the math. Your kernel must produce the same output for the same inputs.
- `kernel.py` -- edit this. Must define `workload(*inputs)` taking the same inputs (plain JAX or Pallas).
- `eval.py` -- run `python eval.py --workload $workload_name --kernel kernel.py --tpu $generation`
  (add `--interpret` to check correctness on CPU with no TPU hours spent; drop it for
  real timing on TPU). Prints a JSON result and exits 0 iff correct.
$taxonomy_section$kernel_pool_section
## Workflow
1. Get a *correct* kernel first (interpret mode is enough for this -- no TPU needed).
2. Profile on real TPU hardware (drop --interpret) and read the reported TFLOPS /
   utilization_pct.
3. Identify what's limiting throughput, make ONE targeted change, re-evaluate. Keep
   the change only if it doesn't regress correctness or speed. Repeat.
4. Stop when you're out of ideas or turns, whichever comes first.
"""
)

TAXONOMY_SECTION_TEMPLATE = Template(
    """\
- `taxonomy/` -- $n_cells reference cell$plural, one per TPU optimization technique. Each
  has `naive_kernel.py` (unoptimized baseline for that technique), `optimized_kernel.py`
  (the hand-written expert example -- read this for syntax), `config.json` (which
  profiler counter proves the technique fired), `guide.md` (protocol notes). These are
  NOT for this workload specifically -- they demonstrate one technique in isolation;
  you compose the relevant ones yourself.
"""
)

KERNEL_POOL_SECTION_TEMPLATE = Template(
    """\
- `kernel_pool/` -- kernels you (or earlier runs) already got correct for OTHER
  workloads on this TPU generation. Worth checking for a composition you can reuse.
"""
)


def build_workspace(
    workload_name: str,
    problem_type: str,
    out_dir: Path,
    generation: str = "v5e",
    use_taxonomy: bool = True,
    use_kernel_pool: bool = True,
    starting_kernel: str | None = None,
) -> Path:
    """Create out_dir/ populated for one agent run and return it.

    use_taxonomy=False is the baseline condition (Table 4 in the paper): same
    workspace, same eval.py, just no taxonomy/ directory -- isolates what the
    taxonomy itself buys.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    eval_src = EVAL_PY.read_text().replace(
        "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))",
        f"sys.path.insert(0, {str(REPO_ROOT)!r})",
    )
    (out_dir / "eval.py").write_text(eval_src)

    shutil.copy(JAXBENCH_BENCHMARKS / workload_name / "baseline.py", out_dir / "baseline.py")

    (out_dir / "kernel.py").write_text(
        starting_kernel or "def workload(*inputs):\n    raise NotImplementedError\n"
    )

    taxonomy_section = ""
    if use_taxonomy:
        gen_dir = TAXONOMY_ROOT / generation
        cell_names = sorted(p.name for p in gen_dir.iterdir() if p.is_dir()) if gen_dir.exists() else []
        if cell_names:
            shutil.copytree(gen_dir, out_dir / "taxonomy", dirs_exist_ok=True)
            taxonomy_section = TAXONOMY_SECTION_TEMPLATE.substitute(
                n_cells=len(cell_names), plural="" if len(cell_names) == 1 else "s"
            )

    kernel_pool_section = ""
    if use_kernel_pool:
        pool_dir = KERNEL_POOL_ROOT / generation
        if pool_dir.exists() and any(pool_dir.iterdir()):
            shutil.copytree(pool_dir, out_dir / "kernel_pool", dirs_exist_ok=True)
            kernel_pool_section = KERNEL_POOL_SECTION_TEMPLATE.substitute()

    prompt = TASK_PROMPT_TEMPLATE.substitute(
        problem_type=problem_type,
        generation=generation,
        workload_name=workload_name,
        taxonomy_section=taxonomy_section,
        kernel_pool_section=kernel_pool_section,
    )
    (out_dir / "task_prompt.md").write_text(prompt)

    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "workload_name": workload_name,
                "generation": generation,
                "use_taxonomy": use_taxonomy,
                "use_kernel_pool": use_kernel_pool,
            },
            indent=2,
        )
    )

    return out_dir
