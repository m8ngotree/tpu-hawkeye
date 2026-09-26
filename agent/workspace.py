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

KERNEL_STUB = '''"""Write the kernel here.

Define workload(*inputs) with the same inputs and output as baseline.py. The main computation
must be a Pallas kernel (pl.pallas_call).
"""


def workload(*inputs):
    raise NotImplementedError
'''

TASK_PROMPT_TEMPLATE = Template(
    """\
# Task: Write an optimized $problem_type kernel for TPU $generation

Score = TFLOPS / baseline_tflops (higher is better). Must pass correctness first.

## Workspace
- `baseline.py` -- the problem specification: `create_inputs()` builds the inputs and
  `workload(*inputs)` is the math. Your kernel must produce the same output for the same inputs.
  It is a plain-JAX reference, not a kernel to copy: it exists to define the problem and to
  give the timing you are compared against.
- `kernel.py` -- write this from scratch. It starts as an empty stub. It must define
  `workload(*inputs)` taking the same inputs and returning the same output as `baseline.py`,
  and the main computation must be a Pallas kernel (`pl.pallas_call`). Plain JAX rewrites do
  not count and are rejected.
- `eval.py` -- the evaluation script; you run it with the `run_eval` tool, which evaluates
  `kernel.py` on the TPU. It returns a JSON result and, for a correct kernel, a `diagnosis`
  block: achieved HBM bandwidth and MXU utilization against the chip's peaks, the workload's
  arithmetic intensity, whether it is memory- or compute-bound, and how close the kernel is to
  that limit (`pct_of_roofline_limit`). `run_eval` is the only way to execute code. The fastest
  correct Pallas kernel you have evaluated is remembered, and that one is what gets scored, so a
  failed experiment does not lose earlier progress.
$taxonomy_section$kernel_pool_section
## How kernels are checked and timed
- Correctness: your output must match `baseline.py`'s output for the same inputs, elementwise,
  within `atol = rtol = 1e-2` (outputs are compared in float32; shapes and the number of outputs
  must match).
- Timing: median device time over 50 timed iterations after 5 warmup iterations, measured with
  the device profiler; speedup = baseline time / kernel time.

## Workflow
You can list, read and write files in this directory. You cannot run other commands.

1. Read `baseline.py` to understand the problem, then write a first correct Pallas kernel in
   `kernel.py` and evaluate it.
2. Use the `diagnosis` block to identify what is limiting throughput, make ONE targeted change
   to `kernel.py`, re-evaluate. Keep the change only if it stays correct and gets faster. Repeat.
3. Stop when you're out of ideas or turns, whichever comes first.
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


def _vendor_jaxbench(out_dir: Path, workload_name: str) -> None:
    """Give the workspace a private copy of the JAXBench harness and ONLY this workload's
    baseline, so eval.py needs nothing outside the workspace and the workspace contains no
    reference to the repo, other workloads, or hand-tuned optimized.py kernels."""
    src = JAXBENCH_BENCHMARKS.parent
    dst = out_dir / "JAXBench"
    (dst / "harness").mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "__init__.py", dst / "__init__.py")
    for f in (src / "harness").glob("*.py"):
        shutil.copy(f, dst / "harness" / f.name)
    bench = dst / "benchmark"
    (bench / workload_name).mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "benchmark" / "__init__.py", bench / "__init__.py")
    shutil.copy(src / "benchmark" / workload_name / "baseline.py", bench / workload_name / "baseline.py")


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

    shutil.copy(EVAL_PY, out_dir / "eval.py")
    _vendor_jaxbench(out_dir, workload_name)

    shutil.copy(JAXBENCH_BENCHMARKS / workload_name / "baseline.py", out_dir / "baseline.py")

    (out_dir / "kernel.py").write_text(starting_kernel if starting_kernel is not None else KERNEL_STUB)

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
