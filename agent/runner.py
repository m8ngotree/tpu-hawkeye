"""Compile, run, and numerically validate a candidate Pallas kernel.

Thin wrapper around JAXBench's own harness (external/accelerator-agents/JAXBench/
harness/evaluator.py) -- JAXBench already does compile -> correctness -> benchmark,
so this module's job is just: point it at a workload + kernel file, and translate its
dict result into something the rest of agent/ can work with (see agent/diagnose.py,
agent/loop.py).
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# JAXBench is imported as the `JAXBench` package -- its own internal imports
# (e.g. `from JAXBench.harness.loader import ...`) expect the *parent* of the
# JAXBench/ directory to be on sys.path, not JAXBench/ itself.
_JAXBENCH_PKG_PARENT = Path(__file__).parent.parent / "external" / "accelerator-agents"
if str(_JAXBENCH_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_JAXBENCH_PKG_PARENT))


@dataclass
class RunResult:
    status: str  # 'correct' | 'incorrect' | 'compile_error' | 'runtime_error' | 'error'
    correct: bool
    max_diff: float | None
    baseline_median_ms: float | None
    kernel_median_ms: float | None
    speedup_vs_baseline: float | None
    speedup_vs_pallas: float | None
    error: str | None
    raw: dict[str, Any]


def run_kernel(
    workload_name: str,
    kernel_path: Path,
    tpu: str = "v5e",
    interpret: bool = True,
    num_warmup: int = 2,
    num_iters: int = 5,
) -> RunResult:
    """Evaluate a candidate kernel file (must export workload(*inputs)) against a
    JAXBench workload.

    interpret=True sets the PALLAS_INTERPRET=1 env var. Kernel files that call
    pl.pallas_call should read it (`os.environ.get("PALLAS_INTERPRET") == "1"`) and
    pass it through as `pl.pallas_call(..., interpret=...)`, which runs the kernel in
    Pallas's CPU interpreter instead of lowering to real TPU code -- lets you validate
    correctness with zero TPU hours. Kernels with no pallas_call (plain jnp ops, like
    every current JAXBench baseline.py) ignore the flag entirely.

    tpu='v5e' only sets which peak-TFLOPS number JAXBench normalizes against when
    reporting utilization_pct -- it does not need to match the real backend. On a CPU
    those utilization/TFLOPS numbers are meaningless; only status/correct/timing
    matter for a plumbing smoke test. Pass tpu='v5e' on a real v5e too (JAXBench's 'auto'
    detection does not recognize the 'TPU v5 lite' device kind).

    num_warmup/num_iters default low (2/5) because this function is meant to also work
    on CPU / interpret mode where each iteration is much slower than on real TPU;
    override upward for real benchmark runs.
    """
    os.environ["PALLAS_INTERPRET"] = "1" if interpret else "0"

    from JAXBench.harness.evaluator import evaluate_kernel

    result = evaluate_kernel(
        workload_name=workload_name,
        kernel_path=str(kernel_path),
        tpu=tpu,
        num_warmup=num_warmup,
        num_iters=num_iters,
    )

    correctness = result.get("correctness", {})
    baseline = result.get("baseline") or {}
    kernel = result.get("kernel") or {}

    return RunResult(
        status=result["status"],
        correct=correctness.get("correct", False),
        max_diff=correctness.get("max_diff"),
        baseline_median_ms=baseline.get("median_ms"),
        kernel_median_ms=kernel.get("median_ms"),
        speedup_vs_baseline=result.get("speedup_vs_baseline"),
        speedup_vs_pallas=result.get("speedup_vs_pallas"),
        error=result.get("error"),
        raw=result,
    )
