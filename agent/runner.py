"""Compile, run, and numerically validate a candidate Pallas kernel.

Wraps JAXBench's harness (external/accelerator-agents/JAXBench/harness) rather than
reimplementing correctness checking / device-side timing.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class RunResult:
    correct: bool
    median_ms: float | None
    tflops: float | None
    error: str | None = None


def run_kernel(workload_name: str, kernel_path: Path, interpret: bool = False) -> RunResult:
    """Compile + execute `kernel_path`'s workload() against JAXBench's baseline inputs.

    interpret=True runs Pallas in interpreter mode (CPU, no TPU hours spent) -- use
    this for correctness iteration; only flip to False for profiling/benchmark runs.
    """
    raise NotImplementedError
