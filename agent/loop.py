"""Orchestrates runner -> profiler -> diagnose -> retriever -> editor, iterating until
no improving patch is found or the iteration budget runs out.
"""

from dataclasses import dataclass, field
from pathlib import Path

from agent.diagnose import diagnose
from agent.editor import propose_patch
from agent.profiler import profile_kernel
from agent.retriever import retrieve
from agent.runner import run_kernel


@dataclass
class IterationLog:
    iteration: int
    median_ms: float
    bottleneck: str
    patch_applied: bool


@dataclass
class OptimizeResult:
    workload_name: str
    final_kernel_path: Path
    history: list[IterationLog] = field(default_factory=list)


def optimize(
    workload_name: str,
    starting_kernel_path: Path,
    max_iterations: int = 10,
    use_taxonomy: bool = True,
) -> OptimizeResult:
    """Run the closed loop. use_taxonomy=False reproduces the ungrounded baseline."""
    raise NotImplementedError
