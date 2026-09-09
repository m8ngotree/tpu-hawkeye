"""Tool: capture deeper profiler counters than evaluate_kernel's TFLOPS/utilization
summary -- the counters taxonomy cells' config.json name (e.g. dma_wait_fraction,
vmem_occupancy).

This is exposed to the coding agent as a callable tool, same role as `ncu` in
Hawkeye's GPU harness (Appendix G.4) -- it returns raw numbers. The agent decides
what they mean and what to do about it; nothing here classifies a "bottleneck" for
it (see agent/README.md for why that's the agent's job, not this codebase's).
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProfileMetrics:
    mxu_utilization: float
    vmem_occupancy: float
    dma_wait_fraction: float
    achieved_bandwidth_gbps: float
    achieved_tflops: float


def profile_kernel(workload_name: str, kernel_path: Path) -> ProfileMetrics:
    """Run the kernel under jax.profiler.trace(), parse the Perfetto trace, return
    the raw counters. Not yet implemented -- needed before any taxonomy cell whose
    config.json targets a counter beyond plain TFLOPS/median_ms (which
    agent/runner.py's evaluate_kernel wrapper already reports)."""
    raise NotImplementedError
