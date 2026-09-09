"""Capture and parse a JAX/XLA profiler trace for a kernel run.

Produces the raw numbers diagnose.py needs: MXU utilization, VMEM occupancy/spills,
DMA wait time, achieved vs. peak bandwidth -- pulled from the Perfetto/XProf trace
JAX's profiler emits (jax.profiler.trace(...)).
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
    """Run under jax.profiler.trace() and parse the resulting trace into ProfileMetrics."""
    raise NotImplementedError
