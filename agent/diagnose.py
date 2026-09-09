"""Classify a kernel's bottleneck from its ProfileMetrics via roofline analysis.

v5e peak specs (used as the roofline ridge point): ~197 TFLOPS bf16, 819 GB/s HBM
bandwidth. Ridge point = peak_tflops / peak_bandwidth_gbps; a kernel's arithmetic
intensity below the ridge point is memory-bound, above it is compute-bound.
"""

from enum import Enum

from agent.profiler import ProfileMetrics

V5E_PEAK_BF16_TFLOPS = 197.0
V5E_PEAK_BANDWIDTH_GBPS = 819.0
V5E_RIDGE_POINT = V5E_PEAK_BF16_TFLOPS / V5E_PEAK_BANDWIDTH_GBPS  # FLOPs/byte


class Bottleneck(Enum):
    MXU_BOUND = "mxu_bound"
    DMA_BOUND = "dma_bound"
    VMEM_SPILL = "vmem_spill"
    SCALAR_FALLBACK = "scalar_fallback"


def diagnose(metrics: ProfileMetrics, arithmetic_intensity: float) -> Bottleneck:
    """Map profiler metrics + roofline position to a single dominant bottleneck label.

    This is intentionally a single label, not a ranked list -- the retriever looks up
    exactly one taxonomy cell per iteration, matching Hawkeye's "one optimization at a
    time" loop rather than trying to fix everything in one patch.
    """
    raise NotImplementedError
