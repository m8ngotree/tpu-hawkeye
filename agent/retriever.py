"""Map (bottleneck, kernel category) -> the matching taxonomy/patterns/ cell."""

from dataclasses import dataclass
from pathlib import Path

from agent.diagnose import Bottleneck

TAXONOMY_ROOT = Path(__file__).parent.parent / "taxonomy" / "patterns"

# Which pattern row to reach for when diagnose() reports a given bottleneck.
# Filled in as taxonomy/patterns/<pattern>/ cells are actually written.
BOTTLENECK_TO_PATTERN: dict[Bottleneck, str] = {
    Bottleneck.DMA_BOUND: "memory_pipelining",
    Bottleneck.MXU_BOUND: "mxu_tiling",
    Bottleneck.VMEM_SPILL: "grid_blockspec_design",
    Bottleneck.SCALAR_FALLBACK: "vpu_vectorization",
}


@dataclass
class TaxonomyCell:
    pattern: str
    category: str
    naive_path: Path
    expert_path: Path
    protocol: str


def retrieve(bottleneck: Bottleneck, kernel_category: str) -> TaxonomyCell:
    """Load the taxonomy cell for this bottleneck/category, or raise if it's not
    filled in yet."""
    raise NotImplementedError
