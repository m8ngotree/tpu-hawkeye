"""The tool surface handed to the coding agent -- matches Hawkeye's Figure 47 system
prompt: evaluate_kernel, read_header (here: read_taxonomy_cell), query_replay (here:
kernel_pool_read/write).

These are plain Python functions; agent/harness.py wraps them as tool definitions for
whichever LLM API drives the agent (see agent/README.md -- backend not chosen yet).
"""

import json
from dataclasses import dataclass
from pathlib import Path

TAXONOMY_ROOT = Path(__file__).parent.parent / "taxonomy"
KERNEL_POOL_ROOT = Path(__file__).parent.parent / "results" / "kernel_pool"


@dataclass
class TaxonomyCell:
    row_name: str
    naive_kernel: str
    optimized_kernel: str
    config: dict
    guide: str


def list_taxonomy_cells(generation: str = "v5e") -> list[str]:
    """Row names available for this TPU generation, e.g. ['01_mxu_feed', ...]."""
    gen_dir = TAXONOMY_ROOT / generation
    if not gen_dir.exists():
        return []
    return sorted(p.name for p in gen_dir.iterdir() if p.is_dir())


def read_taxonomy_cell(row_name: str, generation: str = "v5e") -> TaxonomyCell:
    """Load one cell's four artifacts. Raises FileNotFoundError if the cell hasn't
    been authored yet (see taxonomy/README.md -- writing these is the current gap)."""
    cell_dir = TAXONOMY_ROOT / generation / row_name
    return TaxonomyCell(
        row_name=row_name,
        naive_kernel=(cell_dir / "naive_kernel.py").read_text(),
        optimized_kernel=(cell_dir / "optimized_kernel.py").read_text(),
        config=json.loads((cell_dir / "config.json").read_text()),
        guide=(cell_dir / "guide.md").read_text(),
    )


def kernel_pool_read(workload_name: str, generation: str = "v5e") -> str | None:
    """A previously-accepted correct kernel for this workload on this generation, if
    one exists in the pool -- lets the agent reuse compositions it already found for
    other workloads (Hawkeye Appendix E.3.1: cross-workload reuse)."""
    path = KERNEL_POOL_ROOT / generation / f"{workload_name}.py"
    return path.read_text() if path.exists() else None


def kernel_pool_write(workload_name: str, kernel_source: str, generation: str = "v5e") -> None:
    """Save a kernel that passed evaluate_kernel's correctness check into the pool."""
    path = KERNEL_POOL_ROOT / generation / f"{workload_name}.py"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(kernel_source)
