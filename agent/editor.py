"""Ask an LLM to patch a candidate kernel, grounded in a retrieved taxonomy cell.

The taxonomy-grounded prompt includes the cell's protocol.md + expert.py as few-shot
context. eval/baselines/ calls this with retrieved_cell=None to reproduce the
un-grounded baseline (docs/RAG only) for comparison.
"""

from pathlib import Path

from agent.retriever import TaxonomyCell


def propose_patch(kernel_path: Path, retrieved_cell: TaxonomyCell | None) -> str:
    """Return a patched kernel source string. retrieved_cell=None => ungrounded baseline."""
    raise NotImplementedError
