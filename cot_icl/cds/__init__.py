"""Curvilinear Demonstration Selection (CDS)."""

from cot_icl.cds.tsp import get_best_order, get_optimal_order

__all__ = ["get_embeddings", "get_best_order", "get_optimal_order"]


def __getattr__(name: str):
    if name == "get_embeddings":
        from cot_icl.cds.embeddings import get_embeddings

        return get_embeddings
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
