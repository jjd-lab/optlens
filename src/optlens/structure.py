"""Name-free row classes: rows with the same shape (sense, number of terms, coefficients) are siblings.

Anonymous models (ROW00087, C1234) carry no meaning in their names, so grouping by name only works by
accident; the shape of a row does not depend on its name.
"""
from __future__ import annotations

import numpy as np

from .model import ModelData


def _sense(md: ModelData, i: int) -> str:
    lo, hi = md.row_lo[i], md.row_hi[i]
    if lo == hi:
        return "="
    return "range" if np.isfinite(lo) and np.isfinite(hi) else "<=" if np.isfinite(hi) else ">="


def row_signatures(md: ModelData, digits: int = 6) -> list[tuple]:
    """Exact shape: sense, term count and the sorted, rounded coefficients."""
    A = md.A.tocsr()
    sigs = []
    for i in range(md.num_rows):
        coefs = np.sort(np.round(A.data[A.indptr[i]:A.indptr[i + 1]], digits))
        sigs.append((_sense(md, i), len(coefs), tuple(coefs.tolist())))
    return sigs


def row_classes(md: ModelData, min_size: int = 2) -> dict[str, list[int]]:
    """Structural classes with at least min_size rows, keyed by a short readable label."""
    classes: dict[tuple, list[int]] = {}
    for i, sig in enumerate(row_signatures(md)):
        classes.setdefault(sig, []).append(i)
    out = {}
    for k, (sig, idx) in enumerate(sorted(classes.items(), key=lambda kv: -len(kv[1]))):
        if len(idx) >= min_size:
            coefs = ", ".join(f"{c:g}" for c in sig[2][:4]) + (", ..." if len(sig[2]) > 4 else "")
            out[f"shape#{k} ({sig[0]}, {sig[1]} terms: {coefs})"] = idx
    return out


def names_are_meaningful(md: ModelData) -> bool:
    """Name families are usable when rows share them: at least 3 rows per family on average."""
    from .diagnose import base_name

    return md.num_rows > 0 and len({base_name(n) for n in md.row_names}) * 3 <= md.num_rows
