"""The optimal plan nearest a reference plan: separates the changes a model edit forces from solver tie-breaking.

On a model with many equal-cost plans, a re-solve picks any of them, so a plain before/after diff mostly shows the
solver's choice among ties (over 99 % of the reported changes on an ad-allocation min-cost what-if). One more solve
finds the plan that keeps the new optimum (objective within OBJ_TOL) and is nearest the old plan in total absolute
distance; whatever it still changes, the edit forced."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import scipy.sparse as sp

from optlens.backends import Backend
from optlens.model import INF, ModelData

OBJ_TOL = 1e-6  # relative; the cutoff must not cut off the optimum it was read from


def moved(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Which entries differ beyond the tolerance compare_versions uses."""
    return np.abs(b - a) > 1e-6 * np.maximum(1.0, np.abs(a))


def distance_model(md: ModelData, x0: np.ndarray, obj: float) -> ModelData:
    """md plus a distance column d_j >= |x_j - x0_j| per variable and a cutoff row holding the objective at obj;
    minimizes the sum of d. Linear objectives only."""
    m, n = md.num_rows, md.num_cols
    eye, zero = sp.identity(n, format="csr"), sp.csr_matrix((m, n))
    tol = OBJ_TOL * max(1.0, abs(obj))
    cut = obj - md.obj_offset
    A = sp.bmat([[md.A, zero], [eye, -eye], [eye, eye], [sp.csr_matrix(md.obj), sp.csr_matrix((1, n))]], format="csr")
    return replace(
        md, minimize=True, obj=np.concatenate([np.zeros(n), np.ones(n)]), obj_offset=0.0, A=A, Q=None,
        row_lo=np.concatenate([md.row_lo, np.full(n, -INF), x0, [-INF if md.minimize else cut - tol]]),
        row_hi=np.concatenate([md.row_hi, x0, np.full(n, INF), [cut + tol if md.minimize else INF]]),
        col_lb=np.concatenate([md.col_lb, np.zeros(n)]), col_ub=np.concatenate([md.col_ub, np.full(n, INF)]),
        is_int=np.concatenate([md.is_int, np.zeros(n, bool)]),
        col_names=md.col_names + tuple(f"d__{c}" for c in md.col_names),
        row_names=md.row_names + tuple(f"lo__{c}" for c in md.col_names) + tuple(f"hi__{c}" for c in md.col_names)
        + ("objective_cutoff",),
    )


def closest_optimal(md: ModelData, x0: np.ndarray, obj: float, backend: Backend,
                    time_limit: float) -> tuple[str, np.ndarray | None]:
    """(status, plan): md's optimal plan nearest x0. A search stopped by its time limit returns its best plan, which
    is optimal for md but maybe not the nearest; (status, None) when there is none or md is quadratic."""
    if md.Q is not None:
        return "QUADRATIC", None
    r = backend.solve(distance_model(md, x0, obj), time_limit)
    return r.status, (r.x[:md.num_cols] if r.x is not None else None)
