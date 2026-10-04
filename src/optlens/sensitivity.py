"""Sensitivity: shadow prices, reduced costs and ranging (HiGHS; Gurobi in a session on Gurobi).

For a MIP the integer variables are fixed at their optimal values and the remaining continuous
problem is analysed; its duals answer "what is one more unit worth, keeping these integer
decisions", not "would the integer decisions change".
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import highspy
import numpy as np

from .backends import parallel_solves
from .diagnose import default_backend
from .model import ModelData


@dataclass
class Sensitivity:
    status: str
    obj: float | None = None
    x: np.ndarray | None = None
    # d objective / d (active bound of the row), in the model's own sense (max or min)
    shadow_price: np.ndarray | None = None
    reduced_cost: np.ndarray | None = None
    # RHS range over which shadow_price stays valid; cost range over which x stays optimal (LP only)
    rhs_range: tuple[np.ndarray, np.ndarray] | None = None
    cost_range: tuple[np.ndarray, np.ndarray] | None = None
    integers_fixed: bool = False


def marginal_value(md: ModelData, row: str, delta: float, time_limit: float = 60.0, base=None, backend=None) -> dict:
    """Re-solve the full model (integers free to change) with the row's active bound moved by
    +delta and -delta. Unlike a shadow price this stays meaningful at degenerate points and for MIPs.
    An equality row moves both bounds; otherwise the side that is active at the optimum moves. ``backend``: the
    caller's solver (default: HiGHS for an LP, SCIP for a MIP); ``base``: that solver's result for ``md``, when the
    caller already has it."""
    backend = backend or default_backend(md)
    base = base or backend.solve(md, time_limit)
    if base.status != "OPTIMAL":
        return {"status": base.status}
    i = md.row_index(row)
    lo, hi = md.row_lo[i], md.row_hi[i]
    act = float((md.A.getrow(i) @ base.x)[0])
    upper_active = np.isfinite(hi) and (not np.isfinite(lo) or abs(act - hi) <= abs(act - lo))

    def shifted(d: float) -> ModelData:
        if lo == hi:
            return md.set_row_bounds(row, lo=lo + d, hi=hi + d)
        return md.set_row_bounds(row, hi=hi + d) if upper_active else md.set_row_bounds(row, lo=lo + d)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=parallel_solves(md, 2)) as pool:  # independent solves, each in its own process
        up, down = pool.map(lambda d: backend.solve(shifted(d), time_limit), (delta, -delta))
    per_unit = lambda r, sign: (r.obj - base.obj) / (sign * delta) if r.status == "OPTIMAL" else None  # noqa: E731
    side = "both (equality)" if lo == hi else ("upper" if upper_active else "lower")
    return {"status": "OPTIMAL", "side": side, "activity": act,
            "bound": hi if upper_active else lo, "objective": base.obj,
            "up_status": up.status, "down_status": down.status, "up_objective": up.obj, "down_objective": down.obj,
            "per_unit_up": per_unit(up, 1), "per_unit_down": per_unit(down, -1)}


@dataclass
class UnitMove:
    col: int
    step: int                 # +1 or -1
    improving: bool           # the move alone would improve the objective
    blocked_by: list[int]     # rows the move would violate (empty: the move is feasible for every row)


def integer_unit_moves(md: ModelData, x: np.ndarray, tol: float = 1e-6) -> list[UnitMove]:
    """Try +1 and -1 on every integer variable (within its bounds) and record which rows each move
    would violate. The integer analogue of "binding": a row with slack left can still pin the plan if
    its slack is smaller than one unit step of an integer variable in it (extends the
    effective-binding check). Single-variable moves only; a better plan needing a swap is not seen."""
    act = md.A @ x
    A = md.A.tocsc()
    # ModelData.Q is the Hessian (objective has 0.5 x'Qx), so the exact change of a unit step on x_j
    # is grad_j * step + 0.5 * Q_jj.
    grad = md.obj + (md.Q @ x if md.Q is not None else 0.0)
    curv = 0.5 * md.Q.diagonal() if md.Q is not None else np.zeros(md.num_cols)
    sense = -1.0 if md.minimize else 1.0
    moves = []
    for j in np.flatnonzero(md.is_int):
        rows, coefs = A.indices[A.indptr[j]:A.indptr[j + 1]], A.data[A.indptr[j]:A.indptr[j + 1]]
        for step in (1, -1):
            if not (md.col_lb[j] - tol <= x[j] + step <= md.col_ub[j] + tol):
                continue
            new = act[rows] + coefs * step
            bad = rows[(new > md.row_hi[rows] + tol) | (new < md.row_lo[rows] - tol)]
            moves.append(UnitMove(int(j), step, sense * (grad[j] * step + curv[j]) > tol, [int(i) for i in bad]))
    return moves


def integer_binding(md: ModelData, x: np.ndarray, moves: list[UnitMove] | None = None) -> dict[str, np.ndarray]:
    """Per-row counts from integer_unit_moves: effective_binding (blocks any unit move),
    blocks_improving (improving moves it violates) and sole_blocker (improving moves only it violates)."""
    moves = integer_unit_moves(md, x) if moves is None else moves
    eff, imp, sole = (np.zeros(md.num_rows, int) for _ in range(3))
    for mv in moves:
        eff[mv.blocked_by] += 1
        if mv.improving:
            imp[mv.blocked_by] += 1
            if len(mv.blocked_by) == 1:
                sole[mv.blocked_by[0]] += 1
    return {"effective_binding": eff > 0, "blocks_improving": imp, "sole_blocker": sole}


def fix_integers(md: ModelData, x: np.ndarray) -> ModelData:
    vals = np.round(x[md.is_int])
    lb, ub = md.col_lb.copy(), md.col_ub.copy()
    lb[md.is_int], ub[md.is_int] = vals, vals
    return replace(md, col_lb=lb, col_ub=ub, is_int=np.zeros(md.num_cols, bool))


def sensitivity(md: ModelData, time_limit: float = 60.0, backend=None) -> Sensitivity:
    """Duals and ranges at the optimum (for a MIP, of the continuous problem left with the integers fixed). HiGHS
    computes them, or a licensed ``backend`` (Gurobi) for every step. The ranges hold for the optimal basis: at a
    degenerate optimum the shadow price can differ on each side of the current limit (marginal_value re-solves)."""
    bk = backend if backend is not None and backend.licensed else default_backend(md)
    base = bk.solve(md, time_limit)
    if base.status != "OPTIMAL":
        return Sensitivity(base.status)
    cont = fix_integers(md, base.x) if md.is_mip else md
    if bk.licensed:
        r = bk.ranging(cont, time_limit)
        if r is None:
            return Sensitivity("fixed-integer not solved to optimality")
        return Sensitivity("OPTIMAL", obj=base.obj, x=r["x"], shadow_price=r["shadow_price"],
                           reduced_cost=r["reduced_cost"], rhs_range=r["rhs_range"], cost_range=r["cost_range"],
                           integers_fixed=md.is_mip)
    h = cont.to_highs()
    h.setOptionValue("time_limit", float(time_limit))
    h.run()
    if h.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        return Sensitivity(f"fixed-integer {h.modelStatusToString(h.getModelStatus())}")
    sol = h.getSolution()
    out = Sensitivity("OPTIMAL", obj=base.obj, x=np.array(sol.col_value),
                      shadow_price=np.array(sol.row_dual), reduced_cost=np.array(sol.col_dual),
                      integers_fixed=md.is_mip)
    if not cont.quadratic_objective:  # HiGHS ranging is LP-only
        status, rng = h.getRanging()
        if status == highspy.HighsStatus.kOk and rng.valid:
            lo, hi = np.array(rng.row_bound_dn.value_), np.array(rng.row_bound_up.value_)
            # HiGHS's row ranges are not limit ranges for a row with slack (basic): such a limit can move freely up to
            # the row's activity, as Gurobi's SARHSLow/SARHSUp report
            act = np.array(sol.row_value)
            for i in np.flatnonzero(np.array(h.getBasis().row_status) == highspy.HighsBasisStatus.kBasic):
                rlo, rhi = cont.row_lo[i], cont.row_hi[i]
                upper = np.isfinite(rhi) and (not np.isfinite(rlo) or abs(act[i] - rhi) <= abs(act[i] - rlo))
                lo[i], hi[i] = (act[i], np.inf) if upper else (-np.inf, act[i])
            out.rhs_range = (lo, hi)
            out.cost_range = (np.array(rng.col_cost_dn.value_), np.array(rng.col_cost_up.value_))
    return out
