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


def sensitivity(md: ModelData, time_limit: float = 60.0, backend=None, base=None) -> Sensitivity:
    """Duals and ranges at the optimum (for a MIP, of the continuous problem left with the integers fixed). HiGHS
    computes them, or a licensed ``backend`` (Gurobi) for every step. The ranges hold for the optimal basis: at a
    degenerate optimum the shadow price can differ on each side of the current limit (marginal_value re-solves).
    ``base``: an optimal solve of ``md`` the caller already has, used instead of solving it again (E66: at 279k rows
    the second MIP solve timed out right after the first had finished in 30 s)."""
    if not md.convex_objective():
        return Sensitivity("non-convex quadratic objective, which has no shadow prices; marginal_value re-solves")
    bk = backend if backend is not None and backend.licensed else default_backend(md)
    if base is None or base.status != "OPTIMAL" or base.x is None:
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


@dataclass
class OptimalityCertificate:
    """Why no plan beats the optimum, as a receipt: each binding row's shadow price times the limit it sits at, and
    each variable at a bound its reduced cost times that bound, summed with the objective's constant, give ``bound``:
    the most a maximum's plan can earn, the least a minimum's can cost. Checked by arithmetic (the duals satisfy
    c = A'y + d, every priced row and variable sits at the limit it is priced at, and the signs are those of an
    optimum), so no feasible plan passes it. For a MIP the sum comes from the LP relaxation (``relaxation``), so it
    also bounds every plan with whole numbers: when it equals the plan's objective (``exact``) it proves the plan
    optimal, else the plan is within ``gap`` of it (20 business MILPs: 7 exact, gaps from 0.19 % up). On the
    hotel week the relaxation's bound is the MILP optimum itself (the integers fixed instead leave nothing to price
    in a model whose every variable is an integer). Each family maps to (rows or variables, total)."""
    bound: float
    plan: float
    rows: dict[str, tuple[int, float]]
    bounds: dict[str, tuple[int, float]]
    offset: float
    relaxation: bool
    exact: bool
    gap: float | None = None


def optimality_certificate(md: ModelData, time_limit: float = 60.0, backend=None,
                           tol: float = 1e-6) -> OptimalityCertificate | None:
    """The optimum's dual certificate grouped by family (see OptimalityCertificate). None when the model (or, for a
    MIP, the model or its relaxation) is not solved to optimality, has a quadratic objective, or the duals do not
    check out."""
    from .diagnose import base_name

    if md.quadratic_objective:
        return None
    plan = None
    if md.is_mip:
        plan = (backend or default_backend(md)).solve(md, time_limit)
        if plan.status != "OPTIMAL":
            return None
        md = replace(md, is_int=np.zeros(md.num_cols, dtype=bool))
    s = sensitivity(md, time_limit, backend=backend)
    if s.status != "OPTIMAL" or s.shadow_price is None or s.reduced_cost is None:
        return None
    x, y, d = s.x, s.shadow_price, s.reduced_cost
    scale = 1.0 + float(np.abs(md.obj).max(initial=0.0))
    if np.abs(md.obj - md.A.T @ y - d).max(initial=0.0) > tol * scale:  # the duals price the objective exactly
        return None
    act = md.A @ x
    sign = 1.0 if md.minimize else -1.0  # a minimum's prices are >= 0 at lower limits; a maximum's the reverse

    def at(value, lo, hi, price):
        """The limit a priced row or variable sits at, if its price has an optimum's sign there; else None."""
        near = tol * (1.0 + abs(value))
        if np.isfinite(lo) and abs(value - lo) <= near and (sign * price >= -tol * scale or lo == hi):
            return lo
        if np.isfinite(hi) and abs(value - hi) <= near and (sign * price <= tol * scale or lo == hi):
            return hi
        return None

    groups = ({}, {})
    total, size = float(md.obj_offset), abs(float(md.obj_offset))  # size: what the sum's rounding scales with
    for k, (prices, values, lo, hi, names) in enumerate(((y, act, md.row_lo, md.row_hi, md.row_names),
                                                         (d, x, md.col_lb, md.col_ub, md.col_names))):
        for i in np.flatnonzero(np.abs(prices) > tol * scale):
            b = at(values[i], lo[i], hi[i], prices[i])
            if b is None:
                return None
            n, t = groups[k].get(base_name(names[i]), (0, 0.0))
            groups[k][base_name(names[i])] = (n + 1, t + float(prices[i] * b))
            total += float(prices[i] * b)
            size += abs(float(prices[i] * b))
    bound = float(md.obj @ x + md.obj_offset)
    if abs(total - bound) > 10 * tol * (1.0 + abs(bound) + size):  # big-M terms (1e9 on covid19) cancel
        return None
    value = bound if plan is None else float(plan.obj)
    exact = abs(bound - value) <= 1e-6 * (1.0 + abs(value))
    gap = None if exact else abs(bound - value) / max(1e-9, abs(value))
    return OptimalityCertificate(bound, value, groups[0], groups[1], float(md.obj_offset), plan is not None, exact,
                                 gap)
