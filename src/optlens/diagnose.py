"""Solver-neutral diagnostics built on any Backend: feasibility relaxation and IIS fallback/validation."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, replace

import numpy as np
import scipy.sparse as sp

from .backends import IIS, Backend, IISNotSupported, SolveResult, parallel_solves
from .model import INF, ModelData

# Directions match Gurobi feasRelax artificial-variable semantics (ArtP/ArtN/ArtL/ArtU).
_DIRECTION = {
    "ArtP": ("constraint", "RHS must be decreased by"),
    "ArtN": ("constraint", "RHS must be increased by"),
    "ArtL": ("variable", "lower bound must be decreased"),
    "ArtU": ("variable", "upper bound must be increased"),
}


@dataclass
class RelaxResult:
    status: str
    total_violation: float | None
    relaxations: list[dict]          # [{type, name, direction, value}]
    solution: SolveResult
    objective: float | None = None   # with min_objective: the best original objective at the minimal total change
    best_status: str | None = None   # with min_objective: the best-plan solve's status ("skipped" when not run)


def base_name(name: str) -> str:
    """Constraint/variable family: the name without its index, for ``x[1,2]``, ``price(1day_0_0)``
    and ``min_imp_3`` / ``exclusive_4_1_3`` naming styles."""
    head = re.split(r"[\[(]", name, maxsplit=1)[0]
    return re.sub(r"(_\d+)+$", "", head) or head


def elastic_model(md: ModelData, relax_rows: bool = True, relax_bounds: bool = False,
                  penalty: float = 1.0, only: set[str] | None = None) -> ModelData:
    """L1 feasibility relaxation (Gurobi feasRelaxS(relaxobjtype=0, minrelax=False)).

    Each relaxable row gets ArtP (+1) and ArtN (-1) elastic columns; each finite variable bound gets
    ArtL/ArtU via a new row ``lb <= x + ArtL - ArtU <= ub`` and the original bounds are freed.
    The objective becomes the total penalized violation. Integrality is kept.
    ``only`` restricts relaxation to rows/columns whose name or base name is in the set
    (Gurobi feasRelax with infinite penalty elsewhere).
    """
    m, n = md.num_rows, md.num_cols
    base = replace(md, minimize=True, obj=np.zeros(n), obj_offset=0.0, Q=None)
    names: list[str] = []
    cols: list[sp.csr_matrix] = []

    def selected(name: str) -> bool:
        return only is None or name in only or base_name(name) in only

    if relax_rows:
        rows = [i for i, r in enumerate(md.row_names) if selected(r)]
        eye = sp.identity(m, format="csr")[:, rows]
        cols += [eye, -eye]
        names += [f"ArtP_{md.row_names[i]}" for i in rows] + [f"ArtN_{md.row_names[i]}" for i in rows]
    if names:
        A_new = sp.hstack(cols).tocsr()
        k = len(names)
        base = base.add_cols(names, np.full(k, penalty), np.zeros(k), np.full(k, INF), A_new)

    if relax_bounds:
        bounded = [j for j in range(n) if (np.isfinite(md.col_lb[j]) or np.isfinite(md.col_ub[j]))
                   and selected(md.col_names[j])]
        k = len(bounded)
        if k:
            art_names = [f"ArtL_{md.col_names[j]}" for j in bounded] + [f"ArtU_{md.col_names[j]}" for j in bounded]
            base = base.add_cols(art_names, np.full(2 * k, penalty), np.zeros(2 * k), np.full(2 * k, INF),
                                 sp.csr_matrix((base.num_rows, 2 * k)))
            rows = sp.lil_matrix((k, base.num_cols))
            first_art = base.num_cols - 2 * k
            for r, j in enumerate(bounded):
                rows[r, j], rows[r, first_art + r], rows[r, first_art + k + r] = 1.0, 1.0, -1.0
            col_lb, col_ub = base.col_lb.copy(), base.col_ub.copy()
            col_lb[bounded], col_ub[bounded] = -INF, INF
            base = replace(
                base, A=sp.vstack([base.A, rows.tocsr()]).tocsr(),
                row_lo=np.concatenate([base.row_lo, md.col_lb[bounded]]),
                row_hi=np.concatenate([base.row_hi, md.col_ub[bounded]]),
                row_names=base.row_names + tuple(f"bnd_{md.col_names[j]}" for j in bounded),
                col_lb=col_lb, col_ub=col_ub,
            )
    return base


def feas_relax(md: ModelData, backend: Backend, relax_rows: bool = True, relax_bounds: bool = False,
               penalty: float = 1.0, time_limit: float = 60.0, tol: float = 1e-9,
               only: set[str] | None = None, min_objective: bool = False) -> RelaxResult:
    """The smallest total (penalized) change that makes the model feasible. With ``min_objective`` (Gurobi's
    minrelax=True) a second solve keeps the total at that minimum and optimizes the original objective, so the
    plan and the split of the change among the relaxed rows are the best ones for the business; skipped when the
    first solve stopped at its time limit (its total is then not proven minimal). Both solves share ``time_limit``."""
    em = elastic_model(md, relax_rows, relax_bounds, penalty, only)
    t0 = time.time()
    res = backend.solve(em, time_limit)
    if not res.feasible:
        return RelaxResult(res.status, None, [], res)
    objective, phase1_total = None, res.obj
    took = time.time() - t0
    # both phases share one budget (callers size it to their own timeouts); the best-plan solve is a refinement, so it
    # gets at most three times the first solve (at least 10 s) rather than whatever is left
    left = min(time_limit - took, max(10.0, 3 * took))
    best_status = "skipped" if min_objective else None
    if min_objective and res.status == "OPTIMAL" and left >= 1:
        k = em.num_cols - md.num_cols
        cap = float(res.obj)
        q = None if md.Q is None else sp.bmat([[md.Q, None], [None, sp.csc_matrix((k, k))]], format="csc")
        best = replace(em, obj=np.concatenate([md.obj, np.zeros(k)]), obj_offset=md.obj_offset,
                       minimize=md.minimize, Q=q).add_row(
            "__total_change", {em.col_names[j]: float(em.obj[j]) for j in range(md.num_cols, em.num_cols)}, hi=cap)
        res2 = backend.solve(best, left)
        best_status = res2.status
        if res2.status == "OPTIMAL":
            res, objective = res2, float(res2.obj)
    relaxations = []
    for j in range(md.num_cols, em.num_cols):
        if res.x[j] > tol:
            prefix, _, target = em.col_names[j].partition("_")
            kind, direction = _DIRECTION[prefix]
            relaxations.append({"type": kind, "name": target, "direction": direction, "value": float(res.x[j])})
    if objective is not None and relaxations:
        # phase 2 chose the split; within solver tolerance it can overshoot the minimum, so scale back to it
        over = penalty * sum(r["value"] for r in relaxations) / float(phase1_total) if phase1_total else 1.0
        if over > 1:
            relaxations = [{**r, "value": r["value"] / over} for r in relaxations]
    relaxations = _snap_integral(md, relaxations)
    total = float(sum(r["value"] for r in relaxations)) if relaxations else (0.0 if objective is not None else float(res.obj))
    return RelaxResult(res.status, total, relaxations, res, objective, best_status)


def _snap_integral(md: ModelData, relaxations: list[dict], tol: float = 1e-4) -> list[dict]:
    """Round relaxation amounts up to whole units where the quantity can only move in whole units.

    A row whose variables are all integer with integer coefficients and an integer limit has an integer activity,
    so any change to its limit that helps is a whole number; the same holds for an integer variable's bound. A
    solver reports such an amount minus its tolerance (0.999999 from HiGHS, 0.99999 from Gurobi on neos859080,
    where 0.99999 applied to the model left it infeasible), so snap to the next whole unit: ceil(value - tol)."""
    out = []
    csr = md.A.tocsr()
    for r in relaxations:
        v = r["value"]
        if r["type"] == "constraint":
            i = md.row_index(r["name"])
            row = csr.getrow(i)
            lim = md.row_hi[i] if "increased" in r["direction"] else md.row_lo[i]
            integral = (row.nnz > 0 and bool(np.all(md.is_int[row.indices]))
                        and bool(np.all(np.abs(row.data - np.round(row.data)) < 1e-9))
                        and np.isfinite(lim) and abs(lim - round(lim)) < 1e-9)
        else:
            j = md.col_index(r["name"])
            b = md.col_lb[j] if r["direction"].startswith("lower") else md.col_ub[j]
            integral = bool(md.is_int[j]) and np.isfinite(b) and abs(b - round(b)) < 1e-9
        out.append({**r, "value": float(max(1.0, np.ceil(v - tol)))} if integral and v > tol else r)
    return out


def apply_relaxation(md: ModelData, relaxations: list[dict], margin: float = 1e-6) -> ModelData:
    """Apply a relaxation's suggested changes to the original model (for verifying a repair).

    Each change is widened by ``margin`` (absolute, plus the same relative to the value): applied
    exactly, a repair sits on the feasibility boundary and ill-conditioned models can still be
    declared infeasible within solver tolerances.
    """
    out = md
    for r in relaxations:
        i_or_j, v = r["name"], r["value"] + margin * (1 + abs(r["value"]))
        if r["type"] == "constraint":
            i = out.row_index(i_or_j)
            if r["direction"].endswith("decreased by"):   # ArtP: activity was below row_lo
                out = out.set_row_bounds(i_or_j, lo=out.row_lo[i] - v)
            else:                                           # ArtN: activity was above row_hi
                out = out.set_row_bounds(i_or_j, hi=out.row_hi[i] + v)
        else:
            j = out.col_index(i_or_j)
            if r["direction"].startswith("lower"):
                out = out.set_col_bounds(i_or_j, lb=out.col_lb[j] - v)
            else:
                out = out.set_col_bounds(i_or_j, ub=out.col_ub[j] + v)
    return out


def relaxed_bounds(md: ModelData, relaxations: list[dict]) -> list[dict]:
    """The relaxation as current and new bound values, exact: the exact minimum applied at full precision
    solves (HiGHS, SCIP and Gurobi alike), while a rounded one can land on the infeasible side (mining:
    26,091,666.67 printed as 2.60917e+07 = 26,091,700 was INFEASIBLE)."""
    out = []
    for r in relaxations:
        if r["type"] == "constraint":
            i = md.row_index(r["name"])
            lower = r["direction"].endswith("decreased by")   # ArtP: activity was below row_lo
            cur = md.row_lo[i] if lower else md.row_hi[i]
        else:
            j = md.col_index(r["name"])
            lower = r["direction"].startswith("lower")
            cur = md.col_lb[j] if lower else md.col_ub[j]
        out.append({**r, "side": "lower" if lower else "upper", "current": float(cur),
                    "new": float(cur - r["value"] if lower else cur + r["value"])})
    return out


def _feasibility_version(md: ModelData) -> ModelData:
    """Same constraints, zero objective: the first feasible solution is optimal, so the solver stops there."""
    return replace(md, obj=np.zeros(md.num_cols), obj_offset=0.0, Q=None)


def _infeasible(md: ModelData, backend: Backend, time_limit: float) -> bool:
    return backend.solve(_feasibility_version(md), time_limit).status in ("INFEASIBLE", "INF_OR_UNBD")


def default_backend(md: ModelData) -> Backend:
    """Open-source default: HiGHS for LP, SCIP for MIP (HiGHS has no MIP IIS and is slower on elastic MIPs)."""
    from .backends import BACKENDS

    return BACKENDS["scip"] if md.is_mip else BACKENDS["highs"]


def deletion_filter_iis(md: ModelData, backend: Backend, time_limit: float = 60.0,
                        max_rows: int = 2000, time_budget: float = 120.0) -> IIS:
    """Row-IIS by deletion filtering (Chinneck): works with any backend, LP or MIP. Bounds are kept fixed.

    Starts from the rows with nonzero L1 elastic slack plus rows sharing columns with them, which is
    usually far smaller than the model; falls back to all rows if that subset is feasible.
    A trial that times out keeps its row, and running past ``time_budget`` stops filtering; either way
    the result is still infeasible but only "reduced", not proven irreducible.
    """
    start = time.monotonic()
    candidates = list(range(md.num_rows))
    relax = feas_relax(md, backend, time_limit=time_limit)
    if relax.relaxations:
        seed = {md.row_index(r["name"]) for r in relax.relaxations if r["type"] == "constraint"}
        cols = set(md.A[sorted(seed), :].indices)
        near = set(md.A.tocsc()[:, sorted(cols)].tocoo().row)
        subset = sorted(seed | near)
        if _infeasible(md.keep_rows(subset), backend, time_limit):
            candidates = subset
    if len(candidates) > max_rows:
        raise IISNotSupported(f"deletion filter over {len(candidates)} rows exceeds max_rows={max_rows}")
    keep = list(candidates)
    conclusive = True
    for i in list(candidates):
        if time.monotonic() - start > time_budget:
            conclusive = False
            break
        trial = [k for k in keep if k != i]
        status = backend.solve(_feasibility_version(md.keep_rows(trial)), time_limit).status
        if status in ("INFEASIBLE", "INF_OR_UNBD"):
            keep = trial
        elif status != "OPTIMAL":
            conclusive = False
    label = "irreducible" if conclusive else "reduced"
    return IIS(rows=[md.row_names[i] for i in keep], method=f"deletion_filter:{backend.name}:{label}")


MAX_GROUPS = 60  # more groups than this and the group filter costs as much as a row-level IIS
LOCALIZE_MIN_ROWS = 20_000  # above this, an IIS search starts from the rows the elastic relaxation stretches
LOCALIZE_MAX_ROWS = 50_000  # a neighbourhood grown past this is no longer local: give up and search the whole model
DENSE_SHARE = 0.05  # a row touching this share of all variables (a group-wide budget) links everything: never grown into


def localized_iis(md: ModelData, time_limit: float = 120.0, hops: int = 3) -> IIS | None:
    """An IIS found near the conflict instead of in the whole model, for models too large for a whole-model search
    (HiGHS's whole-model IIS did not finish in 25 minutes on a 279k-row MIP). The seed is the rows of HiGHS's infeasibility proof
    for the LP relaxation (its dual ray), or else the rows its L1 elastic relaxation stretches; the rows sharing a
    variable with them are added a hop at a time until those rows alone
    are infeasible, and HiGHS computes the IIS of that piece. Any infeasible subset of the rows is a conflict of the
    whole model, so the result is a true IIS (of the LP relaxation, as ``lp_relaxation_iis``). None when the LP
    relaxation is feasible (the conflict needs integrality), nothing stretches, or the piece grows too large. Rows over
    DENSE_SHARE of the variables are not grown into, so a model linked by a group-wide row stays local."""
    from .backends import BACKENDS

    start = time.monotonic()
    highs = BACKENDS["highs"]
    lp = replace(md, is_int=np.zeros(md.num_cols, dtype=bool))
    # The seed: the rows of HiGHS's infeasibility proof (one LP solve), else the rows the elastic relaxation
    # stretches (slower: 172 s on the linked hotel model at 279k rows against 5 s for the proof).
    seed = set(highs.farkas_rows(lp, time_limit) or [])
    if not seed:
        relax = feas_relax(lp, highs, time_limit=max(1.0, time_limit - (time.monotonic() - start)))
        seed = {md.row_index(r["name"]) for r in relax.relaxations if r["type"] == "constraint"}
    if not seed:
        return None
    A, At = md.A.tocsr(), md.A.tocsc()
    # A dense linking row enters the piece only if the relaxation stretched it; reached through a shared variable it
    # would pull in the whole model after one hop (a group-wide budget over every hotel and night).
    dense = set(np.flatnonzero(np.diff(A.indptr) > max(1000, DENSE_SHARE * md.num_cols)).tolist()) - seed
    rows = set(seed)
    for _ in range(hops + 1):
        left = time_limit - (time.monotonic() - start)
        if left <= 1 or len(rows) > LOCALIZE_MAX_ROWS:
            return None
        piece = lp.keep_rows(sorted(rows))
        if _infeasible(piece, highs, left):
            try:
                iis = highs.iis(piece, max(1.0, time_limit - (time.monotonic() - start)))
            except IISNotSupported:
                return None
            return replace(iis, method=f"localized({len(rows)} of {md.num_rows} rows):{iis.method}")
        cols = np.unique(A[sorted(rows), :].indices)
        rows |= set(np.unique(At[:, cols].indices).tolist()) - dense
    return None


def lp_relaxation_iis(md: ModelData, time_limit: float = 300.0, backend: Backend | None = None) -> IIS | None:
    """Gurobi's IISMethod 3: when a MIP's LP relaxation is already infeasible, the conflict holds without
    integrality, and HiGHS computes the relaxation's IIS directly (one dual computation, proven minimal for the
    relaxation) instead of the MIP search's many MIP solves. It may not be minimal once integrality is added
    back. A licensed ``backend`` computes it instead of HiGHS. None when the relaxation is feasible or the solver
    does not finish within ``time_limit``."""
    from .backends import BACKENDS, IISNotSupported

    bk = backend if backend is not None and backend.licensed else BACKENDS["highs"]
    lp = replace(md, is_int=np.zeros(md.num_cols, dtype=bool))
    if not _infeasible(lp, bk, min(time_limit, 60.0)):
        return None
    try:
        iis = bk.iis(lp, time_limit)
    except IISNotSupported:
        return None
    return replace(iis, method=f"lp_relaxation:{iis.method}")


def family_first_iis(md: ModelData, backend: Backend, time_limit: float = 60.0,
                     budget: float = 300.0) -> tuple[list[str], IIS]:
    """Deletion filter over groups of rows (whole groups dropped at once), then a row-level IIS on the
    rows of the surviving groups only; one solve per group, so it scales to MIPs where a row-level IIS
    of the full model does not finish. Groups are name families when names are meaningful, otherwise
    structural classes (same row shape); with too many groups the filter is skipped. Past ``budget``
    seconds the remaining groups are kept as they are (still infeasible, possibly not minimal).
    A MIP whose LP relaxation is already infeasible gets the relaxation's IIS instead (``lp_relaxation_iis``)."""
    from .structure import names_are_meaningful, row_classes

    start = time.monotonic()
    # localized_iis is built on HiGHS's infeasibility proof: a licensed backend searches the whole model itself
    if not backend.licensed and md.num_rows >= LOCALIZE_MIN_ROWS \
            and (loc := localized_iis(md, min(budget, 120.0))) is not None:
        return sorted({base_name(r) for r in loc.rows}), loc
    if md.is_mip and (lp := lp_relaxation_iis(md, budget, backend)) is not None:
        return sorted({base_name(r) for r in lp.rows}), lp
    if names_are_meaningful(md):
        groups: dict[str, list[int]] = {}
        for i, n in enumerate(md.row_names):
            groups.setdefault(base_name(n), []).append(i)
        kind = "family"
    else:
        groups = row_classes(md)
        covered = {i for idx in groups.values() for i in idx}
        for i in range(md.num_rows):  # rows with a unique shape form their own group
            if i not in covered:
                groups[md.row_names[i]] = [i]
        kind = "shape"
    if len(groups) > MAX_GROUPS:
        iis = get_iis(md, backend, time_limit)
        return [], replace(iis, method=f"no_group_filter({len(groups)} {kind} groups):{iis.method}")
    keep = sorted(groups)
    for g in sorted(groups, key=lambda f: -len(groups[f])):
        if time.monotonic() - start > budget:
            break
        trial = [f for f in keep if f != g]
        rows = [i for f in trial for i in groups[f]]
        if rows and _infeasible(md.keep_rows(rows), backend, time_limit):
            keep = trial
    sub = md.keep_rows([i for f in keep for i in groups[f]])
    left = max(10.0, budget - (time.monotonic() - start))  # the row-level stage gets what the group filter left
    iis = get_iis(sub, backend, min(time_limit, left), time_budget=left)
    return keep, replace(iis, method=f"{kind}_first:{iis.method}")


def get_iis(md: ModelData, backend: Backend, time_limit: float = 60.0, time_budget: float = 120.0) -> IIS:
    """IIS routing: the backend's native IIS; for a MIP without one, SCIP's native MIP IIS (open-source backends
    only); the deletion filter on the requested backend only as a last resort, within ``time_budget`` seconds.
    A native IIS whose rows are feasible on their own is rejected: gurobipy 13.0.3 leaves out a one-variable row on
    a binary whose fractional limit rounds it to 0 (tests/test_licensed_solver.py)."""
    from .backends import BACKENDS

    natives = [backend]
    if md.is_mip and not backend.licensed and backend.name != "scip":
        natives.append(BACKENDS["scip"])
    rejected = []
    for b in natives:
        try:
            iis = b.iis(md, time_limit)
        except IISNotSupported:
            continue
        base = _iis_bounds(md, iis)
        rows = base.keep_rows([base.row_index(r) for r in iis.rows])
        if b.solve(_feasibility_version(rows), time_limit).status != "OPTIMAL":
            return iis
        rejected.append(b.name)
    iis = deletion_filter_iis(md, backend, time_limit, time_budget=time_budget)
    if rejected:
        iis = replace(iis, note=f"{' and '.join(rejected)}'s own IIS was rejected (its rows are feasible on their own); "
                                "this one was found by removing rows one at a time")
    return iis


def _iis_bounds(md: ModelData, iis: IIS) -> ModelData:
    """The model with the bounds an IIS was computed against. When the solver reported which bounds are in the IIS,
    bounds of other columns are freed (integrality kept); otherwise all original bounds stay."""
    if not iis.bounds_reported:
        return md
    # Integer columns keep their bounds: a binary's [0, 1] is part of its type, and Gurobi
    # doesn't list it as an IIS bound.
    in_iis = md.is_int.copy()
    in_iis[[md.col_index(c) for c in iis.col_bounds]] = True
    return replace(md, col_lb=np.where(in_iis, md.col_lb, -INF), col_ub=np.where(in_iis, md.col_ub, INF))


def check_iis(md: ModelData, iis: IIS, backend: Backend, check_irreducible: bool = True,
              time_limit: float = 60.0) -> dict:
    """Verify an IIS: its rows are infeasible and, optionally, row-irreducible, against the bounds it was computed
    against (``_iis_bounds``)."""
    base = _iis_bounds(md, iis)
    idx = [base.row_index(r) for r in iis.rows]
    infeasible = _infeasible(base.keep_rows(idx), backend, time_limit)
    irreducible = None
    if check_irreducible and infeasible:
        irreducible = all(not _infeasible(base.keep_rows([k for k in idx if k != i]), backend, time_limit)
                          for i in idx)
    return {"infeasible": infeasible, "irreducible": irreducible, "size": len(idx)}


def linking_families(md: ModelData) -> set[str]:
    """Constraint families whose every limit is 0: balance and linking rows that define one quantity from others,
    not business limits a planner sets."""
    rows_of: dict[str, list[int]] = {}
    for i, n in enumerate(md.row_names):
        rows_of.setdefault(base_name(n), []).append(i)
    out = set()
    for fam, idx in rows_of.items():
        bounds = np.concatenate([md.row_lo[idx], md.row_hi[idx]])
        if np.all(np.abs(bounds[np.isfinite(bounds)]) <= 1e-9):
            out.add(fam)
    return out


def fix_menu(md: ModelData, backend: Backend, families: list[str] | None = None,
             time_limit: float = 60.0, min_objective: bool = False) -> list[dict]:
    """One relaxation per constraint family, only that family allowed to change (run in parallel), plus
    the unrestricted minimum for reference. A menu of levers rather than one blended fix; "not sufficient
    alone" is a real outcome. Families default to those in the IIS (family-first for MIPs). Families
    whose rows all have RHS 0 are flagged as likely linking/definitional rather than business levers."""
    from concurrent.futures import ThreadPoolExecutor

    if families is None:
        if md.is_mip:
            families, _ = family_first_iis(md, backend, time_limit)
        else:
            families = sorted({base_name(r) for r in get_iis(md, backend, time_limit).rows})
    linking = linking_families(md)

    def one(only):
        r = feas_relax(md, backend, time_limit=time_limit, only=only, min_objective=min_objective)
        return {"sufficient": r.total_violation is not None, "status": r.status, "objective": r.objective,
                "total_change": r.total_violation, "changes": relaxed_bounds(md, r.relaxations)}

    with ThreadPoolExecutor(max_workers=parallel_solves(md, min(8, len(families) + 1))) as pool:
        per_family = list(pool.map(lambda f: one({f}), families))
        overall = one(None)
    menu = [{"family": f, "structural": f in linking, **res} for f, res in zip(families, per_family)]
    # business levers first: a linking row's "minimal change" is rarely something a planner can decide
    menu.sort(key=lambda e: (not e["sufficient"], e["structural"],
                             e["total_change"] if e["total_change"] is not None else np.inf))
    return menu + [{"family": "(all families together: unrestricted minimum)", "structural": False, **overall}]






def fix_conflicts(base: ModelData, fixed: ModelData, names: list[str], backend: Backend, shrink: float = 1e-3,
                  max_changes: int = 3, budget: float = 45.0, time_limit: float = 30.0) -> dict:
    """What a verified fix clears: the conflict that sets each changed amount, and whether the amount has slack.

    A model can hold several overlapping conflicts, and an IIS shows only one of them, often not the one a fix
    has to clear. ``base`` is the infeasible model, ``fixed`` the feasible one with the changes to rows or
    variables in ``names``. For each change, keep the rest of the fix and move that one change back by
    ``shrink`` of its size (bounds = base + (1 - shrink) * (fixed - base)): if the model is infeasible again,
    an IIS there is the conflict that sets that amount; if it stays feasible, the amount has slack. Works for
    any bound change (a raised limit, a moved pin); dropped rows are not covered. Stops after ``budget`` s.
    Returns {"conflicts": [{"changes": names, "rows": IIS rows}] per distinct conflict (by family set),
    "slack": names, "unchecked": names}."""
    start = time.monotonic()
    out: dict[tuple, dict] = {}
    slack, unchecked = [], []
    for k, name in enumerate(names):
        if k >= max_changes or time.monotonic() - start > budget:
            unchecked.append(name)
            continue
        cut = fixed
        if name in fixed.row_names:
            i, j = base.row_index(name), fixed.row_index(name)
            cut = cut.set_row_bounds(name, lo=_toward(base.row_lo[i], fixed.row_lo[j], shrink),
                                     hi=_toward(base.row_hi[i], fixed.row_hi[j], shrink))
        else:
            i, j = base.col_index(name), fixed.col_index(name)
            cut = cut.set_col_bounds(name, lb=_toward(base.col_lb[i], fixed.col_lb[j], shrink),
                                     ub=_toward(base.col_ub[i], fixed.col_ub[j], shrink))
        if not _infeasible(cut, backend, time_limit):
            slack.append(name)
            continue
        left = max(10.0, budget - (time.monotonic() - start))
        if cut.is_mip:
            rows = family_first_iis(cut, backend, min(time_limit, left), budget=left)[1].rows
        else:
            rows = get_iis(cut, backend, min(time_limit, left)).rows
        key = tuple(sorted({base_name(r) for r in rows}))
        out.setdefault(key, {"changes": [], "rows": rows})["changes"].append(name)
    return {"conflicts": list(out.values()), "slack": slack, "unchecked": unchecked}


def _toward(old: float, new: float, shrink: float) -> float:
    """``new`` moved back toward ``old`` by ``shrink`` of the change (infinite or unchanged ends stay as they are)."""
    if not (np.isfinite(old) and np.isfinite(new)) or old == new:
        return new
    return new - shrink * (new - old)
