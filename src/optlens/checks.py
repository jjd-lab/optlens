"""Rationality checks for a proposed repair: feasible is not enough, the change must also make sense.

Generic checks (a domain pack adds its own): hard rules untouched, change size within tolerance,
objective within bound, integer variables keep their meaning, no constraint silently dropped.
Thresholds live in a Policy, stated by the user (or derived from an answer key in an evaluation).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .backends import SolveResult
from .diagnose import base_name
from .model import ModelData


@dataclass
class Policy:
    hard: set[str] = field(default_factory=set)   # row/column names or families that must not change
    max_rel_change: float = 0.5                   # per changed bound: |new - old| / max(1, |old|)
    max_obj_rel_change: float | None = 0.5        # vs the reference objective, when one is known
    allow_drop: bool = False
    droppable: set[str] = field(default_factory=set)  # rows that may be dropped even when allow_drop is False


def _rel(new: float, old: float) -> float:
    if new == old:
        return 0.0
    if not (np.isfinite(new) and np.isfinite(old)):
        return np.inf
    return abs(new - old) / max(1.0, abs(old))


def diff(original: ModelData, repaired: ModelData) -> dict:
    """Changes by name: rows dropped/added, row and column bound changes, objective and type changes."""
    orow = {n: i for i, n in enumerate(original.row_names)}
    rrow = {n: i for i, n in enumerate(repaired.row_names)}
    ocol = {n: j for j, n in enumerate(original.col_names)}
    rcol = {n: j for j, n in enumerate(repaired.col_names)}
    changes = {"dropped_rows": sorted(set(orow) - set(rrow)), "added_rows": sorted(set(rrow) - set(orow)),
               "row_bounds": [], "col_bounds": [], "objective": [], "type": []}
    for n in set(orow) & set(rrow):
        i, k = orow[n], rrow[n]
        for side, old, new in (("lower", original.row_lo[i], repaired.row_lo[k]),
                               ("upper", original.row_hi[i], repaired.row_hi[k])):
            if old != new:
                changes["row_bounds"].append({"name": n, "side": side, "old": float(old), "new": float(new),
                                              "rel": _rel(new, old)})
    for n in set(ocol) & set(rcol):
        j, k = ocol[n], rcol[n]
        for side, old, new in (("lower", original.col_lb[j], repaired.col_lb[k]),
                               ("upper", original.col_ub[j], repaired.col_ub[k])):
            if old != new:
                changes["col_bounds"].append({"name": n, "side": side, "old": float(old), "new": float(new),
                                              "rel": _rel(new, old)})
        if original.obj[j] != repaired.obj[k]:
            changes["objective"].append({"name": n, "old": float(original.obj[j]), "new": float(repaired.obj[k])})
        if original.is_int[j] != repaired.is_int[k]:
            changes["type"].append({"name": n, "was_integer": bool(original.is_int[j])})
    return changes


def check_repair(original: ModelData, repaired: ModelData, result: SolveResult,
                 policy: Policy | None = None, reference_objective: float | None = None) -> list[dict]:
    policy = policy or Policy()
    d = diff(original, repaired)
    touched = [c["name"] for c in d["row_bounds"] + d["col_bounds"] + d["objective"] + d["type"]] + d["dropped_rows"]
    hard_hit = sorted({n for n in touched if n in policy.hard or base_name(n) in policy.hard})
    bounds = d["row_bounds"] + d["col_bounds"]
    worst = max((c["rel"] for c in bounds), default=0.0)
    checks = [
        {"check": "solves", "passed": result.status == "OPTIMAL", "detail": result.status},
        {"check": "hard_rules_untouched", "passed": not hard_hit, "detail": hard_hit},
        {"check": "change_within_tolerance", "passed": worst <= policy.max_rel_change,
         "detail": f"largest relative bound change {worst:.3g} (limit {policy.max_rel_change:.3g})"},
        {"check": "no_unapproved_drops",
         "passed": policy.allow_drop or not [r for r in d["dropped_rows"] if r not in policy.droppable],
         "detail": d["dropped_rows"]},
    ]
    bad_int = [c["name"] for c in d["type"] if c["was_integer"]]
    col = {n: j for j, n in enumerate(repaired.col_names)}
    for c in d["col_bounds"]:
        j = col[c["name"]]
        if repaired.is_int[j] and np.isfinite(c["new"]) and abs(c["new"] - round(c["new"])) > 1e-9:
            bad_int.append(c["name"])
    checks.append({"check": "integers_keep_meaning", "passed": not bad_int, "detail": sorted(set(bad_int))})
    if policy.max_obj_rel_change is not None and reference_objective is not None and result.obj is not None:
        rel = abs(result.obj - reference_objective) / max(1.0, abs(reference_objective))
        checks.append({"check": "objective_within_bound", "passed": rel <= policy.max_obj_rel_change,
                       "detail": f"relative objective change {rel:.3g} (limit {policy.max_obj_rel_change:.3g})"})
    return checks
