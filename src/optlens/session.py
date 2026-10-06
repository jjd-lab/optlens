"""The engine's tool layer: a Session holds a model and its edited versions, and each method is one diagnostic
operation returning text for an agent (IIS, feasibility relaxation, repair menu, what-if, why-not, sensitivity,
version comparison). TOOLS holds the matching JSON schemas; an MCP server or an agent's own code can call the same
methods (optlens.workspace, the plugin's run_python).

Every solve goes through optlens backends, which enforce a hard time limit in a worker process.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np

import optlens as od
from optlens import closest, sensitivity
from optlens.backends import parallel_solves
from optlens.diagnose import base_name
from optlens.outliers import family_outliers, flag_fix
from optlens.structure import names_are_meaningful

ROW_LIMIT = 60
CHANGES_SHOWN = 20  # per-change lines in a modify_and_resolve result; an edit list can run to thousands
DOC_CHARS = 30000  # about 7.5k tokens: whole short documents; long ones by section
TIME_LIMIT = 60.0
IIS_SOLVE_LIMIT = 60.0   # per solve inside an IIS search (LP: the whole HiGHS IIS)
IIS_BUDGET = 300.0       # a whole IIS search; large MIPs stop "reduced"
LARGE_MIP_ROWS = 500      # a MIP IIS search this size is slow: shorter default budget, and the result says so
FIX_CHECKS = 5           # suspicious_values solves the undo of at most this many flags, each and all together
FIX_CHECK_BUDGET = 60.0  # seconds for all of those solves; the rest are listed unchecked
LARGE_MIP_IIS_BUDGET = 60.0
IIS_SUMMARY_ROWS = 100   # above this, conflicts are shown by family with example rows
DEFAULT_SOLVER = "highs"  # default solver: 5-20x faster than SCIP on the large MIPs it was measured on
MAX_ATTAINABLE = 10


START_MIN_ROWS = 5_000   # a MIP this large solves from a starting plan (M08: its own plans can stay far off for minutes)
START_SHARE = 0.2         # the share of a solve's time limit the starting plan may take; the solve gets the rest
RACE = ("highs", "scip")  # solvers raced on a MIP's first solve
RACE_MIN_SIZE = 20_000   # rows x columns below which a MIP is not raced (every solve takes milliseconds)
RACE_MAX_NNZ = 1_000_000  # above this a MIP goes to HiGHS without a race: SCIP did not finish the 279k- and 837k-row
                          # hotel models, and racing both at 837k rows took 7 GB


def _capability(md: od.ModelData) -> tuple[str, str] | None:
    """A solver the model needs, whatever is faster."""
    if md.quadratic_objective and md.is_mip:
        return "scip", "mixed-integer QP: HiGHS cannot solve it"
    if not md.convex_objective():
        return "scip", "non-convex quadratic objective: HiGHS cannot solve it"
    return None


def chosen_solver(explicit: str | None = None) -> str | None:
    """The solver the user chose: ``explicit``, else the OPTLENS_SOLVER environment variable; None when neither is
    set or it is "auto"."""
    choice = explicit or os.environ.get("OPTLENS_SOLVER", "")
    return None if choice in ("", "auto") else choice


def _backend(solver: str | None, md: od.ModelData | None = None, route: str | None = None,
             default: str = DEFAULT_SOLVER):
    """The requested solver; else one the model needs (an open-source route only: a licensed solver does every step
    itself); else this session's routed solver; else the default."""
    if solver and solver not in od.BACKENDS:
        raise ValueError(f"unknown solver {solver!r}; use 'highs' or 'scip'")
    if solver:
        return od.BACKENDS[solver]
    if md is not None and not (route and od.BACKENDS[route].licensed) and (cap := _capability(md)):
        return od.BACKENDS[cap[0]]
    return od.BACKENDS[route] if route else od.BACKENDS[default]



CHANGE = {"type": "object", "properties": {
    "action": {"type": "string", "enum": ["set_rhs", "set_bounds", "set_objective_coef", "drop_constraint",
                                          "set_coef", "add_constraint"]},
    "name": {"type": "string"},
    "column": {"type": "string", "description": "set_coef: the variable whose coefficient changes."},
    "coefs": {"type": "object", "additionalProperties": {"type": "number"},
              "description": "add_constraint: {variable: coefficient}."},
    "lower": {"type": ["number", "null"]},
    "upper": {"type": ["number", "null"]},
    "value": {"type": "number"},
}, "required": ["action", "name"]}  # one change, as modify_and_resolve and try_options take it

TOOLS = [
    {
        "name": "get_model_overview",
        "description": "Size, variable types and solve status of a model version, plus constraint and variable counts per family (the name before its index). Version 'v0' is the original model. One solve.",
        "input_schema": {"type": "object", "properties": {"version": {"type": "string", "description": "Model version; default 'v0', the original."}}},
    },
    {
        "name": "query_constraints",
        "description": f"List constraints of a version with their full expression, sense, bounds and (if the version solved) activity, slack and binding flag. Filter by family and/or a name substring; at most {ROW_LIMIT} rows are returned, with the total match count.",
        "input_schema": {"type": "object", "properties": {
            "version": {"type": "string"},
            "family": {"type": "string", "description": "Exact family (base name), optional."},
            "name_contains": {"type": "string", "description": "Substring of the constraint name, optional."},
            "only_binding": {"type": "boolean", "description": "Only binding constraints (solved versions)."},
        }},
    },
    {
        "name": "query_variables",
        "description": f"List variables of a version with type, bounds, objective coefficient and (if solved) value. Filter by family and/or name substring; at most {ROW_LIMIT} rows are returned, with the total match count.",
        "input_schema": {"type": "object", "properties": {
            "version": {"type": "string"},
            "family": {"type": "string"},
            "name_contains": {"type": "string"},
            "only_nonzero": {"type": "boolean"},
        }},
    },
    {
        "name": "compute_iis",
        "description": "Irreducible infeasible subset of an infeasible version: the conflicting constraints with their expressions and bounds, grouped by family. Cost: seconds on an LP; on a MIP a search of many solves, which can take minutes on a large model and may stop at its time limit with a subset that is not minimal (it says so). Most useful when the conflict is small.",
        "input_schema": {"type": "object", "properties": {"version": {"type": "string", "description": "Model version; default 'v0', the original."}, "solver": {"type": "string", "enum": ["highs", "scip"], "description": "Omit to use the solver chosen for this model (the first result names it); pass one to override or cross-check."}, "time_limit": {"type": "number", "description": "Seconds for the whole search; default 300, or 60 on a MIP over 500 rows."}}},
    },
    {
        "name": "feasibility_relaxation",
        "description": "Minimal total change (sum of absolute RHS/bound changes) that makes an infeasible version feasible (one relaxation solve; slower than a plain solve on a large MIP). Returns every change needed, each with its current and suggested value; all are required together. Suggested values are exact: copy them as printed rather than rounding or recomputing them. only_families limits which constraint/variable families may change (others stay hard); relax_variable_bounds also allows variable bounds to move.",
        "input_schema": {"type": "object", "properties": {
            "version": {"type": "string"},
            "only_families": {"type": "array", "items": {"type": "string"}},
            "relax_variable_bounds": {"type": "boolean"},
            "solver": {"type": "string", "enum": ["highs", "scip"], "description": "Omit to use the solver chosen for this model (the first result names it); pass one to override or cross-check."}, "time_limit": {"type": "number", "description": "Seconds per solve; default 60."},
        }},
    },
    {
        "name": "suspicious_values",
        "description": "Constraint limits and variable bounds that break the pattern of their siblings (same family, same name pattern, or unusually large for the model), rows whose sense (<=, >=, =) differs from all their siblings, and, for a model that solves but gives a wrong optimum, objective coefficients with a lone opposite sign or a power-of-ten typo of their siblings', a constraint coefficient that alone keeps a row (or column) from matching others of the same shape, and a row missing from a numbered series or an indexed family; these often signal a data error such as a typo. Rows in the IIS are listed first when an IIS was already computed (it never starts one). On an infeasible model it also solves the undo of the first flags (a value put back to its group's typical one, a sense flipped back with the same limit), each alone and then the smallest sets that make the model solvable together, since two input errors can each keep it infeasible alone; those changes are ready to pass to modify_and_resolve (seconds on a small model, at most a minute). Flags are hints to confirm against the model document, not verdicts; many data errors do not look unusual at all.",
        "input_schema": {"type": "object", "properties": {"version": {"type": "string", "description": "Model version; default 'v0', the original."}}},
    },
    {
        "name": "fix_menu",
        "description": "Menu of repair levers for an infeasible version: for each conflicting constraint family (from the IIS by default, or the families you pass), the minimal change when ONLY that family may change, run in parallel. Each entry says whether that family alone is sufficient, its total change and the individual changes as current -> exact suggested value; the last entry is the unrestricted minimum across all families. Families whose rows all have RHS 0 are flagged as likely linking/definitional rather than business levers. Cost: one relaxation solve per family plus one, run in parallel; slow on a large MIP, where a family can come back 'unknown (time limit)'. drop_test and attainable_limit are cheaper screens.",
        "input_schema": {"type": "object", "properties": {
            "version": {"type": "string"},
            "families": {"type": "array", "items": {"type": "string"}},
            "solver": {"type": "string", "enum": ["highs", "scip"], "description": "Omit to use the solver chosen for this model (the first result names it); pass one to override or cross-check."}, "time_limit": {"type": "number", "description": "Seconds per solve; default 60."},
        }},
    },
    {
        "name": "why_not",
        "description": "Counterfactual on a solved version: force a variable to a value (or new bounds), re-solve as a new version, and report the new status, the objective change, which constraints switched between binding and slack (by family, with examples), and the variables that moved most.",
        "input_schema": {"type": "object", "properties": {
            "version": {"type": "string"},
            "variable": {"type": "string"},
            "value": {"type": "number", "description": "Fix the variable to this value."},
            "lower": {"type": "number"},
            "upper": {"type": "number"},
        }, "required": ["variable"]},
    },
    {
        "name": "compare_versions",
        "description": "What differs between two solved versions (default: v0 and the latest): status and objective, per variable family the total and nonzero count before and after, how many variables went up or down, the largest changes (with the total count), and the rows that became binding or slack. When the model has several equal-cost plans, the diff is against version_b's optimal plan closest to version_a's (one more solve), so it shows what the difference forces, not the solver's choice among ties; a first line says so. Use it before saying how a plan changed. Optional family limits it to one variable family.",
        "input_schema": {"type": "object", "properties": {
            "version_a": {"type": "string"}, "version_b": {"type": "string"},
            "family": {"type": "string", "description": "One variable family (the name before its index), optional."}}},
    },
    {
        "name": "version_history",
        "description": "What a version contains: the chain of versions from the original model to it, each with its description, its own changes, and its status and objective if solved. Versions branch: any version (v0 too) can be the base of a new change, so use this to say which changes a plan includes and which it does not.",
        "input_schema": {"type": "object", "properties": {"version": {"type": "string"}}, "required": ["version"]},
    },
    {
        "name": "marginal_value",
        "description": "Value of changing one constraint's limit: re-solves the version with the active bound moved up and down by delta and reports the objective change per unit each way. Works for LP, QP and MIP (integer decisions may change).",
        "input_schema": {"type": "object", "properties": {
            "version": {"type": "string"},
            "constraint": {"type": "string"},
            "delta": {"type": "number", "description": "Size of the change, in the constraint's units (default 1)."},
        }, "required": ["constraint"]},
    },
    {
        "name": "sensitivity_report",
        "description": f"LP/QP: shadow prices of constraints (objective change per unit increase of the active bound, in the model's max/min sense) with the RHS range where each holds (LP only), and reduced costs. MIP: integer-aware view instead: which constraints block improving +/-1 moves of integer variables (and which do so alone), which moves each integer variable is blocked from, plus fixed-plan shadow prices labelled as such. Filter by family or name substring; at most {ROW_LIMIT} rows.",
        "input_schema": {"type": "object", "properties": {
            "version": {"type": "string"},
            "kind": {"type": "string", "enum": ["constraints", "variables"]},
            "family": {"type": "string"},
            "name_contains": {"type": "string"},
        }, "required": ["kind"]},
    },
    {
        "name": "modify_and_resolve",
        "description": "Apply changes to a version, creating a new version, and solve it. Actions: set_rhs (lower/upper bound of a constraint; for <= use upper, for >= use lower, for = set both; a side left out stays unchanged, null removes that limit, so {'lower': 10, 'upper': null} turns <= 10 into >= 10), set_bounds (variable lower/upper), set_objective_coef (value), drop_constraint, set_coef (one coefficient: name is the constraint, column the variable, value the new coefficient; 0 removes the term), add_constraint (a new row: name, coefs {variable: coefficient}, lower and/or upper). Returns the new version id, its status and objective, the objective change versus the base version when both solved, and for each limit changed with set_rhs its activity and slack before and after (on a MIP, when slack was left yet loosening helped: the smallest whole-unit step that slack could not fit). One solve. With explain_conflict (for a change that repairs an infeasible version), it also reports, for each changed bound, the conflict that sets its amount (with that change slightly smaller the model is infeasible again; that conflict's IIS, which can differ from the first IIS), or that the amount has slack; this costs an IIS search per changed bound. To compare several candidate changes, use try_options.",
        "input_schema": {"type": "object", "properties": {
            "base_version": {"type": "string"},
            "description": {"type": "string"},
            "changes": {"type": "array", "items": CHANGE},
            "explain_conflict": {"type": "boolean", "description": "Also report the conflict that sets each changed amount (costly)."},
            "solver": {"type": "string", "enum": ["highs", "scip"], "description": "Omit to use the solver chosen for this model (the first result names it); pass one to override or cross-check."},
        }, "required": ["changes"]},
    },
    {
        "name": "try_options",
        "description": "Try several candidate change sets at once: each is applied to the base version as its own new version and solved (in parallel), and the status, objective and objective change of each are listed. Use it to compare alternative fixes or a few candidate values. Change format as in modify_and_resolve.",
        "input_schema": {"type": "object", "properties": {
            "base_version": {"type": "string"},
            "options": {"type": "array", "items": {"type": "object", "properties": {
                "label": {"type": "string"},
                "changes": {"type": "array", "items": CHANGE},
            }, "required": ["label", "changes"]}},
            "solver": {"type": "string", "enum": ["highs", "scip"], "description": "Omit to use the solver chosen for this model (the first result names it); pass one to override or cross-check."},
        }, "required": ["options"]},
    },
    {
        "name": "attainable_limit",
        "description": f"For each named constraint (up to {MAX_ATTAINABLE}): drop it, then minimize and maximize its own expression over the rest of the model. The range is what everything else allows, so it gives the exact limit that constraint needs as the only change (e.g. a requirement of 120 when at most 100 is reachable), or says the model stays infeasible without it. Called on the other side of a conflict (e.g. a capacity row), it gives what that side needs (the least capacity that meets everything else). Two solves per constraint.",
        "input_schema": {"type": "object", "properties": {
            "constraints": {"type": "array", "items": {"type": "string"}},
            "version": {"type": "string", "description": "Model version; default 'v0', the original."}, "solver": {"type": "string", "enum": ["highs", "scip"], "description": "Omit to use the solver chosen for this model (the first result names it); pass one to override or cross-check."}, "time_limit": {"type": "number", "description": "Seconds per solve; default 60."},
        }, "required": ["constraints"]},
    },
    {
        "name": "drop_test",
        "description": "For each constraint family (the ones you pass, or all), drop all its rows and solve: which single family, removed, restores feasibility, and the objective then. One solve per family, in parallel; a quick way to see which families are levers before sizing a change.",
        "input_schema": {"type": "object", "properties": {
            "families": {"type": "array", "items": {"type": "string"}},
            "version": {"type": "string", "description": "Model version; default 'v0', the original."}, "solver": {"type": "string", "enum": ["highs", "scip"], "description": "Omit to use the solver chosen for this model (the first result names it); pass one to override or cross-check."}, "time_limit": {"type": "number", "description": "Seconds per solve; default 60."},
        }},
    },
    {
        "name": "read_model_document",
        "description": "The model's documentation: business description, assumptions, input data tables, the model-building code and the results it reports. Without a query it returns the whole document, or its section headings when the document is long; with a query it returns the sections whose heading or text contains it (case-insensitive). Use it to check what a constraint or input means, or to compare a result with the documented one.",
        "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}},
    },
]

# Several models side by side (scenarios of one model, or alternatives): offered by the MCP server and to code; not in
# TOOLS, so an agent that works on one model sees a shorter tool list.
MULTI_MODEL_TOOLS = [
    {
        "name": "add_model",
        "description": "Load another model file (.lp, .mps or a .py that builds one) into this session under a name, next to the open model (v0), e.g. the scenarios of one study. Every tool then takes the name as its version; compare_models puts them side by side; compare_versions compares decisions when the models share their variables.",
        "input_schema": {"type": "object", "properties": {
            "path": {"type": "string"},
            "name": {"type": "string", "description": "A short name for it, e.g. min_cost (not v0, v1, ...)."}},
            "required": ["path", "name"]},
    },
    {
        "name": "compare_models",
        "description": "Side by side for several models or versions (default: v0 and every added model; one name gives that model's summary): status, objective, and with the same metrics for each, per constraint family the rows at a limit (for a one-row family such as a budget, its value against its limit), and per variable family the total and nonzero count. Use it first when a question compares scenarios, then drill into one with the other tools.",
        "input_schema": {"type": "object", "properties": {
            "models": {"type": "array", "items": {"type": "string"}, "description": "Names, default v0 and every added model."}}},
    },
]


def _bound_label(r: dict) -> str:
    if r["type"] == "constraint":
        return f"RHS {r['side']} bound ({'>=' if r['side'] == 'lower' else '<='} side)"
    return f"variable {r['side']} bound"


SAFE_NOTE = ("suggested values are the exact minimum; copy them exactly as printed (a rounded boundary value can "
             "land on the infeasible side)")


def _indices(name: str) -> list[str] | None:
    """The index values of a name like x[1,2,3] or price(1day_0_5); None when it has none."""
    m = re.match(r"^[^\[(]+[\[(](.*)[\])]$", name)
    return [p.strip() for p in m.group(1).split(",")] if m else None


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return "inf" if v > 0 else "-inf"
        return f"{v + 0.0:.15g}"  # exact enough to type back in: .6g rounding put a relaxed bound on the infeasible side
    return str(v)


def _proof(r: od.SolveResult) -> str:
    """What a result does not prove: a plan the solver stopped on gives its bound and gap; a time limit without a plan
    is not infeasibility."""
    if r.status == "OPTIMAL":
        return ""
    if r.x is None:
        return " (no plan found within the time limit; not proof that none exists)" if r.status == "TIME_LIMIT" else ""
    if r.gap is not None:
        return f" (best plan found, not proven optimal: bound {_fmt(r.bound)}, gap {r.gap:.2%})"
    return " (best plan found, not proven optimal)"


def _optimum_range(r: od.SolveResult) -> tuple[float, float] | None:
    """Where the optimal objective lies: the objective itself when proven, else between the plan and the bound."""
    if r.status == "OPTIMAL" and r.obj is not None:
        return r.obj, r.obj
    if r.x is None or r.gap is None:
        return None
    return min(r.obj, r.bound), max(r.obj, r.bound)


def _change_range(ra: od.SolveResult, rb: od.SolveResult) -> str:
    """For two results not both proven: the interval the optimal objective change from ra to rb must lie in."""
    a, b = _optimum_range(ra), _optimum_range(rb)
    if a is None or b is None or (a[0] == a[1] and b[0] == b[1]):
        return ""
    return f"; the optimal change lies between {_fmt(b[0] - a[1])} and {_fmt(b[1] - a[0])} (plans not proven optimal)"


def _conflict_rows(md, rows, limit=ROW_LIMIT, per_family=3) -> list[str]:
    """IIS rows with their expressions. A conflict over IIS_SUMMARY_ROWS rows is shown by family, a few example
    rows each: a list of hundreds of rows is unreadable, and large conflicts are real (the ad model's LP
    relaxation needs 451 rows)."""
    def line(n):
        i = md.row_index(n)
        return f"{n}: {md.expression(i)[:300 if len(rows) > IIS_SUMMARY_ROWS else 500]} {_sense(md.row_lo[i], md.row_hi[i])}"
    rows = [n for n in rows if n in md.row_names]
    if len(rows) <= IIS_SUMMARY_ROWS:
        return [line(n) for n in rows[:limit]]
    # Example rows: those breaking their siblings' pattern first (a flipped sense, an odd limit), since a data
    # error hidden among hundreds of conflict rows is the one worth seeing.
    odd = {f["name"] for f in family_outliers(md, set(rows), max_flags=200) if f["in_iis"]}
    by_fam: dict[str, list[str]] = {}
    for n in sorted(rows, key=lambda n: n not in odd):
        by_fam.setdefault(base_name(n), []).append(n)
    out = [f"large conflict ({len(rows)} rows), shown by family with example rows"
           + (" (rows breaking their siblings' pattern first)" if odd else "") + ":"]
    for fam, names in sorted(by_fam.items(), key=lambda kv: -len(kv[1])):
        out.append(f"{fam}: {len(names)} rows, e.g.")
        out += ["  " + line(n) for n in names[:per_family]]
    return out


def _sense(lo: float, hi: float) -> str:
    if lo == hi:
        return f"= {_fmt(hi)}"
    if np.isfinite(lo) and np.isfinite(hi):
        return f"in [{_fmt(lo)}, {_fmt(hi)}]"
    return f"<= {_fmt(hi)}" if np.isfinite(hi) else f">= {_fmt(lo)}"


def _change_text(md: od.ModelData, c: dict) -> str:
    """One flag_fix change, before -> after, in the words compute_iis uses."""
    if c["action"] == "set_rhs":
        i = md.row_index(c["name"])
        lo, hi = md.row_lo[i], md.row_hi[i]
        new_lo, new_hi = _limit(c, "lower"), _limit(c, "upper")
        return f"{c['name']}: {_sense(lo, hi)} -> {_sense(lo if new_lo is None else new_lo, hi if new_hi is None else new_hi)}"
    if c["action"] == "set_bounds":
        j, side = md.col_index(c["name"]), "upper" if "upper" in c else "lower"
        old = md.col_ub[j] if side == "upper" else md.col_lb[j]
        return f"{c['name']} {side} bound: {_fmt(old)} -> {_fmt(c[side])}"
    return f"{c['name']} / {c['column']} coefficient: {_fmt(md.A[md.row_index(c['name']), md.col_index(c['column'])])} -> {_fmt(c['value'])}"


MODELS_KEPT = 4  # edited versions whose model stays built; older ones are rebuilt from their changes when used


class Version:
    """One version of a model. A root (v0, or a model added by name) keeps its ModelData; an edited version keeps
    its parent and the changes that made it, and its model is rebuilt from them when needed. The Session keeps only
    the MODELS_KEPT most recently used edited models built: at 837k rows a full copy is about 125 MB, and a
    conversation makes dozens of versions."""

    def __init__(self, md: od.ModelData | None, parent: str | None, description: str,
                 result: od.SolveResult | None = None, iis_result: tuple | None = None, solver: str | None = None,
                 changes: tuple = (), build=None):
        self._md, self.parent, self.description = md, parent, description
        self.result = result
        self.iis_result = iis_result  # (families, IIS), cached by Session.iis
        self.solver = solver
        self.changes = tuple(changes)  # an edited version: the changes applied to its parent
        self._build = build            # an edited version: returns its model (Session._model)

    @property
    def md(self) -> od.ModelData:
        return self._md if self._md is not None else self._build()


OBJ_TERMS_SHOWN = 10  # get_model_overview lists the objective's terms when it has this many or fewer


class SolverUnavailable(RuntimeError):
    """The solver the user chose cannot run this model, and no other may stand in without asking."""


@dataclass
class Session:
    versions: dict[str, Version] = field(default_factory=dict)
    doc_path: str | None = None
    route: str | None = None       # solver chosen for this model (all versions share its structure)
    route_note: str = ""           # why; shown once in the next tool result
    race: bool = True
    time_limit: float = TIME_LIMIT      # per solve; IIS searches and relaxations have their own budgets
    default_solver: str = DEFAULT_SOLVER  # until a model's first solve picks one (see _route_first)
    prefer: str | None = None  # a solver to use whenever it can run this model (e.g. a licensed Gurobi); else routing
    # The user chose ``prefer`` (not "auto"): a licensed solver that cannot run a model stops with a question instead
    # of handing the model to HiGHS or SCIP, and no other solver may be passed in
    only_prefer: bool = False
    large_mip_iis_budget: float = LARGE_MIP_IIS_BUDGET  # the MCP server lowers it to fit its client's call timeout
    # the most a tool's time_limit may ask for (the MCP server sets it inside its client's call timeout, so a call
    # answers instead of being cut off); engine defaults then stay within time_limit. None: no cap
    max_time_limit: float | None = None
    # Tools may be called from several threads at once (the plugin's analysts share one server): new version ids are
    # allocated under this lock; solves run outside it, in their own processes.
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False, compare=False)
    # closest optimal plans for compare_versions, by (version_a, version_b): (status, plan or None)
    _closest: dict = field(default_factory=dict, repr=False, compare=False)
    # built models of edited versions, most recently used last (at most MODELS_KEPT)
    _models: OrderedDict = field(default_factory=OrderedDict, repr=False, compare=False)
    _models_lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)
    # the file each model added by name came from (export_versions)
    _sources: dict = field(default_factory=dict, repr=False, compare=False)

    def call(self, name: str, args: dict) -> tuple[str, bool]:
        """Run one tool by name, as an agent sees it: (text, is_error). An exception becomes an error result; the
        text ends with the call's seconds and, once, the solver-routing note."""
        t0 = time.time()
        try:
            out, err = getattr(self, name)(**args), False
        except Exception as e:
            out, err = f"{type(e).__name__}: {e}", True
        note, self.route_note = self.route_note, ""
        return f"{out}\n[{time.time() - t0:.1f} s]" + (f"\n[{note}]" if note else ""), err

    def _limit(self, asked: float | None, default: float) -> float:
        """The seconds a search may take. With a ceiling (max_time_limit, set by the MCP server): an engine default is
        kept within the session's solve limit, and what a call asked for within the ceiling, said in the result when
        lowered. Without one (bench, optchat): what the call asked for, else the default."""
        if self.max_time_limit is None:
            return asked or default
        if not asked:
            return min(default, self.time_limit)
        if asked <= self.max_time_limit:
            return asked
        note = (f"time_limit {asked:g} s lowered to {self.max_time_limit:g} s, the most a tool call may take here "
                "(OPTLENS_CALL_LIMIT, which must stay within the client's tool-call timeout)")
        self.route_note = f"{self.route_note}; {note}" if self.route_note else note
        return self.max_time_limit

    def _new_version(self, md, base: str | None, description: str, solver: str | None = None,
                     name: str | None = None, changes=None) -> str:
        """A new version: a root keeps md; an edited one (changes given, made from base) keeps its changes, and md
        is kept among the recently built models."""
        with self._lock:
            if name is not None and name in self.versions:
                raise ValueError(f"{name!r} is already open; existing: {', '.join(self.versions)}")
            vid = name or f"v{len(self.versions)}"
            while vid in self.versions:  # never reuse an id, whatever was added in between
                vid = f"v{int(vid[1:]) + 1}"
            if changes is None or base is None:
                self.versions[vid] = Version(md, base, description, solver=solver)
            else:
                self.versions[vid] = Version(None, base, description, solver=solver, changes=changes,
                                             build=lambda: self._model(vid))
                if md is not None:
                    self._keep(vid, md)
            return vid

    def _model(self, vid: str) -> od.ModelData:
        """An edited version's model: kept if recently used, else rebuilt in one pass from its nearest ancestor whose
        model is at hand (a root, or a kept version), with the changes of every version in between in order. Only
        the version asked for is kept, so walking back a long chain or branching from v0 does not push the versions in
        use out of the kept few."""
        with self._models_lock:
            if vid in self._models:
                self._models.move_to_end(vid)
                return self._models[vid]
            path, cur = [], vid
            while self.versions[cur]._md is None and cur not in self._models:
                path.append(cur)
                cur = self.versions[cur].parent
            start = self.versions[cur]._md if self.versions[cur]._md is not None else self._models[cur]
            md = self._apply(start, [c for p in reversed(path) for c in self.versions[p].changes])
            self._keep(vid, md)
            return md

    def version_history(self, version: str) -> str:
        chain = self.lineage(version)
        changes = sum(len(self.get(v).changes) for v in chain)
        lines = [f"{version}: {len(chain) - 1} step(s) and {changes} change(s) from {chain[0]}"]
        for vid in chain:
            v = self.get(vid)
            line = f"{vid}{' from ' + v.parent if v.parent else ''}: {v.description}"
            if v.result is not None:
                line += f" [{v.result.status}" + (f", objective {_fmt(v.result.obj)}]" if v.result.obj is not None else "]")
            lines.append(line)
            lines += [f"  - {_plain_change(c)}" for c in v.changes[:CHANGES_SHOWN]]
            if len(v.changes) > CHANGES_SHOWN:
                lines.append(f"  ... ({len(v.changes) - CHANGES_SHOWN} more)")
        others = [k for k, v in self.versions.items() if v.parent is not None and k not in chain]
        if others:
            lines.append(f"not in {version}: {', '.join(others)} (other branches)")
        return "\n".join(lines)

    def lineage(self, vid: str) -> list[str]:
        """The versions from the root to vid, root first."""
        out = [vid]
        while self.get(out[-1]).parent is not None:
            out.append(self.get(out[-1]).parent)
        return out[::-1]

    def export_versions(self) -> dict:
        """What it takes to rebuild this session's versions in another process (a resumed conversation), as JSON:
        each added model's file and each edited version's parent, changes, description and solver, in the order they
        were made; plus session.carried (bridge helpers record carried versions there). Solutions are not kept."""
        out = []
        for vid, v in self.versions.items():
            if v.parent is None:
                out.append({"id": vid, "root": True, "path": self._sources.get(vid), "description": v.description,
                            "solver": v.solver})
            elif v.changes:
                out.append({"id": vid, "parent": v.parent, "description": v.description, "solver": v.solver,
                            "changes": [dict(c) for c in v.changes]})
        return json.loads(json.dumps({"versions": out, "carried": dict(getattr(self, "carried", {}))},
                                     default=lambda x: x.item() if hasattr(x, "item") else str(x)))

    def restore_versions(self, state: dict) -> str:
        """Rebuild the versions export_versions wrote, under their ids: added models are opened from their files
        (unless already open) and edited versions are stored as changes, solved on first use. Returns a summary."""
        made, skipped = [], []
        for rec in state.get("versions", []):
            vid = rec["id"]
            if vid in self.versions:
                continue
            if rec.get("root"):
                if rec.get("path") and Path(rec["path"]).exists():
                    self.add_model(rec["path"], vid, rec.get("description"))
                    made.append(vid)
                else:
                    skipped.append(f"{vid} (its file is gone)")
            elif rec.get("parent") in self.versions:
                with self._lock:
                    self.versions[vid] = Version(None, rec["parent"], rec.get("description", ""),
                                                 solver=rec.get("solver"), changes=rec["changes"],
                                                 build=lambda vid=vid: self._model(vid))
                made.append(vid)
            else:
                skipped.append(f"{vid} (its parent {rec.get('parent')} is missing)")
        if state.get("carried"):
            self.carried = {**getattr(self, "carried", {}), **state["carried"]}
        return (f"restored {len(made)} versions" + (f" ({', '.join(made)})" if made else "")
                + "; edited versions solve again on first use"
                + (f"; not restored: {', '.join(skipped)}" if skipped else ""))

    def _keep(self, vid: str, md: od.ModelData) -> None:
        with self._models_lock:
            self._models[vid] = md
            self._models.move_to_end(vid)
            while len(self._models) > MODELS_KEPT:
                self._models.popitem(last=False)

    def strict(self) -> bool:
        """The user chose a licensed solver: every step stays on it."""
        return self.only_prefer and self.prefer in od.BACKENDS and od.BACKENDS[self.prefer].licensed

    def _bk(self, md, solver=None):
        if solver is None and self.route is None and "v0" in self.versions:
            self.solved("v0")  # the first solve picks this model's solver
        if solver and self.strict() and solver != self.prefer:
            raise ValueError(f"this session uses only {self.prefer}, the user's choice; ask the user before using "
                             f"{solver}, and if they agree, open the model again with solver {solver}")
        return _backend(solver, md, self.route, self.default_solver)

    def _route_first(self, v: "Version") -> od.SolveResult:
        """First solve of the session: pick the solver for this model and keep it. A preferred solver is tried
        first; if it is not installed, cannot handle the model, or its license is too small for it (a size-limited
        Gurobi license), the model is routed as usual and the note says why."""
        md = v.md
        fallback = ""
        if self.prefer:
            try:
                r = self._started(md, lambda t, s: od.BACKENDS[self.prefer].solve(md, t, start=s))
                self.route = self.prefer
                self._route_note_first(f"solver for this model: {self.prefer} (preferred); pass solver to override")
                return r
            except (od.LicenseLimit, od.UnsupportedModel, ImportError) as e:
                if self.strict():
                    raise SolverUnavailable(
                        f"{self.prefer} cannot run this model ({type(e).__name__}: {str(e)[:120]}). The user chose "
                        f"{self.prefer}; ask them before switching. HiGHS and SCIP (open source) can run it: open the "
                        "model again with solver highs or scip, or unset OPTLENS_SOLVER.") from e
                fallback = f"preferred solver {self.prefer} not used ({type(e).__name__}: {str(e)[:120]}); "
        r = self._route_open_source(md)
        self.route_note = fallback + self.route_note
        return r

    def _route_open_source(self, md: od.ModelData) -> od.SolveResult:
        if cap := _capability(md):
            self.route, why = cap
        elif not md.is_mip:
            self.route, why = "highs", "convex QP" if md.quadratic_objective else "LP"
        elif not self.race or md.num_rows * md.num_cols < RACE_MIN_SIZE:
            self.route, why = self.default_solver, "small MIP"
        elif md.A.nnz > RACE_MAX_NNZ:
            self.route, why = "highs", f"large MIP ({md.A.nnz:,} nonzeros): HiGHS, without a race"
        else:
            t0 = time.time()
            won = []

            def raced(t, s):
                b, r = od.race(md, [od.BACKENDS[n] for n in RACE], t, start=s)
                won.append(b)
                return r
            r = self._started(md, raced)
            b = won[0]
            self.route, why = b.name, f"finished first when {' and '.join(RACE)} raced on this model ({time.time() - t0:.1f} s)"
            self._route_note_first(f"solver for this model: {self.route} ({why}); pass solver to override")
            return r
        self.route_note = f"solver for this model: {self.route} ({why}); pass solver to override"
        return self._started(md, lambda t, s: od.BACKENDS[self.route].solve(md, t, start=s))

    def _route_note_first(self, note: str) -> None:
        """The routing note, before what the solve itself noted (a starting plan)."""
        self.route_note = f"{note}; {self.route_note}" if self.route_note else note

    def _started(self, md: od.ModelData, solve, limit: float | None = None) -> od.SolveResult:
        """``solve(time_limit, start)`` for one model within ``limit`` (default: the session's). A MIP of
        START_MIN_ROWS or more first gets a starting plan built from its LP relaxation (od.starting_plan, within
        START_SHARE of the limit; the solve gets the rest), and the note says whether the solver improved on it, which
        tells an agent whether a longer time_limit could help; the start is kept if the solver returns nothing better
        (a solve killed past its limit loses its plan)."""
        limit = limit or self.time_limit
        if not (md.is_mip and md.num_rows >= START_MIN_ROWS):
            return solve(limit, None)
        t0 = time.time()
        lp_backend = od.BACKENDS[self.prefer] if self.strict() else od.BACKENDS["highs"]
        plan = od.starting_plan(md, lp_backend, START_SHARE * limit)
        t1 = time.time()
        r = solve(max(1.0, limit - (t1 - t0)), None if plan is None else plan.x)
        if plan is None:
            return r
        better = r.obj is not None and abs(r.obj - plan.obj) > 1e-9 * max(1.0, abs(plan.obj)) and (
            r.obj < plan.obj if md.minimize else r.obj > plan.obj)
        if r.status == "OPTIMAL":
            then = f"then solved to optimality{f' ({r.obj:,.6g})' if better else ', keeping it'}"
        elif better:
            then = f"the solver improved it to {r.obj:,.6g} in {time.time() - t1:.0f} s"
        else:
            then = f"the solver found no better plan in {time.time() - t1:.0f} s"
        note = f"started from a plan built from the {plan.how} ({plan.obj:,.6g}, {plan.seconds:.1f} s); {then}"
        self.route_note = f"{self.route_note}; {note}" if self.route_note else note
        worse = r.obj is None or (r.obj > plan.obj if md.minimize else r.obj < plan.obj)
        if r.status in ("TIME_LIMIT", "OTHER") and worse:
            return od.SolveResult("TIME_LIMIT", obj=plan.obj, x=plan.x, bound=r.bound)
        return r

    def get(self, vid: str) -> Version:
        if vid not in self.versions:
            raise KeyError(f"unknown version {vid!r}; existing: {', '.join(self.versions)}")
        return self.versions[vid]

    def solved(self, vid: str, time_limit: float | None = None) -> od.SolveResult:
        """The version's solve, done once; ``time_limit`` for that first solve (default: the session's)."""
        v = self.get(vid)
        if v.result is None:
            t0 = time.time()
            if v.solver is None and self.route is None:
                v.result = self._route_first(v)
            else:
                bk = self._bk(v.md, v.solver)
                v.result = self._started(v.md, lambda t, s: bk.solve(v.md, t, start=s), time_limit)
            v.result = self._other_verdict(vid, v, v.result, time.time() - t0)
        return v.result

    def _other_verdict(self, vid: str, v: "Version", r: od.SolveResult, spent: float) -> od.SolveResult:
        """A solve with no verdict (no plan, and not proven infeasible or unbounded) gets one try on the other
        open-source solver, within what is left of the most a call may solve: HiGHS ran 120 s without deciding the
        E64 feed model, which SCIP proves infeasible in 5 s. The one that decides becomes this version's solver (and
        the model's, for the first version), so the tools after it use it too."""
        used = v.solver or self.route
        other = {"highs": "scip", "scip": "highs"}.get(used or "")
        if r.x is not None or r.status not in ("OTHER", "TIME_LIMIT") or not other or self.strict():
            return r
        left = min(self.time_limit, (self.max_time_limit or self.time_limit + spent) - spent)
        if left < 5:
            self._note(f"{used} gave no verdict on {vid} in its time, and none was left to try {other}: pass "
                       f"solver='{other}' to try it")
            return r
        try:
            r2 = self._started(v.md, lambda t, s: od.BACKENDS[other].solve(v.md, t, start=s), left)
        except ImportError:
            return r
        if r2.x is None and r2.status in ("OTHER", "TIME_LIMIT"):
            self._note(f"neither {used} nor {other} gave a verdict on {vid} in the time")
            return r
        v.solver = other
        if self.route == used and all(w is v or w.result is None for w in self.versions.values()):
            self.route = other
        self._note(f"{used} gave no verdict on {vid} ({r.status}), so {other} solved it: {r2.status}")
        return r2

    def _note(self, note: str) -> None:
        self.route_note = f"{self.route_note}; {note}" if self.route_note else note

    def iis(self, vid: str, solver: str | None = None, time_limit: float | None = None) -> tuple[list[str], od.IIS]:
        """The version's IIS, computed once and shared by compute_iis and suspicious_values (each search can take
        up to IIS_BUDGET on a large MIP). Returns (conflicting families for a MIP, IIS)."""
        v = self.get(vid)
        large = v.md.is_mip and v.md.num_rows > LARGE_MIP_ROWS
        bk, budget = self._bk(v.md, solver), self._limit(time_limit, self.large_mip_iis_budget if large else IIS_BUDGET)
        if v.iis_result is None:
            if v.md.is_mip:
                # A row-level MIP IIS of the full model doesn't finish on open-source solvers at this
                # size; find the conflicting families first, then the IIS within them.
                v.iis_result = od.family_first_iis(v.md, bk, min(IIS_SOLVE_LIMIT, budget), budget=budget)
            else:
                v.iis_result = ([], od.get_iis(v.md, bk, budget, time_budget=budget))
        return v.iis_result

    def has_solution(self, vid: str) -> bool:
        r = self.solved(vid)
        return r.x is not None

    # ---- tools ----

    def get_model_overview(self, version: str = "v0") -> str:
        v, r = self.get(version), self.solved(version)
        md = v.md
        lines = [f"version {version}: {v.description}",
                 f"rows {md.num_rows}, columns {md.num_cols} ({int(md.is_int.sum())} integer), "
                 f"{'minimize' if md.minimize else 'maximize'}",
                 f"status: {r.status}{_proof(r)}"]
        if md.quadratic_objective:
            lines.append("objective is quadratic (a QP; MIQP if integer variables are present)")
        if r.obj is not None:
            lines.append(f"objective: {_fmt(r.obj)}")
        nz = np.flatnonzero(md.obj)
        if len(nz) <= OBJ_TERMS_SHOWN:  # a short objective is read at a glance (a lone term with the wrong sign)
            terms = " ".join(f"{md.obj[j]:+.15g} {md.col_names[j]}" + (f" (value {_fmt(r.x[j])})" if r.x is not None else "")
                             for j in nz)
            lines.append(f"objective terms ({len(nz)} nonzero): {terms or 'none'}"
                         + (f" + constant {_fmt(md.obj_offset)}" if md.obj_offset else ""))
        else:
            lines.append(f"objective terms: {len(nz)} nonzero")
        # Anonymous names (ROW00001, B17) make every row its own family; their digit patterns group them instead.
        group = base_name if names_are_meaningful(md) else (lambda n: re.sub(r"\d+", "#", n))
        rows_by_family: dict[str, list[int]] = {}
        for i, n in enumerate(md.row_names):
            rows_by_family.setdefault(group(n), []).append(i)
        solved = self.has_solution(version)
        ib = sensitivity.integer_binding(md, r.x) if solved and md.is_mip else None
        header = ", binding" if solved else ""
        header += ", integer-binding (a +/-1 move of an integer variable would violate it)" if ib else ""
        lines.append(f"constraint families (count{header}){'' if group is base_name else '; names without meaning, grouped by their pattern with digits as #'}:")
        act = md.A @ r.x if solved else None
        for fam, idx in sorted(rows_by_family.items()):
            extra = ""
            if act is not None:
                b = sum(1 for i in idx if abs(act[i] - md.row_hi[i]) <= 1e-6 or abs(act[i] - md.row_lo[i]) <= 1e-6)
                extra = f", {b} binding"
            if ib is not None:
                extra += f", {int(ib['effective_binding'][idx].sum())} integer-binding"
            lines.append(f"  {fam}: {len(idx)}{extra}")
        cols_by_family: dict[str, int] = {}
        for n in md.col_names:
            cols_by_family[group(n)] = cols_by_family.get(group(n), 0) + 1
        lines.append("variable families (count): " + ", ".join(f"{k}: {c}" for k, c in sorted(cols_by_family.items())))
        return "\n".join(lines)

    def _match(self, names, family, name_contains):
        def in_family(n):  # a digit pattern from the overview (ROW#) or a family name
            return re.sub(r"\d+", "#", n) == family if "#" in family else base_name(n) == family

        return [i for i, n in enumerate(names)
                if (not family or in_family(n)) and (not name_contains or name_contains in n)]

    def query_constraints(self, version="v0", family=None, name_contains=None, only_binding=False) -> str:
        md = self.get(version).md
        idx = self._match(md.row_names, family, name_contains)
        act = md.A @ self.solved(version).x if self.has_solution(version) else None
        rows = []
        for i in idx:
            line = f"{md.row_names[i]}: {md.expression(i)} {_sense(md.row_lo[i], md.row_hi[i])}"
            if act is not None:
                slack_u = md.row_hi[i] - act[i] if np.isfinite(md.row_hi[i]) else np.inf
                slack_l = act[i] - md.row_lo[i] if np.isfinite(md.row_lo[i]) else np.inf
                binding = min(abs(slack_u), abs(slack_l)) <= 1e-6
                if only_binding and not binding:
                    continue
                line += f" | activity {_fmt(act[i])}, slack {_fmt(min(slack_u, slack_l))}, binding {binding}"
            rows.append(line)
        head = f"{len(rows)} constraints match" + (f"; showing first {ROW_LIMIT}" if len(rows) > ROW_LIMIT else "")
        return "\n".join([head] + [r[:600] for r in rows[:ROW_LIMIT]])

    def query_variables(self, version="v0", family=None, name_contains=None, only_nonzero=False) -> str:
        md = self.get(version).md
        x = self.solved(version).x if self.has_solution(version) else None
        rows = []
        for j in self._match(md.col_names, family, name_contains):
            if only_nonzero and (x is None or abs(x[j]) <= 1e-9):
                continue
            kind = "int" if md.is_int[j] else "cont"
            line = f"{md.col_names[j]}: {kind} [{_fmt(md.col_lb[j])}, {_fmt(md.col_ub[j])}], obj {_fmt(md.obj[j])}"
            if x is not None:
                line += f", value {_fmt(x[j])}"
            rows.append(line)
        head = f"{len(rows)} variables match" + (f"; showing first {ROW_LIMIT}" if len(rows) > ROW_LIMIT else "")
        return "\n".join([head] + rows[:ROW_LIMIT])

    def compute_iis(self, version="v0", solver=None, time_limit=None) -> str:
        md = self.get(version).md
        if self.solved(version).status not in ("INFEASIBLE", "INF_OR_UNBD"):
            return f"version {version} is {self.solved(version).status}, not infeasible; no IIS."
        lines = []
        kept, iis = self.iis(version, solver, time_limit)
        if md.is_mip:
            lines.append(f"conflicting constraint families (each needed for infeasibility): {', '.join(kept)}")
        fams: dict[str, int] = {}
        for n in iis.rows:
            fams[base_name(n)] = fams.get(base_name(n), 0) + 1
        lines.append(f"IIS ({iis.method}): {len(iis.rows)} constraints; by family: "
                     + ", ".join(f"{k}: {c}" for k, c in sorted(fams.items(), key=lambda kv: -kv[1])))
        if iis.note:
            lines.append(f"note: {iis.note}")
        step = next((t for t in iis.method.split(":") if t in od.BACKENDS), None)
        if step and step != (chosen := self._bk(md, solver or self.get(version).solver).name):
            lines.append(f"note: computed by {step}, not {chosen}: both are open source, and the engine uses whichever "
                         "can do this step")
        if iis.method.endswith(":reduced"):
            lines.append("note: time limit reached; this subset is infeasible but may not be minimal"
                         + ("; on a MIP this size drop_test and attainable_limit usually answer faster, or pass a longer "
                            "time_limit" if md.num_rows > LARGE_MIP_ROWS else ""))
        if iis.method.startswith("lp_relaxation") or ":lp_relaxation" in iis.method:
            lines.append("note: the conflict holds even without integrality (the LP relaxation is infeasible); "
                         "this is the relaxation's minimal conflict")
        lines += _conflict_rows(md, iis.rows)
        if iis.col_bounds:
            lines.append("variable bounds in the IIS:")
            for c in iis.col_bounds[:ROW_LIMIT]:
                j = md.col_index(c)
                lines.append(f"  {c}: [{_fmt(md.col_lb[j])}, {_fmt(md.col_ub[j])}]")
        return "\n".join(lines)

    def feasibility_relaxation(self, version="v0", only_families=None, relax_variable_bounds=False, solver=None,
                               time_limit=None) -> str:
        md = self.get(version).md
        only = set(only_families) if only_families else None
        res = od.feas_relax(md, self._bk(md, solver), relax_bounds=relax_variable_bounds, time_limit=self._limit(time_limit, self.time_limit),
                            only=only, min_objective=True)
        if res.total_violation is None:
            scope = f" when only {sorted(only)} may change" if only else ""
            if res.status == "TIME_LIMIT":
                return f"unknown{scope}: the relaxed model hit the time limit without a feasible point (not proof that none exists)"
            return f"no relaxation found{scope}: relaxed model status {res.status}"
        changes = od.relaxed_bounds(md, res.relaxations)
        linking = od.linking_families(md)
        tag = lambda r: " [linking row, not a business lever]" if base_name(r["name"]) in linking else ""  # noqa: E731
        changes.sort(key=lambda r: bool(tag(r)))  # business levers first
        lines = [f"status {res.status}; total change {_fmt(res.total_violation)}; {len(changes)} changes, all required together:"]
        lines += [f"  {r['name']}: {_bound_label(r)} {_fmt(r['current'])} -> {_fmt(r['new'])} (minimal change {_fmt(r['value'])}){tag(r)}"
                  for r in changes]
        if changes and all(tag(r) for r in changes):
            lines.append("every change is on a linking row: the smallest fix is no business decision; fix_menu or "
                         "attainable_limit show the business levers and their amounts")
        if not only:
            lines.append(self._other_levers(version, {base_name(r["name"]) for r in changes}, linking))
        lines.append(SAFE_NOTE)
        if res.objective is not None:
            lines.append(f"best plan at this minimal change: objective {res.objective:.10g}")
        else:
            why = ("the smallest change was not proven within the time limit" if res.status != "OPTIMAL" else
                   "no time was left for the best-plan solve" if res.best_status == "skipped" else
                   "the best-plan solve did not finish in time" if res.best_status == "TIME_LIMIT" else
                   f"the best-plan solve ended {res.best_status}")
            lines.append(f"best plan at this minimal change: not computed ({why})")
        if res.status == "TIME_LIMIT":
            lines.append("note: solver hit its time limit; this relaxation is feasible but may not be minimal")
        return "\n".join(lines)

    def _other_levers(self, version: str, used: set[str], linking: set[str]) -> str:
        """The conflict's business families this relaxation left unchanged: the smallest total change is one fix,
        and a planner usually weighs each lever on its own. Uses an IIS already computed; never starts one."""
        v = self.get(version)
        if v.iis_result is None:
            return ("this is the smallest fix in total; other limits may each fix it alone with a different amount "
                    "(compute_iis shows the conflict, fix_menu one fix per family)")
        iis = v.iis_result[1]
        rows = sorted({base_name(n) for n in iis.rows} - used - linking)
        cols = sorted({base_name(n) for n in iis.col_bounds} - used)
        others = rows + [f"bounds of {c}" for c in cols]
        if not others:
            return "every business family in the conflict is changed by this fix"
        shown = ", ".join(others[:8]) + (f" and {len(others) - 8} more" if len(others) > 8 else "")
        return (f"also in the conflict, unchanged by this fix: {shown}; each may fix it alone with a different amount "
                "(fix_menu gives one per family, attainable_limit one per row)")

    @staticmethod
    def _apply(md, changes, what: str = ""):
        """The changes applied in order. Each is checked against the model as it stands when it applies (an earlier
        one can add or drop a row); a change to a name an earlier change failed on is not reported again. Crossed
        limits are checked once all changes are in, so a limit can move in two steps. All problems are raised
        together, so nothing half-applied is kept."""
        problems, failed, touched = [], set(), {}
        drops: list[str] = []  # consecutive drop_constraint names, applied together (one copy of the model, not one each)
        batch: list[dict] = []  # consecutive valid set_rhs or set_objective_coef changes, applied together: a carried
        # upstream result (new prices and demand) is thousands of them, one copy of the model each otherwise
        names: dict = {}

        def flush():
            nonlocal md, batch
            if batch:
                md, batch = _apply_batch(md, batch), []

        for k, c in enumerate(changes, 1):
            if c.get("name") in failed:
                continue
            if c.get("action") == "drop_constraint" and c.get("name") in md.row_names and c.get("name") not in drops:
                flush()
                drops.append(c["name"])
                continue
            if drops:
                md, drops = md.drop_rows(drops), []
                names = {}
            if c.get("action") in ("set_rhs", "set_objective_coef") and (not batch or batch[0]["action"] == c["action"]):
                if not names or names.get("_n") != len(md.row_names):  # != : `is not` on ints rebuilt the sets per change
                    names = {"_n": len(md.row_names), "rows": set(md.row_names), "cols": set(md.col_names)}
                ok = c.get("name") in (names["rows"] if c["action"] == "set_rhs" else names["cols"])
                if ok and c["action"] == "set_rhs":
                    ok = all(k_ not in c or c[k_] is None or (isinstance(c[k_], (int, float)) and not isinstance(c[k_], bool)
                                                           and c[k_] == c[k_]) for k_ in ("lower", "upper"))
                elif ok:
                    ok = isinstance(c.get("value"), (int, float)) and not isinstance(c["value"], bool) and c["value"] == c["value"]
                if ok:
                    batch.append(c)
                    if c["action"] == "set_rhs":
                        touched[(False, c["name"])] = k
                    continue
            flush()
            try:
                md = _apply_one(md, c)
                if c.get("action") in ("set_rhs", "set_bounds", "add_constraint"):
                    touched[(c["action"] == "set_bounds", c["name"])] = k
            except ValueError as e:
                failed.add(c.get("name"))
                problems.append((k, c, e))
        flush()
        if drops:
            md = md.drop_rows(drops)
        for (is_col, name), k in touched.items():
            names, lo, hi = (md.col_names, md.col_lb, md.col_ub) if is_col else (md.row_names, md.row_lo, md.row_hi)
            if name in names and lo[names.index(name)] > hi[names.index(name)]:
                i = names.index(name)
                problems.append((k, changes[k - 1], ValueError(f"lower {_fmt(lo[i])} is above upper {_fmt(hi[i])}")))
        if problems:
            problems.sort(key=lambda p: p[0])
            shown = [f"{what}change {k} ({c.get('action')} {c.get('name')}): {e}{_suggestion(e)}"
                     for k, c, e in problems[:CHANGES_SHOWN]]
            if len(problems) > CHANGES_SHOWN:
                shown.append(f"... ({len(problems) - CHANGES_SHOWN} more)")
            raise ValueError("no changes applied; fix these and call again:\n" + "\n".join(shown))
        return md

    def modify_and_resolve(self, base_version="v0", description="", changes=(), explain_conflict=False,
                           solver=None, time_limit=None) -> str:
        return self._modify(base_version, description, changes, explain_conflict, solver, time_limit)[1]

    def _modify(self, base_version, description, changes, explain_conflict=False, solver=None,
                time_limit=None) -> tuple[str, str]:
        """(the new version's id, the tool text)."""
        base_md = self.get(base_version).md  # once: an edited base may be rebuilt on each access
        md = self._apply(base_md, changes)
        if _same_model(md, base_md):
            return base_version, (f"no change: the {len(changes)} changes leave {base_version} as it is; no version "
                                  f"created, nothing solved")
        vid = self._new_version(md, base_version, description, solver, changes=changes)
        r = self.solved(vid, self._limit(time_limit, self.time_limit) if time_limit else None)
        lines = [f"created {vid} from {base_version} ({len(changes)} changes): status {r.status}{_proof(r)}"]
        if self.has_solution(vid):
            lines.append(f"objective: {_fmt(r.obj)}")
            base = self.solved(base_version)
            if self.has_solution(base_version):
                lines.append(f"objective change vs {base_version}: {_fmt(r.obj - base.obj)}{_change_range(base, r)}")
                lines += self._limit_use_lines(base_md, base, md, r, changes)
            elif base.status in ("INFEASIBLE", "INF_OR_UNBD") and explain_conflict:
                lines += self._fix_conflict_lines(base_md, md, changes, solver)
        return vid, "\n".join(lines)

    @staticmethod
    def _limit_use_lines(md0, r0, md1, r1, changes) -> list[str]:
        """For each limit a set_rhs change moved: the row's activity and slack before and after. When a MIP limit had
        slack left and loosening it still paid, say why: no whole-unit step that would improve the plan fits in that
        slack (the smallest such step this limit blocked)."""
        out = []
        for c in changes:
            if c.get("action") != "set_rhs" or c.get("name") not in md0.row_names or c["name"] not in md1.row_names:
                continue
            # the row's index in each model: another change in the same set may have dropped or added rows
            i, i1 = md0.row_index(c["name"]), md1.row_index(c["name"])
            a0, a1 = float((md0.A.getrow(i) @ r0.x)[0]), float((md1.A.getrow(i1) @ r1.x)[0])

            def use(md, k, a):
                lim = md.row_hi[k] if np.isfinite(md.row_hi[k]) else md.row_lo[k]
                return f"{a:.6g} of limit {lim:.6g} (slack {abs(lim - a):.6g})", abs(lim - a)

            before, slack0 = use(md0, i, a0)
            after, _ = use(md1, i1, a1)
            line = f"limit {c['name']}: activity {before} -> {after}"
            improved = (r1.obj - r0.obj) * (-1 if md0.minimize else 1) > 1e-9
            if md0.is_mip and slack0 > 1e-6 and improved:
                steps = [abs(float(md0.A[i, mv.col])) for mv in sensitivity.integer_unit_moves(md0, r0.x)
                         if mv.improving and i in mv.blocked_by]
                if steps:
                    line += (f"; the {slack0:.6g} left was too small for any whole-unit step that improves the plan "
                             f"(the smallest such step this limit blocked needs {min(steps):.6g})")
            out.append(line)
        if len(out) > CHANGES_SHOWN:
            out = out[:CHANGES_SHOWN] + [f"... ({len(out) - CHANGES_SHOWN} more limits changed)"]
        return out

    def try_options(self, base_version="v0", options=(), solver=None) -> str:
        """Several change sets, each applied to the base version as its own new version and solved (in parallel)."""
        from concurrent.futures import ThreadPoolExecutor

        base = self.get(base_version).md
        built = []  # every option checked before any version is made; only the last few keep their built model
        for n, opt in enumerate(options, 1):
            changes = opt.get("changes") or []
            md = self._apply(base, changes, f"option {n} ({opt.get('label', '')}), ")
            built.append((str(opt.get("label", "")), changes, _same_model(md, base),
                          md if n > len(options) - MODELS_KEPT else None))
            del md
        unchanged = [label for label, _, same, _ in built if same]
        made = [(self._new_version(md, base_version, label, solver, changes=changes), label)
                for label, changes, same, md in built if not same]
        del built
        with ThreadPoolExecutor(max_workers=parallel_solves(base)) as pool:
            list(pool.map(lambda m: self.solved(m[0]), made))
        base_ok = self.has_solution(base_version)
        lines = [f"{len(made)} options on {base_version}:"]
        lines += [f"  {label}: no change from {base_version} (not solved)" for label in unchanged]
        for vid, label in made:
            r = self.solved(vid)
            line = f"  {vid} {label}: {r.status}"
            if r.x is not None:
                line += f", objective {_fmt(r.obj)}"
                if base_ok:
                    base = self.solved(base_version)
                    line += f" (change {_fmt(r.obj - base.obj)}{_change_range(base, r)})"
            line += _proof(r)
            lines.append(line)
        return "\n".join(lines)

    def attainable_limit(self, constraints=(), version="v0", solver=None, time_limit=None) -> str:
        """For each constraint: drop it, then minimize and maximize its own expression. The range is what the rest
        of the model allows, so it gives the exact limit that constraint needs as the only change (and, called on
        the other side of a conflict, what that side needs)."""
        from concurrent.futures import ThreadPoolExecutor
        from dataclasses import replace

        md = self.get(version).md
        bk, lim = self._bk(md, solver), self._limit(time_limit, self.time_limit)
        names = list(dict.fromkeys(constraints))[:MAX_ATTAINABLE]

        def one(name):
            i = md.row_index(name)
            rest = md.drop_rows([name])
            coef = np.asarray(md.A[i].todense()).ravel()
            lo_r = bk.solve(replace(rest, obj=coef, minimize=True, obj_offset=0.0, Q=None), lim)
            hi_r = bk.solve(replace(rest, obj=coef, minimize=False, obj_offset=0.0, Q=None), lim)
            return name, i, lo_r, hi_r

        def end(r, word):
            if r.status in ("UNBOUNDED", "INF_OR_UNBD"):
                return f"unbounded {word}"
            if r.x is None:
                return f"unknown ({r.status})"
            return _fmt(r.obj) + ("" if r.status == "OPTIMAL" else f" (best found, {r.status}: not proven)")

        with ThreadPoolExecutor(max_workers=parallel_solves(md)) as pool:
            results = list(pool.map(one, names))
        lines = []
        for name, i, lo_r, hi_r in results:
            cur = _sense(md.row_lo[i], md.row_hi[i])
            if "INFEASIBLE" in (lo_r.status, hi_r.status):
                lines.append(f"{name} ({cur}): still infeasible with this row dropped, so no limit on it alone fixes the model")
                continue
            line = f"{name} ({cur}): with this row dropped it can range from {end(lo_r, 'below')} to {end(hi_r, 'above')}"
            need = []
            if np.isfinite(md.row_lo[i]) and hi_r.status == "OPTIMAL" and md.row_lo[i] > hi_r.obj + 1e-9:
                need.append(f"lower limit {_fmt(md.row_lo[i])} -> at most {_fmt(hi_r.obj)}")
            if np.isfinite(md.row_hi[i]) and lo_r.status == "OPTIMAL" and md.row_hi[i] < lo_r.obj - 1e-9:
                need.append(f"upper limit {_fmt(md.row_hi[i])} -> at least {_fmt(lo_r.obj)}")
            line += ("; as the only change: " + "; ".join(need)) if need else "; its current limits are within that range"
            lines.append(line)
        if len(set(constraints)) > MAX_ATTAINABLE:
            lines.append(f"(first {MAX_ATTAINABLE} constraints only)")
        return "\n".join(lines)

    def drop_test(self, version="v0", families=(), solver=None, time_limit=None) -> str:
        """For each constraint family: drop all its rows and solve. Which single family, removed, restores
        feasibility, and at what objective."""
        from concurrent.futures import ThreadPoolExecutor

        md = self.get(version).md
        bk, lim = self._bk(md, solver), self._limit(time_limit, self.time_limit)
        fams = list(dict.fromkeys(families)) or list(dict.fromkeys(base_name(n) for n in md.row_names))

        def one(fam):
            rows = [n for n in md.row_names if base_name(n) == fam]
            return fam, len(rows), (bk.solve(md.drop_rows(rows), lim) if rows else None)

        with ThreadPoolExecutor(max_workers=parallel_solves(md)) as pool:
            results = list(pool.map(one, fams))
        lines = []
        for fam, n, r in sorted(results, key=lambda t: (t[2] is None or t[2].x is None, t[0])):
            if r is None:
                lines.append(f"{fam}: no such family")
            elif r.x is not None:
                lines.append(f"{fam} ({n} rows): dropping it restores feasibility; objective {_fmt(r.obj)}"
                             + ("" if r.status == "OPTIMAL" else f" ({r.status})"))
            else:
                word = "still infeasible" if r.status == "INFEASIBLE" else f"unknown ({r.status})"
                lines.append(f"{fam} ({n} rows): {word} without it")
        return "\n".join(lines)

    def _fix_conflict_lines(self, base_md, md, changes, solver=None) -> list[str]:
        """For a fix that makes an infeasible version feasible: the conflict that sets each changed amount (the
        fix with that one change slightly smaller is infeasible again; its IIS), which can differ from the first
        IIS, since a model can hold several overlapping conflicts; or that the amount has slack."""
        names = list(dict.fromkeys(c["name"] for c in changes if c["action"] in ("set_rhs", "set_bounds")))
        if not names:
            return []
        res = od.fix_conflicts(base_md, md, names, self._bk(base_md, solver), budget=self._limit(None, FIX_CHECK_BUDGET),
                               time_limit=self._limit(None, IIS_SOLVE_LIMIT))
        lines = []
        for conf in res["conflicts"]:
            fams: dict[str, int] = {}
            for n in conf["rows"]:
                fams[base_name(n)] = fams.get(base_name(n), 0) + 1
            lines.append(f"the amount of {', '.join(conf['changes'])} is set by this conflict (with it slightly smaller the "
                         f"model is infeasible again; {len(conf['rows'])} rows: "
                         + ", ".join(f"{k}: {c}" for k, c in sorted(fams.items(), key=lambda kv: -kv[1])) + "), e.g.:")
            changed = set(conf["changes"])
            fam_changed = {base_name(n) for n in changed}
            rows = sorted(conf["rows"], key=lambda n: (n not in changed, base_name(n) not in fam_changed))
            lines += ["  " + l for l in _conflict_rows(base_md, rows, limit=8, per_family=2)]
        if res["slack"]:
            lines.append(f"has slack (still feasible with the change slightly smaller): {', '.join(res['slack'])}")
        if res["unchecked"]:
            lines.append(f"not checked (limit or time budget): {', '.join(res['unchecked'])}")
        return lines

    def suspicious_values(self, version="v0") -> str:
        md = self.get(version).md
        iis_rows: set[str] = set()
        v = self.get(version)
        if v.iis_result is not None:  # rank by an IIS already computed; never start a search here
            iis_rows = set(v.iis_result[1].rows)
        infeasible = self.solved(version).status in ("INFEASIBLE", "INF_OR_UNBD")
        flags = family_outliers(md, iis_rows, infeasible=infeasible)
        note = ("\n(infeasible: objective and missing-row checks skipped, coefficient checks shown only for IIS rows"
                + ("" if iis_rows else ", after compute_iis") + ")") if infeasible else ""
        if not flags:
            return "no values break their siblings' pattern" + note
        lines = [f"{len(flags)} values break their group's pattern"
                 + (" (IIS rows first):" if iis_rows else " (no IIS computed yet, so not ranked by it):")]
        for f in flags:
            where = f"{f['name']} / {f['column']}" if "column" in f else f["name"]
            value = "" if f["kind"] == "missing constraint" else f" = {_fmt(f['value'])}"
            lines.append(f"{'[in IIS] ' if f['in_iis'] else ''}{f['kind']} {where}{value}: "
                         f"{f['reason']} (group {f['group']})")
        if infeasible:
            lines += self._check_flag_fixes(md, flags)
        return "\n".join(lines) + note

    def _check_flag_fixes(self, md, flags: list[dict]) -> list[str]:
        """Solve the undo of the first flags (flag_fix): each alone, then the smallest sets that clear the model
        together (several typos can each leave it infeasible alone), pairs before triples, within FIX_CHECK_BUDGET.
        One undo per row: a row flagged for its sense also lands among the other sense's rows by shape, so its limit
        flag there follows from the flip and the flip is kept."""
        sense_rows = {f["name"] for f in flags if f["kind"] == "constraint sense"}
        fixes, seen = [], set()
        for f in flags:
            c = flag_fix(md, f)
            key = (c or {}).get("name"), (c or {}).get("column")
            if c and key not in seen and not (f["kind"] == "constraint limit" and f["name"] in sense_rows):
                seen.add(key)
                fixes.append(c)
        fixes = fixes[:FIX_CHECKS]
        if not fixes:
            return []
        bk, t0 = self._bk(md), time.time()
        lines = ["checked fixes (a flagged value put back, a sense flipped back with the same limit; solved):"]

        def solve(changes) -> str | None:
            left = self._limit(None, FIX_CHECK_BUDGET) - (time.time() - t0)
            if left <= 1:
                return None
            try:
                changed = self._apply(md, changes)
            except ValueError as e:
                return f"not checked ({str(e).splitlines()[-1].strip()})"
            r = bk.solve(changed, min(self.time_limit, left))
            return f"{r.status}, objective {_fmt(r.obj)}" if r.x is not None else f"still {r.status}"

        solved_alone = False
        for c in fixes:
            res = solve([c])
            lines.append(f"  {_change_text(md, c)}: {res or 'not checked (time budget used)'}")
            solved_alone |= bool(res) and not res.startswith(("still", "not"))
        if not solved_alone and len(fixes) > 1:
            for k in range(2, len(fixes) + 1):
                found = []
                for combo in combinations(fixes, k):
                    res = solve(list(combo))
                    if res is None:
                        lines.append(f"  sets of {k}: not all checked (time budget used)")
                        break
                    if not res.startswith(("still", "not")):
                        found.append((combo, res))
                for combo, res in found:
                    lines.append("  together: " + "; ".join(_change_text(md, c) for c in combo) + f": {res}"
                                 + "\n    changes: " + json.dumps(list(combo)))
                if found or res is None:
                    break
            else:
                lines.append(f"  no set of these {len(fixes)} makes the model solvable")
        return lines

    def fix_menu(self, version="v0", families=None, solver=None, time_limit=None) -> str:
        md = self.get(version).md
        if self.solved(version).status not in ("INFEASIBLE", "INF_OR_UNBD"):
            return f"version {version} is {self.solved(version).status}; fix_menu needs an infeasible version"
        bk, lim = self._bk(md, solver), self._limit(time_limit, self.time_limit)
        if families is None:  # the conflict's families (as od.fix_menu would pick them)
            families = self.iis(version, solver)[0] if md.is_mip else sorted({base_name(r) for r in self.iis(version, solver)[1].rows})
        from concurrent.futures import ThreadPoolExecutor

        # the screen and the relaxations share one time limit, so the whole menu returns within it
        start = time.monotonic()

        def screen(fam):
            rows = [n for n in md.row_names if base_name(n) == fam]
            return fam, (bk.solve(md.drop_rows(rows), max(1.0, lim / 3)).status if rows else "OTHER")

        with ThreadPoolExecutor(max_workers=parallel_solves(md)) as pool:
            screened = dict(pool.map(screen, families))
        # Dropping a whole family is the largest change it can make: if the model is still infeasible without it,
        # no change to that family alone fixes it, and its relaxation need not be run.
        hopeless = [f for f, st in screened.items() if st == "INFEASIBLE"]
        menu = od.fix_menu(md, bk, [f for f in families if f not in hopeless],
                           max(1.0, lim - (time.monotonic() - start)), min_objective=True)
        menu = menu[:-1] + [{"family": f, "structural": False, "sufficient": False, "status": "INFEASIBLE (still infeasible with the whole family dropped)",
                             "total_change": None, "changes": []} for f in hopeless] + menu[-1:]
        lines = []
        for e in menu:
            tag = " [likely linking, not a business lever]" if e["structural"] else ""
            if not e["sufficient"]:
                if e["status"] == "TIME_LIMIT":  # no feasible point in time is not proof that none exists
                    lines.append(f"{e['family']}{tag}: unknown (time limit reached before a feasible relaxation was found)")
                else:
                    lines.append(f"{e['family']}{tag}: NOT sufficient alone (relaxed model {e['status']})")
                continue
            lines.append(f"{e['family']}{tag}: sufficient; total change {_fmt(e['total_change'])} "
                         f"over {len(e['changes'])} bounds{' (time limit: may not be minimal)' if e['status'] == 'TIME_LIMIT' else ''}"
                         + (f"; best plan with it: objective {e['objective']:.10g}" if e.get("objective") is not None else ""))
            for r in e["changes"][:6]:
                lines.append(f"    {r['name']}: {_bound_label(r)} {_fmt(r['current'])} -> {_fmt(r['new'])} (minimal change {_fmt(r['value'])})")
            if len(e["changes"]) > 6:
                lines.append(f"    ... {len(e['changes']) - 6} more")
        lines.append(SAFE_NOTE)
        return "\n".join(lines)

    def read_model_document(self, query=None) -> str:
        if not self.doc_path:
            return "no model document is available for this model"
        doc = Path(self.doc_path).read_text()
        sections = re.split(r"(?m)^(?=#{1,6} )", doc)
        if query:
            hits = [sec for sec in sections if query.lower() in sec.lower()]
            if not hits:
                return f"no section mentions {query!r}; headings: " + "; ".join(_headings(doc))
            return "\n".join(hits)[:DOC_CHARS]
        if len(doc) <= DOC_CHARS:
            return doc
        return (f"the document is {len(doc)} characters; call again with a query. Headings:\n"
                + "\n".join(_headings(doc)))

    def why_not(self, variable, version="v0", value=None, lower=None, upper=None) -> str:
        if not self.has_solution(version):
            return f"version {version} has no solution to compare against (status {self.solved(version).status})"
        lb, ub = (value, value) if value is not None else (lower, upper)
        if lb is None and ub is None:
            raise ValueError("give value, or lower and/or upper")
        change = {"action": "set_bounds", "name": variable, "lower": lb, "upper": ub}
        vid, out = self._modify(version, f"why-not: {variable} in [{_fmt(lb)}, {_fmt(ub)}]", [change])
        if not self.has_solution(vid):
            return out + "\nthe forced alternative is infeasible; compute_iis on the new version shows what blocks it"
        return "\n".join([out, *_changes(self.get(version).md, self.solved(version).x,
                                          self.get(vid).md, self.solved(vid).x, forced=variable)])

    def compare_versions(self, version_a="v0", version_b=None, family=None) -> str:
        """What differs between two solved versions: objective, and per family the decisions and limits that changed."""
        version_b = version_b or list(self.versions)[-1]  # the latest added (an added model has a name, not vN)
        ra, rb = self.solved(version_a), self.solved(version_b)
        head = [f"{version_a}: {ra.status}, objective {_fmt(ra.obj)}{_proof(ra)}; {version_b}: {rb.status}, objective "
                f"{_fmt(rb.obj)}{_proof(rb)}"
                + (f"; change {_fmt(rb.obj - ra.obj)}{_change_range(ra, rb)}" if ra.obj is not None and rb.obj is not None else "")]
        if not (self.has_solution(version_a) and self.has_solution(version_b)):
            return head[0] + "\nboth versions need a solution to compare decisions"
        ma, mb = self.get(version_a).md, self.get(version_b).md
        if ma.col_names != mb.col_names:
            return head[0] + "\nthe versions have different variables (a row or column was added or dropped); compare by family with query_variables"
        x_b, note = self._closest_plan(version_a, version_b)
        head += note
        fam = lambda n: base_name(n)  # noqa: E731
        head.append("per variable family (total and nonzero count, before -> after; only families that changed):")
        for f in sorted({fam(n) for n in ma.col_names}):
            if family and f != family:
                continue
            j = np.array([k for k, n in enumerate(ma.col_names) if fam(n) == f])
            xa, xb = ra.x[j], x_b[j]
            if np.allclose(xa, xb, atol=1e-6):
                continue
            head.append(f"  {f}: total {_fmt(float(xa.sum()))} -> {_fmt(float(xb.sum()))}, nonzero "
                        f"{int((np.abs(xa) > 1e-6).sum())} -> {int((np.abs(xb) > 1e-6).sum())} of {len(j)}")
        return "\n".join(head + _changes(ma, ra.x, mb, x_b, family=family))

    def _closest_plan(self, version_a: str, version_b: str) -> tuple[np.ndarray, list[str]]:
        """version_b's plan to diff against version_a: its optimal plan nearest version_a's (one more solve, cached),
        when that moves fewer variables than the solver's own plan; else the solver's plan. With the note to show."""
        key = (version_a, version_b)
        if key not in self._closest:
            ra, rb, mb = self.solved(version_a), self.solved(version_b), self.get(version_b).md
            status, xc = closest.closest_optimal(mb, ra.x, rb.obj, self._bk(mb, self.get(version_b).solver),
                                                 self.time_limit)
            self._closest[key] = (status, xc)
        status, xc = self._closest[key]
        x_b = self.solved(version_b).x
        if xc is None:
            return x_b, []
        x_a = self.solved(version_a).x
        n_solver, n_closest = int(closest.moved(x_a, x_b).sum()), int(closest.moved(x_a, xc).sum())
        if n_closest >= n_solver:
            return x_b, []
        which = ("the closest" if status == "OPTIMAL"
                 else "a nearer (the closest-plan search hit its time limit)")
        return xc, [f"plan compared: {which} optimal plan of {version_b} to {version_a}'s plan (same objective); it "
                    f"changes {n_closest} variable{'s' * (n_closest != 1)}. The solver's own plan changes {n_solver}: the other "
                    f"{n_solver - n_closest} are a choice among equal-cost plans, not an effect of the difference"]

    def add_model(self, path, name, description=None) -> str:
        """Another model (a file path or ModelData) in this session under a name, next to v0."""
        if not re.fullmatch(r"[A-Za-z_][\w.-]*", str(name)) or re.fullmatch(r"v\d+", str(name)):
            raise ValueError(f"name {name!r}: use a short word such as min_cost, not v0, v1, ...")
        if name in self.versions:
            raise ValueError(f"{name!r} is already open; existing: {', '.join(self.versions)}")
        md = path if isinstance(path, od.ModelData) else od.load(path)  # loading runs outside the lock
        v0 = self.versions.get("v0")
        same = v0 is not None and (md.is_mip, md.Q is None) == (v0.md.is_mip, v0.md.Q is None)
        # the session's solver was chosen for v0's kind of model; a different kind gets its own
        licensed = (self.route or self.prefer) in od.BACKENDS and od.BACKENDS[self.route or self.prefer].licensed
        solver = None if same or v0 is None or licensed \
            else "highs" if not md.is_mip else _backend(None, md, None, self.default_solver).name
        self._new_version(md, None, description or f"model {getattr(path, 'name', Path(str(path)).name)}", solver,
                          name=name)
        if not isinstance(path, od.ModelData):
            self._sources[name] = str(Path(path).resolve())
        shared = v0 is not None and md.col_names == v0.md.col_names and md.row_names == v0.md.row_names
        return (f"added {name}: {md.num_rows} rows, {md.num_cols} columns"
                f"{' (' + str(int(md.is_int.sum())) + ' integer)' if md.is_mip else ''}; "
                + ("same rows and columns as v0 (compare_versions works across them)" if shared else
                   "different rows or columns from v0 (compare with compare_models)"))

    def compare_models(self, models=None) -> str:
        """Status, objective and the same per-family metrics for each model, as one table."""
        names = list(models) if models else [k for k, v in self.versions.items() if v.parent is None]
        res = {n: self.solved(n) for n in names}
        rows = [["status"] + [res[n].status for n in names],
                ["objective"] + [_fmt(res[n].obj) for n in names]]
        cfam: dict[str, dict] = {}
        vfam: dict[str, dict] = {}
        for n in names:
            md, r = self.get(n).md, res[n]
            if r.x is None:
                continue
            act = md.A @ r.x
            for i, rn in enumerate(md.row_names):
                at = any(np.isfinite(b) and abs(act[i] - b) <= 1e-6 * (1 + abs(b)) for b in (md.row_lo[i], md.row_hi[i]))
                c = cfam.setdefault(base_name(rn), {}).setdefault(n, [0, 0, None])
                c[0] += 1
                c[1] += bool(at)
                c[2] = (float(act[i]), _sense(md.row_lo[i], md.row_hi[i]))  # kept for one-row families
            for j, cn in enumerate(md.col_names):
                v = vfam.setdefault(base_name(cn), {}).setdefault(n, [0.0, 0, 0])
                v[0] += float(r.x[j])
                v[1] += abs(r.x[j]) > 1e-6
                v[2] += 1
        for f in sorted(cfam):
            if all(c[0] == 1 for c in cfam[f].values()):  # a single row (a budget, a total): its value, not a count
                rows.append([f"{f} (value; limit)"] + [f"{_fmt(cfam[f][n][2][0])}; {cfam[f][n][2][1]}"
                                                      + (" (at the limit)" if cfam[f][n][1] else "")
                                                      if n in cfam[f] else "-" for n in names])
                continue
            rows.append([f"{f} (rows at a limit)"] + [f"{cfam[f][n][1]} of {cfam[f][n][0]}" if n in cfam[f] else "-"
                                                     for n in names])
        for f in sorted(vfam):
            rows.append([f"{f} (total; nonzero)"] + [f"{_fmt(vfam[f][n][0])}; {vfam[f][n][1]} of {vfam[f][n][2]}"
                                                    if n in vfam[f] else "-" for n in names])
        lines = ["| | " + " | ".join(f"{n} ({self.get(n).description})" for n in names) + " |",
                 "|---|" + "---|" * len(names)] + ["| " + " | ".join(r) + " |" for r in rows]
        if any(res[n].x is None for n in names):
            lines.append("models without a solution show no family metrics (compute_iis explains an infeasible one)")
        return "\n".join(lines)


    def marginal_value(self, constraint, version="v0", delta=1.0) -> str:
        v = self.get(version)
        md = v.md
        # the session's solver for all three solves (the default is SCIP for a MIP, which does not finish at 837k rows)
        mv = sensitivity.marginal_value(md, constraint, float(delta), self.time_limit, base=self.solved(version),
                                        backend=self._bk(md, v.solver))
        if mv["status"] != "OPTIMAL":
            return f"version {version} is {mv['status']}; marginal value needs an optimal solution"
        lines = [f"{constraint}: active {mv['side']} bound {_fmt(mv['bound'])}, activity {_fmt(mv['activity'])}, objective {_fmt(mv['objective'])}"]
        for sign, way in ((1, "up"), (-1, "down")):
            line = f"bound {_fmt(mv['bound'])} -> {_fmt(mv['bound'] + sign * float(delta))}: {mv[way + '_status']}"
            if mv[f"per_unit_{way}"] is not None:  # the plain change first: the per-unit rate has the opposite sign for a lowered bound
                change = mv[way + "_objective"] - mv["objective"]
                line += (f", objective {_fmt(mv[way + '_objective'])} (change {f'{change:+.15g}' if change else '0'}; "
                         f"{_fmt(mv['per_unit_' + way])} per unit the bound rises)")
            lines.append(line)
        if mv["per_unit_up"] is not None and mv["per_unit_down"] is not None and \
                abs(mv["per_unit_up"] - mv["per_unit_down"]) > 1e-4 * max(1.0, abs(mv["per_unit_up"]), abs(mv["per_unit_down"])):
            lines.append("up and down values differ: the optimum is at a kink (degenerate or integer effect); no single shadow price describes it")
        return "\n".join(lines)


    def _mip_sensitivity(self, md, s, kind, family, name_contains) -> str:
        moves = sensitivity.integer_unit_moves(md, s.x)
        caveat = ("MIP: integer-aware view from +/-1 moves of each integer variable (single-variable moves "
                  "only; a better plan needing a swap is not detected). Shadow prices are from the plan with "
                  "integers fixed: valid only if the integer decisions do not change; confirm with marginal_value.")
        if kind == "constraints":
            ib = sensitivity.integer_binding(md, s.x, moves)
            idx = self._match(md.row_names, family, name_contains)
            idx = sorted(idx, key=lambda i: (-ib["sole_blocker"][i], -ib["blocks_improving"][i],
                                             -abs(s.shadow_price[i])))[:ROW_LIMIT]
            lines = [caveat, f"{len(idx)} constraints (most limiting first):"]
            for i in idx:
                lines.append(f"{md.row_names[i]}: {_sense(md.row_lo[i], md.row_hi[i])}, integer-binding "
                             f"{bool(ib['effective_binding'][i])}, blocks {ib['blocks_improving'][i]} improving unit "
                             f"moves ({ib['sole_blocker'][i]} alone), fixed-plan shadow price {_fmt(s.shadow_price[i])}")
            return "\n".join(lines)
        by_col: dict[int, list] = {}
        for mv in moves:
            by_col.setdefault(mv.col, []).append(mv)
        idx = self._match(md.col_names, family, name_contains)
        lines = [caveat, f"{len(idx)} variables match" + (f"; showing first {ROW_LIMIT}" if len(idx) > ROW_LIMIT else "") + ":"]
        for j in idx[:ROW_LIMIT]:
            if not md.is_int[j]:
                lines.append(f"{md.col_names[j]}: continuous, value {_fmt(s.x[j])}, reduced cost (fixed plan) {_fmt(s.reduced_cost[j])}")
                continue
            parts = []
            for mv in by_col.get(j, []):
                if mv.improving:
                    who = ", ".join(md.row_names[i] for i in mv.blocked_by[:4]) or "nothing (feasible improving move: plan is not optimal or tolerance)"
                    parts.append(f"{mv.step:+d} would improve, blocked by {who}")
            lines.append(f"{md.col_names[j]}: integer, value {_fmt(s.x[j])}; "
                         + ("; ".join(parts) if parts else "no improving +/-1 move"))
        return "\n".join(lines)

    def sensitivity_report(self, kind, version="v0", family=None, name_contains=None) -> str:
        v = self.get(version)
        md = v.md
        s = sensitivity.sensitivity(md, self.time_limit, backend=self._bk(md, v.solver))
        if s.status != "OPTIMAL":
            return f"no sensitivity: status {s.status}"
        if md.is_mip:
            return self._mip_sensitivity(md, s, kind, family, name_contains)
        note = "RHS ranges unavailable for QP; " if s.rhs_range is None else ""
        if kind == "constraints":
            idx = self._match(md.row_names, family, name_contains)
            idx = sorted(idx, key=lambda i: -abs(s.shadow_price[i]))[:ROW_LIMIT]
            lines = [f"{note}{len(idx)} constraints (largest |shadow price| first; a range is where this solution's "
                     "shadow price holds, and at a degenerate solution it can differ on each side: marginal_value "
                     "re-solves to check a change):"]
            for i in idx:
                line = f"{md.row_names[i]}: {_sense(md.row_lo[i], md.row_hi[i])}, shadow price {_fmt(s.shadow_price[i])}"
                if s.rhs_range is not None:
                    line += f", valid for RHS in [{_fmt(s.rhs_range[0][i])}, {_fmt(s.rhs_range[1][i])}]"
                lines.append(line)
        else:
            idx = self._match(md.col_names, family, name_contains)
            idx = sorted(idx, key=lambda j: -abs(s.reduced_cost[j]))[:ROW_LIMIT]
            lines = [f"{note}{len(idx)} variables (largest |reduced cost| first):"]
            for j in idx:
                line = f"{md.col_names[j]}: value {_fmt(s.x[j])}, reduced cost {_fmt(s.reduced_cost[j])}"
                if s.cost_range is not None:
                    line += (f", objective coefficient {_fmt(md.obj[j])} keeps this solution while it stays between "
                             f"{_fmt(s.cost_range[0][j])} and {_fmt(s.cost_range[1][j])}")
                lines.append(line)
        return "\n".join(lines)


def _changes(md: od.ModelData, x0: np.ndarray, md1: od.ModelData, x1: np.ndarray, forced: str | None = None,
             family: str | None = None) -> list[str]:
    """Rows that became binding or slack, and the variables that changed: the count, a per-family tally, the
    changes sharing each index value of a forced variable, and the largest changes. With 0/1 variables every
    change has the same size, so a top-15 list alone is arbitrary and reads as the total."""
    lines = []
    b0, b1 = _binding(md, x0), _binding(md1, x1)
    if md1.row_names != md.row_names:  # a dropped or added constraint: compare the rows both versions have
        pos1 = {n: i for i, n in enumerate(md1.row_names)}
        b1 = np.array([n in pos1 and bool(b1[pos1[n]]) for n in md.row_names], dtype=bool)
        b0 &= np.array([n in pos1 for n in md.row_names], dtype=bool)
        names0 = set(md.row_names)
        for label, names in (("rows only in the first version", [n for n in md.row_names if n not in pos1]),
                             ("rows only in the second version", [n for n in md1.row_names if n not in names0])):
            if names:
                lines.append(f"{label}: {len(names)}; e.g. " + ", ".join(names[:8]))
    for label, mask in (("became binding", ~b0 & b1), ("became slack", b0 & ~b1)):
        names = [md.row_names[i] for i in np.flatnonzero(mask)]
        fams: dict[str, int] = {}
        for n in names:
            fams[base_name(n)] = fams.get(base_name(n), 0) + 1
        lines.append(f"{label}: {len(names)}" + (" (" + ", ".join(f"{k}: {c}" for k, c in sorted(fams.items(), key=lambda kv: -kv[1])) + "); e.g. " + ", ".join(names[:8]) if names else ""))
    changed = np.flatnonzero(np.abs(x1 - x0) > 1e-6 * np.maximum(1.0, np.abs(x0)))
    if family:
        changed = np.array([j for j in changed if base_name(md.col_names[j]) == family], dtype=int)
    tally: dict[str, list[int]] = {}
    for j in changed:
        tally.setdefault(base_name(md.col_names[j]), [0, 0])[0 if x1[j] > x0[j] else 1] += 1
    top = sorted(tally.items(), key=lambda kv: -sum(kv[1]))[:8]
    lines.append(f"variables changed: {len(changed)} of {md.num_cols} ("
                 + ", ".join(f"{f}: {u} up, {d} down" for f, (u, d) in top) + ")")
    if forced:
        idx, fam = _indices(forced), base_name(forced)
        for p, v in enumerate(idx or []):
            same = [j for j in changed if md.col_names[j] != forced and base_name(md.col_names[j]) == fam
                    and len(ix := _indices(md.col_names[j]) or []) == len(idx) and ix[p] == v]
            if same:
                u = sum(1 for j in same if x1[j] > x0[j])
                lines.append(f"  {fam} with index {p + 1} = {v}: {u} up, {len(same) - u} down")
    pattern_lines, shown = _patterns(md, x0, x1, changed, family)
    lines += pattern_lines
    rest = np.array([j for j in changed if base_name(md.col_names[j]) not in shown], dtype=int)
    if not len(rest):
        return lines
    moved = rest[np.argsort(-np.abs(x1[rest] - x0[rest]))][:15]
    lines.append(f"largest variable changes{' in the other families' if shown else ''} (showing {len(moved)} of {len(rest)}):")
    lines += [f"  {md.col_names[j]}: {_fmt(x0[j])} -> {_fmt(x1[j])}" for j in moved]
    return lines


PATTERN_COLS = 80  # families up to this size are shown as before -> after sequences, as a planner reads a schedule


def _patterns(md: od.ModelData, x0: np.ndarray, x1: np.ndarray, changed: np.ndarray,
              family: str | None) -> tuple[list[str], set[str]]:
    """Small indexed families that changed, as before -> after sequences over their last index, one line per value
    of the other indices: O/. for 0/1 variables (a schedule), numbers otherwise; and the families shown."""
    fams: dict[str, list[int]] = {}
    for j, n in enumerate(md.col_names):
        fams.setdefault(base_name(n), []).append(j)
    touched = {base_name(md.col_names[j]) for j in changed}
    out, shown = [], set()
    for fam in sorted(touched):
        cols = fams[fam]
        if (family and fam != family) or len(cols) > PATTERN_COLS:
            continue
        parts = {j: _indices(md.col_names[j]) for j in cols}
        if any(p is None for p in parts.values()):
            continue
        # one index like 1day_3 holds two (product, day) when every name splits the same way and ends in a number
        split = {j: p[0].split("_") for j, p in parts.items()} if all(len(p) == 1 for p in parts.values()) else None
        if split and len({len(s) for s in split.values()}) == 1 and all(len(s) > 1 and s[-1].isdigit() for s in split.values()):
            parts = split
        groups: dict[tuple, dict[str, int]] = {}
        for j in cols:
            groups.setdefault(tuple(parts[j][:-1]), {})[parts[j][-1]] = j
        positions = list(dict.fromkeys(parts[j][-1] for j in cols))
        if all(q.lstrip("-").isdigit() for q in positions):
            positions.sort(key=int)
        vals = np.concatenate([x0[cols], x1[cols]])
        binary = bool(np.all(np.isin(np.round(vals, 6), (0.0, 1.0))))

        def seq(x, at: dict[str, int]) -> str:
            cells = [("-" if q not in at else "O" if x[at[q]] > 0.5 else ".") if binary else
                     ("-" if q not in at else f"{x[at[q]] + 0.0:.6g}") for q in positions]  # + 0.0: no "-0"
            return ("" if binary else " ").join(cells)

        out.append(f"{fam} over its last index ({', '.join(positions)}), before -> after"
                   + (" (O = 1, . = 0, - = none):" if binary else ":"))
        for g, at in groups.items():
            a, b = seq(x0, at), seq(x1, at)
            if a != b:
                out.append(f"  {' '.join(g) or fam}: {a} -> {b}")
        shown.add(fam)
    return out, shown


def _binding(md: od.ModelData, x: np.ndarray) -> np.ndarray:
    act = md.A @ x
    return (np.abs(act - md.row_hi) <= 1e-6) | (np.abs(act - md.row_lo) <= 1e-6)



def _headings(doc: str) -> list[str]:
    return [line.strip() for line in doc.splitlines() if re.match(r"#{1,6} ", line)]


def _limit(change: dict, side: str) -> float | None:
    """A change's new limit: absent = unchanged, null = no limit (e.g. restoring a flipped sense)."""
    if side not in change:
        return None
    v = change[side]
    return (-od.INF if side == "lower" else od.INF) if v is None else v


class _UnknownName(ValueError):
    def __init__(self, name, names, kind: str):
        super().__init__(f"no {kind} named {name!r}")
        self.name, self.names = str(name), names


def _suggestion(e: Exception) -> str:
    """Close matches for an unknown name; on a large model only among names with the same first three characters."""
    if not isinstance(e, _UnknownName):
        return ""
    pool = e.names if len(e.names) <= 5000 else [n for n in e.names if n[:3] == e.name[:3]]
    close = difflib.get_close_matches(e.name, pool, n=3, cutoff=0.6)
    return f"; did you mean {', '.join(close)}?" if close else ""


def _check_name(name, names, kind: str) -> None:
    if name not in names:
        raise _UnknownName(name, names, kind)


def _number(change: dict, key: str, allow_none: bool = False) -> None:
    v = change.get(key)
    if v is None and allow_none:
        return
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != v:
        raise ValueError(f"{key} must be a number{' or null' if allow_none else ''}, got {v!r}")


def _apply_batch(md, batch: list[dict]):
    """Valid set_rhs changes, or valid set_objective_coef changes, applied in one copy (the last change to a name
    wins, as applying them one by one would)."""
    from dataclasses import replace

    if batch[0]["action"] == "set_rhs":
        index = {n: i for i, n in enumerate(md.row_names)}
        lo, hi = md.row_lo.copy(), md.row_hi.copy()
        for c in batch:
            i = index[c["name"]]
            if "lower" in c:
                lo[i] = _limit(c, "lower")
            if "upper" in c:
                hi[i] = _limit(c, "upper")
        return replace(md, row_lo=lo, row_hi=hi)
    index = {n: i for i, n in enumerate(md.col_names)}
    obj = md.obj.copy()
    for c in batch:
        obj[index[c["name"]]] = float(c["value"])
    return replace(md, obj=obj)


def _apply_one(md, c: dict):
    action, name = c.get("action"), c.get("name")
    for side in ("lower", "upper"):
        if side in c:
            _number(c, side, allow_none=True)
    if action == "set_rhs":
        _check_name(name, md.row_names, "constraint")
        return md.set_row_bounds(name, lo=_limit(c, "lower"), hi=_limit(c, "upper"))
    if action == "set_bounds":
        _check_name(name, md.col_names, "variable")
        md = md.set_col_bounds(name, lb=_limit(c, "lower"), ub=_limit(c, "upper"))
        if md.is_int[md.col_index(name)]:
            for side in ("lower", "upper"):
                v = c.get(side)
                if v is not None and np.isfinite(v) and v != int(v):
                    raise ValueError(f"{name} is an integer variable; {side} {_fmt(v)} is not a whole number")
        return md
    if action == "set_objective_coef":
        _check_name(name, md.col_names, "variable")
        _number(c, "value")
        return md.set_obj_coef(name, c["value"])
    if action == "drop_constraint":
        _check_name(name, md.row_names, "constraint")
        return md.drop_rows([name])
    if action == "set_coef":
        _check_name(name, md.row_names, "constraint")
        _check_name(c.get("column"), md.col_names, "variable")
        _number(c, "value")
        return md.set_coef(name, c["column"], c["value"])
    if action == "add_constraint":
        if name in md.row_names:
            raise ValueError(f"constraint {name!r} already exists")
        coefs = c.get("coefs") or {}
        for col, v in coefs.items():
            _check_name(col, md.col_names, "variable")
            _number(coefs, col)
        lo, hi = _limit(c, "lower"), _limit(c, "upper")
        return md.add_row(name, coefs, -od.INF if lo is None else lo, od.INF if hi is None else hi)
    raise ValueError(f"unknown action {action!r}")


def _plain_change(c: dict) -> str:
    """One change as stored, without the model's old values (version_history; the model may not be built)."""
    a, n = c.get("action"), c.get("name")
    side = ", ".join(f"{k} {_fmt(c[k]) if c[k] is not None else 'none'}" for k in ("lower", "upper") if k in c)
    if a in ("set_rhs", "set_bounds"):
        return f"{'limit' if a == 'set_rhs' else 'bounds'} of {n}: {side}"
    if a == "drop_constraint":
        return f"dropped {n}"
    if a == "add_constraint":
        return f"added {n} ({len(c.get('coefs') or {})} terms; {side})"
    if a == "set_coef":
        return f"coefficient of {c.get('column')} in {n}: {_fmt(c.get('value'))}"
    if a == "set_objective_coef":
        return f"objective coefficient of {n}: {_fmt(c.get('value'))}"
    return json.dumps(c)


def _same_model(a, b) -> bool:
    """The edits left the model identical (limits, bounds, objective, coefficients, rows)."""
    return (a.row_names == b.row_names and a.A.shape == b.A.shape and (a.A != b.A).nnz == 0
            and all(np.array_equal(x, y) for x, y in ((a.row_lo, b.row_lo), (a.row_hi, b.row_hi),
                                                       (a.col_lb, b.col_lb), (a.col_ub, b.col_ub), (a.obj, b.obj))))
