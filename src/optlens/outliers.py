"""Suspicious values: constraint limits and variable bounds that break the pattern of their siblings.

Data errors (a typo in a right-hand side or a mistyped bound) often look
out of place next to the rows or variables they belong with. Groups, tried in order for each row:
the family (name without its index), the name with digits removed (ROW00087 -> ROW), and the name
prefixes of 1-3 underscore segments (Octane_tol_premium_fuel ~ Octane_tol_regular_fuel). Flags are hints
for a human or the agent to confirm against the model document, not verdicts. On our test cases, restricted
to IIS rows, the injected RHS typo ranks first in about half the LP cases with ~0.6 other flags per case;
a within-row coefficient rule was tried and dropped (75% of false alarms, no hits).
A row whose sense differs from every other row of its group (<= among >=) is flagged as a possible flipped sense.
Sense groups also work without names: rows of the same shape apart from their sense, and runs of consecutive rows
whose coefficients share one sign (an anonymous model keeps its author's row order, so a block of caps written
together stays together). When several groups flag a row, the smallest group's reason is kept: "36 of 37 rows of
the same shape" is stronger evidence than "84 of 105 rows named ROW#".
"""
from __future__ import annotations

import re

import numpy as np

from .diagnose import base_name
from .model import ModelData
from .structure import names_are_meaningful, row_classes, row_signatures

MODE_SHARE = 0.75   # a value shared by this share of a group is its pattern; other values are flagged
NEAR_MISS = 0.2     # three members, two equal: the third is flagged when within this share of them (0.99 among 1s)
ROBUST_Z = 6.0      # otherwise, flag values this many MADs from the group median
LARGE_FACTOR = 10.0  # model-wide: a limit this many times the 95th percentile of the other limits
SENSE_MIN_GROUP = 4  # a row whose sense differs from every other row in a group this large or larger is flagged
PREFIX_MAX = 20     # name-prefix groups larger than this are too loose to define a pattern
WHOLE_SHARE, WHOLE_MIN = 0.9, 20  # a name group holding this share of a model's many names is the model, not a family


def _groups(names: tuple[str, ...]) -> list[tuple[str, list[int]]]:
    keys = [lambda n: base_name(n), lambda n: re.sub(r"\d+", "#", n)]
    for k in (1, 2, 3):  # shared name prefixes: Octane_tol_premium_fuel ~ Octane_tol_regular_fuel
        keys.append(lambda n, k=k: "_".join(n.split("_")[:k]) + "_*" if n.count("_") >= k else None)
    out, seen = [], set()
    for key in keys:
        g: dict[str, list[int]] = {}
        for i, n in enumerate(names):
            kv = key(n)
            if kv is not None:
                g.setdefault(kv, []).append(i)
        for k, idx in g.items():
            t = tuple(idx)
            if len(idx) >= 2 and t not in seen and (len(idx) <= PREFIX_MAX or not k.endswith("_*")):
                seen.add(t)
                out.append((k, idx))
    return out


def _flag_group(values: np.ndarray, idx: list[int], n_all: int = 0) -> list[tuple[int, str, float]]:
    """(member index, reason, typical value) for members breaking the group's pattern. A group holding nearly all
    n_all names (B1..B174: the whole model, not a family) is only read for a shared value, not for spread."""
    vals = values[idx]
    ok = np.isfinite(vals)
    if ok.sum() < 2:
        return []
    v = vals[ok]
    members = [i for i, f in zip(idx, ok) if f]
    uniq, counts = np.unique(v, return_counts=True)
    top = counts.argmax()
    flags = []
    repeated = set(uniq[counts >= 2])  # a value shared by several members is itself a pattern, not a typo
    if counts[top] >= max(2, MODE_SHARE * len(v)) and counts[top] < len(v):
        for i, x in zip(members, v):
            if x != uniq[top] and x not in repeated:
                flags.append((i, f"differs from its {counts[top]} of {len(v)} siblings at {uniq[top]:.15g}", float(uniq[top])))
        return flags
    if len(v) == 3 and counts[top] == 2:  # too small for a share: only a near miss or a power-of-ten slip of the pair
        k = int(np.flatnonzero(v != uniq[top])[0])
        t = float(uniq[top])
        if t != 0 and (abs(v[k] - t) <= NEAR_MISS * abs(t) or _is_typo(float(v[k]), t)):
            return [(members[k], f"differs from its 2 siblings at {t:.15g}", t)]
        return []
    if len(v) == 2 and (v == 0).sum() == 1:  # two siblings, one zero: ambiguous, report the nonzero one
        i = members[int(np.flatnonzero(v != 0)[0])]
        return [(i, "nonzero where its only sibling is 0", 0.0)]
    if len(idx) >= WHOLE_MIN and len(idx) >= WHOLE_SHARE * n_all:
        return flags
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    if mad > 0:
        for i, x in zip(members, v):
            if abs(x - med) > ROBUST_Z * mad and x not in repeated:
                flags.append((i, f"{abs(x - med) / mad:.0f} MADs from its group median {med:.15g}", med))
    return flags


def _senses(md: ModelData) -> np.ndarray:
    """Row sense as text: "<=", ">=", "=" or "range"."""
    lo_inf, hi_inf = np.isinf(md.row_lo), np.isinf(md.row_hi)
    return np.where(lo_inf & ~hi_inf, "<=", np.where(hi_inf & ~lo_inf, ">=", np.where(md.row_lo == md.row_hi, "=", "range")))


def _flag_senses(sense: np.ndarray, idx: list[int], flips_only: bool = False) -> list[tuple[int, str]]:
    """(member index, reason) for the one row whose sense differs from all the others in a group. flips_only: only
    <= against >= (a typo swaps those; an equality among inequalities is usually meant)."""
    if len(idx) < SENSE_MIN_GROUP:
        return []
    vals, counts = np.unique(sense[idx], return_counts=True)
    if len(vals) != 2 or counts.min() != 1:
        return []  # a flip is a lone row; several dissenters are a pattern (or mixed constraint types)
    odd, common = vals[counts.argmin()], vals[counts.argmax()]
    if flips_only and {odd, common} != {"<=", ">="}:
        return []
    i = idx[int(np.flatnonzero(sense[idx] == odd)[0])]
    return [(i, f"sense {odd} where its {len(idx) - 1} siblings are all {common}")]


def _sense_free_groups(md: ModelData) -> list[tuple[str, list[int]]]:
    """Name-free groups for the sense check: rows alike in everything but their sense (term count and sorted
    coefficients), and maximal runs of consecutive rows whose coefficients all have the same sign."""
    shape: dict[tuple, list[int]] = {}
    for i, sig in enumerate(row_signatures(md)):
        shape.setdefault(sig[1:], []).append(i)
    out = [(f"shape without sense ({k[0]} terms)", idx) for k, idx in shape.items() if len(idx) >= SENSE_MIN_GROUP]
    A = md.A.tocsr()
    sign = [np.sign(A.data[A.indptr[i]:A.indptr[i + 1]]) for i in range(md.num_rows)]
    one_sign = [int(s[0]) if len(s) and (s == s[0]).all() else 0 for s in sign]
    start = 0
    for i in range(1, md.num_rows + 1):
        if i == md.num_rows or one_sign[i] != one_sign[start] or one_sign[i] == 0:
            if one_sign[start] != 0 and i - start >= SENSE_MIN_GROUP:
                out.append((f"rows {md.row_names[start]}..{md.row_names[i - 1]} (consecutive, one coefficient sign)",
                            list(range(start, i))))
            start = i
    return out


def _active_limit(md: ModelData) -> np.ndarray:
    """The finite side of each row (the upper one for ranged/equality rows)."""
    return np.where(np.isfinite(md.row_hi), md.row_hi, md.row_lo)


def _reach(md: ModelData) -> np.ndarray:
    """Largest |activity| each row's terms can reach within the variable bounds (inf when a term is unbounded)."""
    big = np.maximum(np.abs(md.col_lb), np.abs(md.col_ub))
    A = abs(md.A.tocsr())
    out = np.zeros(md.num_rows)
    for i in range(md.num_rows):
        cols = A.indices[A.indptr[i]:A.indptr[i + 1]]
        out[i] = float(A.data[A.indptr[i]:A.indptr[i + 1]] @ big[cols]) if np.isfinite(big[cols]).all() else np.inf
    return out


def family_outliers(md: ModelData, iis_rows: set[str] | None = None, max_flags: int = 40,
                    infeasible: bool = False) -> list[dict]:
    """infeasible: the model has no solution, so only flags that can cause that are kept (see pattern_breaks)."""
    iis_rows = iis_rows or set()
    flags: dict[tuple[str, str], dict] = {}

    def add(kind, name, value, reason, typical, group, size=None):
        key = (kind, name)
        if key not in flags or (size is not None and size < flags[key]["_size"]):  # the tightest group's reason
            flags[key] = {"kind": kind, "name": name, "value": float(value), "typical": typical,
                          "reason": reason, "group": group, "in_iis": name in iis_rows,
                          "_size": size if size is not None else float("inf")}

    rhs = _active_limit(md)
    groups = _groups(md.row_names) + list(row_classes(md).items())  # names and, name-free, row shape
    for g, idx in groups:
        for i, why, typ in _flag_group(rhs, idx, md.num_rows):
            add("constraint limit", md.row_names[i], rhs[i], why, typ, g, len(idx))
    sense = _senses(md)
    for g, idx in _groups(md.row_names):
        for i, why in _flag_senses(sense, idx):
            add("constraint sense", md.row_names[i], rhs[i], why, None, g, len(idx))
    if not names_are_meaningful(md):  # named models have their families; shape and file order stand in for them
        for g, idx in _sense_free_groups(md):
            for i, why in _flag_senses(sense, idx, flips_only=True):
                add("constraint sense", md.row_names[i], rhs[i], why, None, g, len(idx))
    for bound_name, bound in (("variable upper bound", md.col_ub), ("variable lower bound", md.col_lb)):
        for g, idx in _groups(md.col_names):
            for j, why, typ in _flag_group(bound, idx, md.num_cols):
                add(bound_name, md.col_names[j], bound[j], why, typ, g)

    nz = np.flatnonzero(np.isfinite(rhs) & (rhs != 0))
    if len(nz) >= 5:
        p95 = float(np.percentile(np.abs(rhs[nz]), 95))  # one row barely moves it; computed once
        reach = _reach(md)
        fam = [base_name(md.row_names[i]) for i in nz]
        by_fam: dict[str, list[float]] = {}
        for f, i in zip(fam, nz):
            by_fam.setdefault(f, []).append(abs(rhs[i]))
        top2 = {f: sorted(v, reverse=True)[:2] for f, v in by_fam.items()}  # once per family, not per row (279k rows)
        for f, i in zip(fam, nz):
            peers = top2[f]
            second = peers[1] if peers[0] == abs(rhs[i]) and len(peers) > 1 else peers[0] if peers[0] != abs(rhs[i]) else 0.0
            if second * 3 >= abs(rhs[i]):
                continue  # its own family has values of this size: a large family, not a typo
            if np.isfinite(reach[i]) and abs(rhs[i]) <= reach[i]:
                continue  # its own terms can add up to it (a budget of costs next to count limits of 1): its own scale
            if p95 > 0 and abs(rhs[i]) > LARGE_FACTOR * p95:
                add("constraint limit", md.row_names[i], rhs[i],
                    f"{abs(rhs[i]) / p95:.0f}x the model's 95th-percentile limit {p95:.15g}", p95, "(whole model)")

    out = [{k: v for k, v in f.items() if k != "_size"} for f in flags.values()]
    # IIS rows first, then the pattern breaks (precise, at most a few of each kind), then the other limits and bounds
    out = out + pattern_breaks(md, iis_rows, infeasible)
    rank = {"coefficient": 1, "objective coefficient": 1, "missing constraint": 1}
    # within a tier the farthest from typical first: the cut at max_flags kept a 35,200 among 1,477s only by file order
    out.sort(key=lambda f: (not f["in_iis"], -rank.get(f["kind"], 0), -_distance(f.get("value"), f.get("typical"))))
    return out[:max_flags]


def _distance(value, typical) -> float:
    """How far a flagged value is from its group's typical value, as an order of magnitude; 0 when unknown or either
    is 0 (a zero limit is as often a planned shutdown as a typo)."""
    if value is None or not isinstance(typical, (int, float)) or not np.isfinite(value) or not np.isfinite(typical):
        return 0.0
    a, b = abs(float(value)), abs(float(typical))
    return abs(float(np.log10(a / b))) if a and b else 0.0


def flag_fix(md: ModelData, flag: dict) -> dict | None:
    """The change that undoes a flag if it is a typo, as a modify_and_resolve change: a flipped sense back with the same
    limit, a limit or bound back to its group's typical value, a coefficient back to its row's typical one. None when
    the flag has no single value to put back (a missing row, an equality among inequalities)."""
    kind, name, typ = flag["kind"], flag["name"], flag.get("typical")
    if kind == "constraint sense":
        i = md.row_index(name)
        lo, hi = md.row_lo[i], md.row_hi[i]
        if np.isfinite(lo) and not np.isfinite(hi):
            return {"action": "set_rhs", "name": name, "lower": None, "upper": float(lo)}
        if np.isfinite(hi) and not np.isfinite(lo):
            return {"action": "set_rhs", "name": name, "lower": float(hi), "upper": None}
        return None
    if typ is None or not np.isfinite(typ):
        return None
    if kind == "constraint limit":
        i = md.row_index(name)
        side = "upper" if np.isfinite(md.row_hi[i]) else "lower"
        return {"action": "set_rhs", "name": name, side: float(typ)}
    if kind in ("variable upper bound", "variable lower bound"):
        if md.is_int[md.col_index(name)]:  # an integer variable's bound stays whole
            typ = float(np.floor(typ) if kind == "variable upper bound" else np.ceil(typ))
        return {"action": "set_bounds", "name": name, kind.split()[1]: float(typ)}
    if kind == "coefficient" and "column" in flag:
        return {"action": "set_coef", "name": name, "column": flag["column"], "value": float(typ)}
    return None


MAX_TYPOS = 3  # more flags than this of one kind: the model is written that way, not mistyped


def pattern_breaks(md: ModelData, iis_rows: set[str] | None = None, infeasible: bool = False) -> list[dict]:
    """Objective coefficients, constraint coefficients and missing rows that break their siblings' pattern (see
    objective_breaks, coefficient_breaks, missing_rows), in family_outliers' flag format. For an infeasible model
    only coefficients of IIS rows can matter: the objective does not constrain, a missing row only loosens, and a
    row outside the conflict does not cause it (a coefficient flag outside the IIS has led an agent's fix astray)."""
    iis_rows = iis_rows or set()
    if infeasible:
        return [f for f in pattern_breaks(md, iis_rows) if f["kind"] == "coefficient" and f["name"] in iis_rows]
    obj = [{"kind": "objective coefficient", "name": md.col_names[j], "value": float(md.obj[j]), "typical": t,
            "reason": why, "group": g, "in_iis": False} for j, why, t, g in objective_breaks(md)]
    seen, objs = set(), []
    for f in obj:  # one flag per column, the first group's reason
        if f["name"] not in seen:
            seen.add(f["name"])
            objs.append(f)
    coef = [{"kind": "coefficient", "name": md.row_names[r], "column": md.col_names[k], "value": float(md.A[r, k]),
             "typical": t, "reason": why, "group": "same shape", "in_iis": md.row_names[r] in iis_rows}
            for r, k, why, t in coefficient_breaks(md)]
    miss = [{"kind": "missing constraint", "name": n, "value": float("nan"), "typical": None, "reason": why,
             "group": "row names", "in_iis": False} for n, why in missing_rows(md)]
    return [f for kind in (coef, objs, miss) if len(kind) <= MAX_TYPOS for f in kind]


# Models that still solve but give a wrong optimum: the error sits in the objective, a coefficient or a missing row,
# which the limit and bound checks above never see. A typo changes a value by its sign or a power of ten, so these
# checks flag only a value that is such a typo of its siblings' common value (or, for signs, the lone opposite sign).
RUN_MIN = 4  # equal neighbours needed on one side before a value is read against them
SERIES_MIN = 12  # numbered rows needed before a lone gap counts (a 1-9 series often skips a number)


def _is_typo(x: float, typical: float, sign: bool = True) -> bool:
    """x is typical with its sign flipped (when sign) or shifted by a power of ten."""
    if x == typical or x == 0 or typical == 0:
        return False
    if np.isclose(x, -typical, rtol=1e-9, atol=0):
        return sign
    k = np.log10(abs(x / typical))
    return abs(k - round(k)) < 1e-9 and round(k) != 0 and np.sign(x) == np.sign(typical)


def objective_breaks(md: ModelData) -> list[tuple[int, str, float, str]]:
    """(column, reason, typical value, group) for objective coefficients that break their siblings' pattern: a lone
    opposite sign in a name group, a power-of-ten typo of a group's common value, or, in file order (anonymous
    models keep their author's column order), a typo of a run of equal coefficients next to it."""
    c, out = md.obj, []
    for g, idx in _groups(md.col_names):
        nz = [j for j in idx if c[j] != 0]
        if len(nz) < RUN_MIN:
            continue
        signs = np.sign(c[nz])
        if (signs > 0).sum() == 1 or (signs < 0).sum() == 1:
            lone = 1 if (signs > 0).sum() == 1 else -1
            j = nz[int(np.flatnonzero(signs == lone)[0])]
            others = c[[k for k in nz if k != j]]
            typ = float(np.median(others))
            if len(nz) > 2:
                out.append((j, f"sign opposite to its {len(nz) - 1} siblings (all {'negative' if lone > 0 else 'positive'})", typ, g))
        uniq, counts = np.unique(c[nz], return_counts=True)
        t = uniq[counts.argmax()]
        if counts.max() >= MODE_SHARE * len(nz):
            out += [(j, f"a sign or power-of-ten typo of its {counts.max()} of {len(nz)} siblings' {t:.15g}", float(t), g)
                    for j in nz if _is_typo(c[j], t)]
    for j in range(md.num_cols):
        for side in (range(j - 1, j - 1 - RUN_MIN, -1), range(j + 1, j + 1 + RUN_MIN)):
            ks = [k for k in side if 0 <= k < md.num_cols]
            if len(ks) == RUN_MIN and c[ks[0]] != 0 and all(c[k] == c[ks[0]] for k in ks) and _is_typo(c[j], c[ks[0]]):
                out.append((j, f"a sign or power-of-ten typo of the {RUN_MIN}+ columns next to it, all {c[ks[0]]:.15g}",
                            float(c[ks[0]]), f"columns {md.col_names[min(ks)]}..{md.col_names[max(ks)]} (file order)"))
                break
    return out


def coefficient_breaks(md: ModelData, max_patterns: int = 50) -> list[tuple[int, int, str, float]]:
    """(row, column, reason, typical value) for one coefficient that keeps a row (or column) from matching a shape
    that at least two other rows (columns) share exactly: same term count and the same coefficients but one, which
    is a power-of-ten typo of the shared one. Signs are left out: a +1 among -1s is how balance rows are written."""
    out, seen = [], set()
    for axis in ("row", "column"):
        M = md.A.tocsr() if axis == "row" else md.A.tocsc()
        n = M.shape[0] if axis == "row" else M.shape[1]
        sigs, buckets = [], {}
        for i in range(n):
            vals = np.round(M.data[M.indptr[i]:M.indptr[i + 1]], 9)
            sigs.append(tuple(sorted(vals.tolist())))
            buckets.setdefault(len(vals), {}).setdefault(sigs[-1], []).append(i)
        for by_sig in buckets.values():
            patterns = sorted(((s, m) for s, m in by_sig.items() if len(m) >= 2 and len(s) >= 2), key=lambda p: -len(p[1]))
            for sig, members in by_sig.items():
                if len(members) != 1:
                    continue
                i = members[0]
                for pat, pm in patterns[:max_patterns]:
                    extra = _multiset_minus(sig, pat)
                    missing = _multiset_minus(pat, sig)
                    if len(extra) == 1 and len(missing) == 1 and _is_typo(extra[0], missing[0], sign=False):
                        lo, hi = M.indptr[i], M.indptr[i + 1]
                        k = M.indices[lo + int(np.flatnonzero(np.isclose(M.data[lo:hi], extra[0], rtol=1e-9))[0])]
                        r, col = (i, k) if axis == "row" else (k, i)
                        if (r, col) not in seen:
                            seen.add((r, col))
                            name = md.row_names[i] if axis == "row" else md.col_names[i]
                            out.append((r, col, f"{axis} {name} matches {len(pm)} other {axis}s term for term except "
                                                f"this coefficient, which is {missing[0]:.15g} in them", float(missing[0])))
                        break
    return out


def _multiset_minus(a: tuple, b: tuple) -> list[float]:
    from collections import Counter
    return list((Counter(a) - Counter(b)).elements())


_NUMBERED = re.compile(r"^(\D*?)(\d+)(\D*)$")


def missing_rows(md: ModelData) -> list[tuple[str, str]]:
    """(name, reason) for the one row absent from an otherwise complete set: a numbered series (ROW00031,
    ROW00033: ROW00032) or an indexed family whose index combinations all appear but one."""
    out, present = [], set(md.row_names)
    series: dict[tuple, list[int]] = {}
    for n in md.row_names:
        m = _NUMBERED.match(n)
        if m and "[" not in n and "(" not in n:
            series.setdefault((m.group(1), len(m.group(2)) if m.group(2).startswith("0") else 0, m.group(3)), []).append(int(m.group(2)))
    for (pre, width, suf), nums in series.items():
        nums = sorted(set(nums))
        gaps = [a + 1 for a, b in zip(nums, nums[1:]) if b - a == 2]
        if len(nums) >= SERIES_MIN and len(gaps) == 1 and all(b - a in (1, 2) for a, b in zip(nums, nums[1:])):
            name = f"{pre}{str(gaps[0]).zfill(width)}{suf}"
            out.append((name, f"the only gap in the numbered rows {pre}{str(nums[0]).zfill(width)}{suf}.."
                              f"{pre}{str(nums[-1]).zfill(width)}{suf}"))
    fams: dict[tuple, list[tuple[str, ...]]] = {}
    for n in md.row_names:
        m = re.match(r"^([^\[(]+)([\[(])([^\])]*)([\])])$", n)
        if m:
            fams.setdefault((m.group(1), m.group(2), m.group(4)), []).append(tuple(m.group(3).split(",")))
    for (head, op, cl), idx in fams.items():
        dims = {len(t) for t in idx}
        if len(idx) < 8 or len(dims) != 1:
            continue
        values = [sorted({t[d] for t in idx}) for d in range(dims.pop())]
        size = int(np.prod([len(v) for v in values]))
        if size == len(set(idx)) + 1:
            import itertools
            gap = next(t for t in itertools.product(*values) if t not in set(idx))
            name = f"{head}{op}{','.join(gap)}{cl}"
            if name not in present:
                out.append((name, f"the only index combination missing from {head} ({len(idx)} of {size} rows present)"))
    return out
