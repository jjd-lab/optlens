"""Solver-neutral model data: load any LP/MPS file, edit it, and build elastic (relaxed) copies.

Every backend receives the same ``ModelData`` (written to MPS by HiGHS), so edits and
feasibility relaxation are implemented once, independent of the solver.
"""
from __future__ import annotations

import os
import re
import weakref
from dataclasses import dataclass, replace
from pathlib import Path

import highspy
import numpy as np
import scipy.sparse as sp

INF = highspy.kHighsInf
CONVEXITY_CHECK_MAX = 3000  # variables in the quadratic term above which convexity is assumed, not checked (a dense
                            # eigenvalue check on that many takes seconds)
_CONVEX: dict[tuple[int, bool], tuple[weakref.ref, bool]] = {}  # (id(Q), minimize) -> (Q, convex): edits keep Q
_POSITION: dict[int, tuple[tuple, dict[str, int]]] = {}  # id(names) -> (names, name -> first position)


def _position(names: tuple, name: str, kind: str) -> int:
    # a dict per names tuple (edits keep it): tuple.index made one 103k-term add_row take 76 s on 73k columns, and
    # dropping a 262,800-row family from 279k rows take ~560 s
    hit = _POSITION.get(id(names))
    if hit is None or hit[0] is not names:
        if len(_POSITION) >= 16:
            _POSITION.clear()
        hit = _POSITION[id(names)] = (names, {n: i for i, n in reversed(list(enumerate(names)))})
    if (i := hit[1].get(name)) is None:
        raise ValueError(f"{name!r} is not a {kind}")
    return i


def _inf(v: float | None) -> float | None:
    """Solver-style infinity (1e20 in HiGHS and SCIP, 1e30 in Gurobi) as a true infinity."""
    return v if v is None or abs(v) < 1e20 else float(np.copysign(INF, v))


class UnsupportedModel(Exception):
    """The file uses features the model can't hold (general/indicator constraints, SOS, quadratic constraints)."""


@dataclass(frozen=True)
class ModelData:
    name: str
    minimize: bool
    obj: np.ndarray            # (n,)
    obj_offset: float
    A: sp.csr_matrix           # (m, n)
    row_lo: np.ndarray         # (m,)  -INF when absent
    row_hi: np.ndarray         # (m,)  +INF when absent
    col_lb: np.ndarray         # (n,)
    col_ub: np.ndarray         # (n,)
    is_int: np.ndarray         # (n,) bool
    col_names: tuple[str, ...]
    row_names: tuple[str, ...]
    Q: sp.csc_matrix | None = None  # symmetric (n, n); objective = obj @ x + 0.5 x' Q x + obj_offset

    @property
    def quadratic_objective(self) -> bool:
        return self.Q is not None

    def convex_objective(self) -> bool:
        """Whether the objective is convex for its sense (a linear one is): 0.5 x'Qx needs Q positive semidefinite to
        minimize, negative semidefinite to maximize. HiGHS solves only convex QPs. Checked on the variables the quadratic
        term touches; above CONVEXITY_CHECK_MAX of them it is assumed convex."""
        if self.Q is None:
            return True
        key = (id(self.Q), self.minimize)
        if (hit := _CONVEX.get(key)) is not None and hit[0]() is self.Q:
            return hit[1]
        idx = np.unique(self.Q.nonzero()[0])
        if len(idx) > CONVEXITY_CHECK_MAX:
            return True
        sub = self.Q[idx][:, idx].toarray() * (1.0 if self.minimize else -1.0)
        eig = np.linalg.eigvalsh((sub + sub.T) / 2)
        convex = bool(eig.min() >= -1e-9 * max(1.0, float(np.abs(eig).max())))
        _CONVEX[key] = (weakref.ref(self.Q), convex)
        return convex

    def objective_value(self, x: np.ndarray) -> float:
        quad = 0.5 * float(x @ (self.Q @ x)) if self.Q is not None else 0.0
        return float(self.obj @ x) + quad + self.obj_offset

    @property
    def num_rows(self) -> int:
        return self.A.shape[0]

    @property
    def num_cols(self) -> int:
        return self.A.shape[1]

    @property
    def is_mip(self) -> bool:
        return bool(self.is_int.any())

    def row_index(self, name: str) -> int:
        return _position(self.row_names, name, "row")

    def col_index(self, name: str) -> int:
        return _position(self.col_names, name, "column")

    # ---- edits (each returns a new ModelData) ----

    def set_row_bounds(self, row: str, lo: float | None = None, hi: float | None = None) -> "ModelData":
        lo, hi = _inf(lo), _inf(hi)
        i = self.row_index(row)
        row_lo, row_hi = self.row_lo.copy(), self.row_hi.copy()
        if lo is not None:
            row_lo[i] = lo
        if hi is not None:
            row_hi[i] = hi
        return replace(self, row_lo=row_lo, row_hi=row_hi)

    def set_col_bounds(self, col: str, lb: float | None = None, ub: float | None = None) -> "ModelData":
        lb, ub = _inf(lb), _inf(ub)
        j = self.col_index(col)
        col_lb, col_ub = self.col_lb.copy(), self.col_ub.copy()
        if lb is not None:
            col_lb[j] = lb
        if ub is not None:
            col_ub[j] = ub
        return replace(self, col_lb=col_lb, col_ub=col_ub)

    def lp_relaxation(self) -> "ModelData":
        return replace(self, is_int=np.zeros(self.num_cols, bool))

    def set_obj_coef(self, col: str, value: float) -> "ModelData":
        obj = self.obj.copy()
        obj[self.col_index(col)] = value
        return replace(self, obj=obj)

    def set_coef(self, row: str, col: str, value: float) -> "ModelData":
        A = self.A.tolil()
        A[self.row_index(row), self.col_index(col)] = float(value)
        A = A.tocsr()
        A.eliminate_zeros()  # a coefficient set to 0 leaves the row
        return replace(self, A=A)

    def keep_rows(self, rows: list[int]) -> "ModelData":
        rows = sorted(rows)
        return replace(
            self, A=self.A[rows, :].tocsr(), row_lo=self.row_lo[rows], row_hi=self.row_hi[rows],
            row_names=tuple(self.row_names[i] for i in rows),
        )

    def drop_rows(self, names: list[str]) -> "ModelData":
        drop = {self.row_index(n) for n in names}
        return self.keep_rows([i for i in range(self.num_rows) if i not in drop])

    def add_row(self, name: str, coefs: dict[str, float], lo: float = -INF, hi: float = INF) -> "ModelData":
        new = sp.csr_matrix(
            ([float(v) for v in coefs.values()], ([0] * len(coefs), [self.col_index(c) for c in coefs])),
            shape=(1, self.num_cols),
        )
        return replace(
            self, A=sp.vstack([self.A, new]).tocsr(), row_lo=np.append(self.row_lo, lo),
            row_hi=np.append(self.row_hi, hi), row_names=self.row_names + (name,),
        )

    def add_cols(self, names: list[str], obj: np.ndarray, lb: np.ndarray, ub: np.ndarray,
                 A_cols: sp.spmatrix) -> "ModelData":
        Q = None
        if self.Q is not None:
            k = len(names)
            Q = sp.bmat([[self.Q, None], [None, sp.csc_matrix((k, k))]], format="csc")
        return replace(
            self, Q=Q, obj=np.concatenate([self.obj, obj]), A=sp.hstack([self.A, A_cols]).tocsr(),
            col_lb=np.concatenate([self.col_lb, lb]), col_ub=np.concatenate([self.col_ub, ub]),
            is_int=np.concatenate([self.is_int, np.zeros(len(names), bool)]),
            col_names=self.col_names + tuple(names),
        )

    def expression(self, i: int) -> str:
        row = self.A.getrow(i)
        terms = [f"{v:+.15g} {self.col_names[j]}" for j, v in zip(row.indices, row.data)]
        return " ".join(terms).lstrip("+") if terms else "0"

    # ---- I/O ----

    def to_highs_lp(self) -> highspy.HighsLp:
        csc = self.A.tocsc()
        lp = highspy.HighsLp()
        lp.num_col_, lp.num_row_ = self.num_cols, self.num_rows
        lp.sense_ = highspy.ObjSense.kMinimize if self.minimize else highspy.ObjSense.kMaximize
        lp.offset_ = self.obj_offset
        lp.col_cost_, lp.col_lower_, lp.col_upper_ = self.obj, self.col_lb, self.col_ub
        lp.row_lower_, lp.row_upper_ = self.row_lo, self.row_hi
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.num_col_, lp.a_matrix_.num_row_ = self.num_cols, self.num_rows
        lp.a_matrix_.start_, lp.a_matrix_.index_, lp.a_matrix_.value_ = csc.indptr, csc.indices, csc.data
        if self.is_mip:
            lp.integrality_ = [highspy.HighsVarType.kInteger if b else highspy.HighsVarType.kContinuous
                               for b in self.is_int]
        lp.col_names_, lp.row_names_ = list(self.col_names), list(self.row_names)
        return lp

    def to_highs(self) -> highspy.Highs:
        h = _quiet_highs()
        h.passModel(self.to_highs_lp())
        if self.Q is not None:
            L = sp.tril(self.Q, format="csc")
            L.sort_indices()
            h.passHessian(self.num_cols, L.nnz, int(highspy.HessianFormat.kTriangular),
                          L.indptr.astype(np.int32), L.indices.astype(np.int32), L.data.astype(np.float64))
        return h

    def write_mps(self, path: str) -> str:
        # HiGHS refuses to write a model with no rows (an IIS check can drop the last one); a free empty row is inert.
        h = (self if self.num_rows else self.add_row("_free", {})).to_highs()
        if h.writeModel(path) != highspy.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS could not write {path}")
        return path


def _quiet_highs() -> highspy.Highs:
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    return h


def _from_highs(h: highspy.Highs, name: str) -> ModelData:
    lp = h.getLp()
    m, n = lp.num_row_, lp.num_col_
    am = lp.a_matrix_
    if am.format_ == highspy.MatrixFormat.kRowwise:
        A = sp.csr_matrix((am.value_, am.index_, am.start_), shape=(m, n))
    else:
        A = sp.csc_matrix((am.value_, am.index_, am.start_), shape=(m, n)).tocsr()
    integrality = list(lp.integrality_ or [])
    is_int = np.array([t != highspy.HighsVarType.kContinuous for t in integrality], bool) if integrality \
        else np.zeros(n, bool)
    col_names = tuple(lp.col_names_) if len(lp.col_names_) == n else tuple(f"C{j}" for j in range(n))
    row_names = tuple(lp.row_names_) if len(lp.row_names_) == m else tuple(f"R{i}" for i in range(m))
    return ModelData(
        name=name, minimize=lp.sense_ == highspy.ObjSense.kMinimize,
        obj=np.asarray(lp.col_cost_, float), obj_offset=float(lp.offset_), A=A,
        row_lo=np.asarray(lp.row_lower_, float), row_hi=np.asarray(lp.row_upper_, float),
        col_lb=np.asarray(lp.col_lower_, float), col_ub=np.asarray(lp.col_upper_, float),
        is_int=is_int, col_names=col_names, row_names=row_names,
        Q=_hessian(h.getModel().hessian_, n),
    )


def _hessian(hess, n: int) -> sp.csc_matrix | None:
    """HiGHS stores the lower triangle column-wise (or a full square); return the full symmetric matrix."""
    if len(hess.value_) == 0:
        return None
    M = sp.csc_matrix((hess.value_, hess.index_, hess.start_), shape=(n, n))
    if hess.format_ == highspy.HessianFormat.kTriangular:
        M = (M + M.T - sp.diags(M.diagonal())).tocsc()
    return M


def _from_gurobi(path: str, name: str) -> ModelData:
    """Build the model from Gurobi's own reader, for files HiGHS rejects or reads only with warnings.
    HiGHS warnings are not cosmetic: on Gurobi-written MPS a fixed binary (BV then FX) keeps only its
    first bound, silently unfixing it. Reading and matrix access are allowed on Gurobi's size-limited
    license; only optimize() is capped."""
    import gurobipy as gp

    with gp.Env(params={"OutputFlag": 0}) as env, gp.read(path, env=env) as m:
        return from_gurobipy(m, name)


def from_gurobipy(m, name: str | None = None) -> ModelData:
    """``ModelData`` from a gurobipy ``Model`` (pending changes are applied with ``update()``; nothing is solved,
    so this works on the size-limited license at any size)."""
    from gurobipy import GRB

    m.update()
    name = name or m.ModelName or "gurobipy_model"
    if m.NumObj > 1 or m.NumGenConstrs or m.NumSOS or m.NumQConstrs:
        raise UnsupportedModel(
            f"{name}: {m.NumObj} objectives, {m.NumGenConstrs} general, {m.NumSOS} SOS, "
            f"{m.NumQConstrs} quadratic constraints")
    vs, cs = m.getVars(), m.getConstrs()
    if any(v.VType in (GRB.SEMICONT, GRB.SEMIINT) for v in vs):
        raise UnsupportedModel(f"{name}: semi-continuous variables")
    inf = lambda a: np.where(a >= GRB.INFINITY, INF, np.where(a <= -GRB.INFINITY, -INF, a))  # noqa: E731
    rhs = np.array([c.RHS for c in cs], float)
    sense = [c.Sense for c in cs]
    Q = None
    if m.IsQP:
        G = m.getQ().tocsc()  # Gurobi's objective term is x'Gx; ModelData uses 0.5 x'Qx
        Q = (G + G.T).tocsc()
    return ModelData(
        name=name, minimize=m.ModelSense == GRB.MINIMIZE,
        obj=np.array([v.Obj for v in vs], float), obj_offset=float(m.ObjCon),
        A=m.getA().tocsr() if cs else sp.csr_matrix((0, len(vs))),
        row_lo=np.array([r if s != "<" else -INF for r, s in zip(rhs, sense)]),
        row_hi=np.array([r if s != ">" else INF for r, s in zip(rhs, sense)]),
        col_lb=inf(np.array([v.LB for v in vs], float)), col_ub=inf(np.array([v.UB for v in vs], float)),
        is_int=np.array([v.VType in (GRB.BINARY, GRB.INTEGER) for v in vs], bool),
        col_names=tuple(v.VarName for v in vs), row_names=tuple(c.ConstrName for c in cs), Q=Q,
    )


def load(path: str | os.PathLike) -> ModelData:
    """Load an LP/MPS(.gz) file. HiGHS first (open source); for an LP file HiGHS fails or warns on, optlens's
    own LP reader (``lpfile``); then Gurobi's reader when gurobipy is installed. A ``.py`` file (``model.py`` or
    ``model.py:name``) is run and the model it builds is read (``loaders.load_script``)."""
    path = str(path)
    script = re.fullmatch(r"(.+\.py)(?::([A-Za-z_]\w*))?", path)
    if script:
        from .loaders import load_script

        return load_script(script[1], script[2])
    name = Path(path).name.split(".")[0]
    h = _quiet_highs()
    status = h.readModel(path)
    if status == highspy.HighsStatus.kOk and h.getLp().num_col_ > 0:
        return _from_highs(h, name)
    if path.endswith(".gz") and status == highspy.HighsStatus.kError:
        # The Windows HiGHS wheel reads no compressed files: read a decompressed copy instead.
        import gzip
        import shutil
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            plain = os.path.join(tmp, Path(path).name[:-3])
            with gzip.open(path, "rb") as src, open(plain, "wb") as dst:
                shutil.copyfileobj(src, dst)
            return load(plain)
    lp_error = None
    if path.lower().endswith(".lp"):
        # HiGHS rejects names with brackets, which Gurobi writes for every indexed variable: read it ourselves.
        from .lpfile import LPParseError, read_lp

        try:
            return read_lp(path, name)
        except LPParseError as e:
            lp_error = e
    try:
        import gurobipy  # noqa: F401
    except ImportError:
        if status == highspy.HighsStatus.kWarning:
            return _from_highs(h, name)
        raise RuntimeError(f"HiGHS could not read {path} and gurobipy is not installed"
                           + (f"; LP reader: {lp_error}" if lp_error else "")) from None
    return _from_gurobi(path, name)
