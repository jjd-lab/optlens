"""Models built in Python: a Pyomo, gurobipy or PuLP model in memory, or a ``.py`` file that builds one.

Experts have modeling code, not MPS files. ``from_pyomo`` reads the active constraints and the objective through
Pyomo's own linear/quadratic representation, so the model keeps its component names (``avail[1day,0]``, not the
LP writer's ``avail(1day_0)``), a ranged constraint stays one row, and a fixed variable stays a column fixed at its
value (a fixed variable is often the cause of an infeasibility; the LP writer would fold it into the constants).
Each library is imported only when used (extras ``optlens[pyomo]``, ``optlens[gurobi]``, ``optlens[pulp]``).
"""
from __future__ import annotations

import contextlib
import importlib.util
import math
import os
import re
import sys
import types
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

from .model import INF, ModelData, UnsupportedModel, from_gurobipy


def _clean(name: str) -> str:
    # Backends exchange the model as MPS, where whitespace ends a name; Gurobi does the same replacement.
    return re.sub(r"\s", "_", name)


def _bound(v, default: float) -> float:
    return default if v is None or abs(v) >= 1e20 else float(v)


def from_pyomo(model: Any, name: str | None = None) -> ModelData:
    """Read a Pyomo ``ConcreteModel`` (or block) into ``ModelData``: active constraints, the one active objective,
    and every variable they use, in order of first appearance."""
    import pyomo.environ as pyo
    from pyomo.core.base.sos import SOSConstraint
    from pyomo.repn import generate_standard_repn

    model_name = name or _clean(model.name or "pyomo_model")
    objectives = list(model.component_data_objects(pyo.Objective, active=True, descend_into=True))
    if len(objectives) != 1:
        raise UnsupportedModel(f"{model_name}: {len(objectives)} active objectives (need exactly one)")
    if next(model.component_data_objects(SOSConstraint, active=True, descend_into=True), None) is not None:
        raise UnsupportedModel(f"{model_name}: SOS constraints")
    cons = list(model.component_data_objects(pyo.Constraint, active=True, descend_into=True))

    cols: dict[int, int] = {}
    vars_: list = []

    def col(v) -> int:
        j = cols.get(id(v))
        if j is None:
            j = cols[id(v)] = len(vars_)
            vars_.append(v)
        return j

    # Fixed variables would become constants in the representation; read them as columns fixed at their value.
    fixed = [v for v in model.component_data_objects(pyo.Var, descend_into=True) if v.fixed]
    values = [v.value for v in fixed]
    for v in fixed:
        v.unfix()
    try:
        obj = objectives[0]
        rep = generate_standard_repn(obj.expr, compute_values=True, quadratic=True)
        if rep.nonlinear_expr is not None:
            raise UnsupportedModel(f"{model_name}: nonlinear objective {obj.name}")
        c: dict[int, float] = {}
        for v, a in zip(rep.linear_vars, rep.linear_coefs):
            c[col(v)] = c.get(col(v), 0.0) + float(a)
        quad: dict[tuple[int, int], float] = {}
        for (v1, v2), a in zip(rep.quadratic_vars, rep.quadratic_coefs):
            i, j = col(v1), col(v2)
            if i == j:  # 0.5 x'Qx: a square's coefficient doubles, a product splits over Q_ij and Q_ji
                quad[i, i] = quad.get((i, i), 0.0) + 2 * float(a)
            else:
                quad[i, j] = quad.get((i, j), 0.0) + float(a)
                quad[j, i] = quad.get((j, i), 0.0) + float(a)
        offset = float(rep.constant)

        r, cc, d, lo, hi, row_names = [], [], [], [], [], []
        for k, con in enumerate(cons):
            rep = generate_standard_repn(con.body, compute_values=True, quadratic=False)
            if rep.nonlinear_expr is not None or rep.quadratic_vars:
                raise UnsupportedModel(f"{model_name}: nonlinear constraint {con.name}")
            for v, a in zip(rep.linear_vars, rep.linear_coefs):
                r.append(k)
                cc.append(col(v))
                d.append(float(a))
            const = float(rep.constant)
            lo.append(_bound(pyo.value(con.lower) if con.has_lb() else None, -INF) - const)
            hi.append(_bound(pyo.value(con.upper) if con.has_ub() else None, INF) - const)
            row_names.append(_clean(con.name))

        n = len(vars_)
        col_lb = np.array([_bound(v.lb, -INF) for v in vars_])
        col_ub = np.array([_bound(v.ub, INF) for v in vars_])
        is_int = np.array([v.is_integer() or v.is_binary() for v in vars_], bool)
    finally:
        for v, x in zip(fixed, values):
            v.fix(x)
    for v, x in zip(fixed, values):
        if id(v) in cols:
            if x is None:
                raise UnsupportedModel(f"{model_name}: {v.name} is fixed without a value")
            col_lb[cols[id(v)]] = col_ub[cols[id(v)]] = float(x)

    obj_vec = np.zeros(n)
    for j, a in c.items():
        obj_vec[j] = a
    Q = None
    if any(quad.values()):
        keys = list(quad)
        Q = sp.csc_matrix(([quad[k] for k in keys], ([k[0] for k in keys], [k[1] for k in keys])), shape=(n, n))
    return ModelData(
        name=model_name, minimize=obj.sense == pyo.minimize, obj=obj_vec, obj_offset=offset,
        A=sp.csr_matrix((d, (r, cc)), shape=(len(cons), n)), row_lo=np.array(lo, float),
        row_hi=np.array(hi, float), col_lb=col_lb, col_ub=col_ub, is_int=is_int,
        col_names=tuple(_clean(v.name) for v in vars_), row_names=tuple(row_names), Q=Q,
    )


def from_pulp(prob: Any, name: str | None = None) -> ModelData:
    """``ModelData`` from a PuLP ``LpProblem`` (PuLP 2.x and 4.x): its variables in PuLP's order, its constraints
    in the order they were added."""
    import pulp

    model_name = name or _clean(prob.name or "pulp_model")
    has_sos = prob.has_sos() if callable(getattr(prob, "has_sos", None)) else \
        bool(getattr(prob, "sos1", None) or getattr(prob, "sos2", None))
    if has_sos:
        raise UnsupportedModel(f"{model_name}: SOS constraints")
    vars_ = list(prob.variables())
    names = [v.name for v in vars_]
    if len(set(names)) != len(names):
        raise UnsupportedModel(f"{model_name}: two variables share a name")
    col = {n: j for j, n in enumerate(names)}

    def j_of(v) -> int:
        if v.name not in col:  # PuLP 2.x lists only variables it has seen; a stray one is added
            col[v.name] = len(vars_)
            vars_.append(v)
        return col[v.name]

    cons = prob.constraints() if callable(prob.constraints) else prob.constraints
    pairs = list(cons.items()) if isinstance(cons, dict) else [(c.name, c) for c in cons]
    r, cc, d, lo, hi, row_names = [], [], [], [], [], []
    for i, (cname, c) in enumerate(pairs):
        for v, a in c.items():
            r.append(i)
            cc.append(j_of(v))
            d.append(float(a))
        rhs = -float(c.constant)  # PuLP stores expr + constant (sense) 0
        lo.append(rhs if c.sense in (pulp.LpConstraintGE, pulp.LpConstraintEQ) else -INF)
        hi.append(rhs if c.sense in (pulp.LpConstraintLE, pulp.LpConstraintEQ) else INF)
        row_names.append(_clean(cname or f"_C{i + 1}"))
    obj_terms = list(prob.objective.items()) if prob.objective is not None else []
    for v, _ in obj_terms:
        j_of(v)
    n = len(vars_)
    obj = np.zeros(n)
    for v, a in obj_terms:
        obj[col[v.name]] += float(a)
    offset = float(prob.objective.constant) if prob.objective is not None else 0.0

    def bound(b, default):
        return default if b is None or (isinstance(b, float) and math.isinf(b)) else _bound(b, default)

    return ModelData(
        name=model_name, minimize=prob.sense == pulp.LpMinimize, obj=obj, obj_offset=offset,
        A=sp.csr_matrix((d, (r, cc)), shape=(len(pairs), n)), row_lo=np.array(lo, float),
        row_hi=np.array(hi, float), col_lb=np.array([bound(v.lowBound, -INF) for v in vars_]),
        col_ub=np.array([bound(v.upBound, INF) for v in vars_]),
        is_int=np.array([v.cat in ("Integer", "Binary") for v in vars_], bool),
        col_names=tuple(_clean(v.name) for v in vars_), row_names=tuple(row_names),
    )


_LIBRARIES = {"pyomo": "Pyomo", "gurobipy": "gurobipy", "pulp": "PuLP"}
_UNNAMED = {"unknown", "pyomo_model", "gurobipy_model", "pulp_model", "NoName"}  # the libraries' default names


def _library(obj: Any) -> str | None:
    lib = type(obj).__module__.split(".")[0]
    return lib if lib in _LIBRARIES else None


def from_object(obj: Any, name: str | None = None) -> ModelData:
    """``ModelData`` from a model object of a supported modeling library (Pyomo, gurobipy, PuLP)."""
    if isinstance(obj, ModelData):
        return obj
    lib = _library(obj)
    if lib == "pyomo":
        return from_pyomo(obj, name)
    if lib == "gurobipy":
        return from_gurobipy(obj, name)
    if lib == "pulp":
        return from_pulp(obj, name)
    raise UnsupportedModel(f"{type(obj).__module__}.{type(obj).__name__} is not a supported model object "
                           f"({', '.join(_LIBRARIES.values())})")


def _is_model(obj: Any) -> bool:
    lib = _library(obj)
    if lib == "pyomo":
        import pyomo.environ as pyo

        return isinstance(obj, pyo.ConcreteModel)
    if lib == "gurobipy":
        import gurobipy

        return isinstance(obj, gurobipy.Model)
    if lib == "pulp":
        import pulp

        return isinstance(obj, pulp.LpProblem)
    return False


BUILDER_NAMES = ("build_model", "create_model", "make_model", "get_model")


class _Solved(BaseException):
    """Raised by an intercepted solve call: the script stops with its model as it was at that moment, already
    converted (unwinding the script can dispose of the model, as ``with gp.Model() as m:`` does)."""

    def __init__(self, model):
        super().__init__("solve intercepted")
        self.model = from_object(model)


class _CapturingPyomoSolver:
    """What SolverFactory returns while a script is loaded: solve() hands over the model instead of solving."""

    def __init__(self, *args, **kwargs):
        self.options: dict = {}
        self._model = None

    def available(self, *args, **kwargs):
        return True

    def set_instance(self, model, *args, **kwargs):
        self._model = model

    def solve(self, model=None, *args, **kwargs):
        raise _Solved(model if model is not None else self._model)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@contextlib.contextmanager
def _solves_intercepted():
    """Make gurobipy's Model.optimize, PuLP's LpProblem.solve and Pyomo's SolverFactory(...).solve stop the script
    and hand over the model, for each library that is installed."""
    undo = []

    def patch(owner, attr, value):
        undo.append((owner, attr, getattr(owner, attr)))
        setattr(owner, attr, value)

    def stop(self, *args, **kwargs):
        raise _Solved(self)

    try:
        if importlib.util.find_spec("gurobipy"):
            import gurobipy

            patch(gurobipy.Model, "optimize", stop)
        if importlib.util.find_spec("pulp"):
            import pulp

            patch(pulp.LpProblem, "solve", stop)
        if importlib.util.find_spec("pyomo"):
            import pyomo.environ as pyo

            patch(type(pyo.SolverFactory), "__call__", lambda self, *a, **k: _CapturingPyomoSolver())
        yield
    finally:
        for owner, attr, value in reversed(undo):
            setattr(owner, attr, value)


def _call(fn):
    try:
        return fn()
    except _Solved as s:
        return s.model


def load_script(path: str | os.PathLike, attr: str | None = None) -> ModelData:
    """Run a Python file and read the model it builds (Pyomo, gurobipy or PuLP).

    The file runs as a module named ``__optlens__`` (so a block under ``if __name__ == "__main__":`` does not run),
    from its own folder, so local imports and data files work. Solving is intercepted: the first call to gurobipy's
    ``optimize()``, PuLP's ``solve()`` or a Pyomo ``SolverFactory`` solver's ``solve()`` stops the file, and that
    model, as it is at that moment, is the one read (scripts often build and solve at module level, and the
    size-limited Gurobi license refuses to solve large models). ``attr`` names the model, or a function with no
    required arguments that returns it; without it, the model the file solved, else its one module-level model, else
    its one builder function named build_model, create_model, make_model or get_model.
    This executes the file: load only code you trust."""
    path = Path(path).resolve()
    module = types.ModuleType("__optlens__")
    module.__file__ = str(path)
    ns = module.__dict__
    saved_main, cwd = sys.modules.get("__optlens__"), os.getcwd()
    sys.modules["__optlens__"] = module  # dataclasses and pickling look the module up
    sys.path.insert(0, str(path.parent))
    try:
        os.chdir(path.parent)
        with _solves_intercepted():
            code = compile(path.read_bytes(), str(path), "exec")
            solved = None
            try:
                exec(code, ns)
            except _Solved as s:
                solved = s.model
            except UnsupportedModel:
                raise
            except (Exception, SystemExit) as e:
                raise RuntimeError(f"running {path.name} failed: {type(e).__name__}: {e}") from e
            if attr is not None:
                if attr not in ns:
                    raise UnsupportedModel(f"{path.name} defines no {attr!r}"
                                           + (" before its first solve" if solved is not None else ""))
                obj = ns[attr]
                obj = _call(obj) if callable(obj) and not _is_model(obj) else obj
            elif solved is not None:
                obj = solved
            else:
                models = {k: v for k, v in ns.items() if _is_model(v)}
                builders = {k: ns[k] for k in BUILDER_NAMES if callable(ns.get(k))}
                if len(models) == 1:
                    obj = next(iter(models.values()))
                elif not models and len(builders) == 1:
                    obj = _call(next(iter(builders.values())))
                else:
                    found = sorted(models) or sorted(builders)
                    raise UnsupportedModel(
                        f"{path.name}: {'found ' + ', '.join(found) if found else 'no model found'}; name one as "
                        f"{path.name}:<name> (a model, or a function with no arguments that returns one)")
            if obj is None or not (_is_model(obj) or isinstance(obj, ModelData)):
                raise UnsupportedModel(f"{path.name}: {type(obj).__name__} is not a model "
                                       f"({', '.join(_LIBRARIES.values())})")
            md = from_object(obj)
            return replace(md, name=path.stem) if md.name in _UNNAMED else md
    finally:
        os.chdir(cwd)
        sys.path.remove(str(path.parent))
        if saved_main is None:
            sys.modules.pop("__optlens__", None)
        else:
            sys.modules["__optlens__"] = saved_main
