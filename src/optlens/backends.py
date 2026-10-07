"""Solver backends. Each one solves a ``ModelData`` and, where the solver can, computes an IIS natively.

Statuses are normalized to: OPTIMAL, INFEASIBLE, UNBOUNDED, INF_OR_UNBD, TIME_LIMIT, OTHER.
"""
from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
import threading
from dataclasses import dataclass, field, replace

import highspy
import numpy as np

from .model import ModelData, UnsupportedModel

SOLVE_MEMORY = 4e9         # bytes that one tool's parallel solves may use together (each runs in its own process)
SOLVE_BYTES_PER_NZ = 500   # a solve process's peak memory per constraint coefficient (3.5 GB at 837k rows, 7M nonzeros)


def parallel_solves(md: ModelData, most: int = 4) -> int:
    """How many solves of a model this size can run at once within SOLVE_MEMORY (at least one): parallel solves of an
    837k-row model, each in its own process, ran a 16 GB machine out of memory."""
    return max(1, min(most, int(SOLVE_MEMORY // max(1, md.A.nnz * SOLVE_BYTES_PER_NZ))))


class LicenseLimit(Exception):
    """Gurobi's size-limited license can't optimize this model (over 2,000 vars or constraints)."""


class IISNotSupported(Exception):
    pass


@dataclass
class SolveResult:
    status: str
    obj: float | None = None
    x: np.ndarray | None = None
    row_dual: np.ndarray | None = None      # LP only, and only where the backend reports it
    reduced_cost: np.ndarray | None = None
    bound: float | None = None              # MIP only: the solver's proven bound on the optimum (same sense as obj)

    @property
    def feasible(self) -> bool:
        return self.status == "OPTIMAL" or (self.status == "TIME_LIMIT" and self.x is not None)

    @property
    def gap(self) -> float | None:
        """Relative distance from the plan's objective to the bound, as the solvers define it; None without both."""
        if self.obj is None or self.bound is None or not np.isfinite(self.bound):
            return None
        return abs(self.obj - self.bound) / max(abs(self.obj), 1e-10)


@dataclass
class IIS:
    rows: list[str]
    col_bounds: list[str] = field(default_factory=list)  # columns whose bounds are part of the IIS
    method: str = ""
    # True when the solver reports which bounds belong to the IIS (Gurobi, HiGHS); bounds outside
    # col_bounds are then not part of the conflict. False means the IIS is relative to all bounds.
    bounds_reported: bool = False
    note: str = ""  # why the IIS came from a fallback (a native IIS rejected), for the user


def crossed_bounds(md: ModelData, tol: float = 1e-9) -> tuple[list[str], list[str]]:
    """Rows and columns whose lower limit exceeds the upper one."""
    rows = [md.row_names[i] for i in np.flatnonzero(md.row_lo > md.row_hi + tol)]
    cols = [md.col_names[j] for j in np.flatnonzero(md.col_lb > md.col_ub + tol)]
    return rows, cols


_GRACE = 5.0  # seconds past the solver's own time limit before the child process is killed


class _TimedOut(Exception):
    pass


def _worker(conn):
    """A reusable solver process: run each (fn, args) it receives and send back the result."""
    while True:
        try:
            fn, args = conn.recv()
        except EOFError:  # the parent closed the pipe
            return
        try:
            conn.send(("ok", fn(*args)))
        except BaseException as e:  # sent to the parent and re-raised there
            try:
                conn.send(("err", e))
            except Exception:  # an exception that doesn't pickle
                conn.send(("err", RuntimeError(f"{type(e).__name__}: {e}")))


_CTX = None


def _context():
    """forkserver, not fork: forking a process that already runs solver threads can deadlock the
    child. The server is a clean single-threaded process with the solver modules preloaded. Windows has
    no forkserver; spawn starts each worker as a fresh interpreter there (slower to start, same limit)."""
    global _CTX
    if _CTX is None:
        if "forkserver" in mp.get_all_start_methods():
            _CTX = mp.get_context("forkserver")
            _CTX.set_forkserver_preload(["optlens"])
        else:
            _CTX = mp.get_context("spawn")
    return _CTX


class _Worker:
    def __init__(self):
        ctx = _context()
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_worker, args=(child,), daemon=True)
        self.proc.start()
        child.close()
        self.calls = 0

    def kill(self):
        if self.proc.is_alive():
            self.proc.kill()
        self.proc.join()
        self.conn.close()
        _remove_temp_of(self.proc.pid)  # a killed solve skips its own cleanup (one model copy each, 0.5 GB at 837k rows)


_IDLE: list[_Worker] = []
_IDLE_LOCK = threading.Lock()
_MAX_IDLE = 8
_MAX_CALLS = 500  # recycle a worker after this many calls (solver memory growth)
_KEEP_MAX_NNZ = 1_000_000  # a worker that solved a larger model is not kept: it holds on to that memory (1.5 GB at 837k
                           # rows), and starting a new one (about a second) is small next to such a solve


def _with_deadline(fn, args, seconds: float):
    """Run fn(*args) in a solver process and kill that process after ``seconds``. Solvers do not always
    honour their own time limit (SCIP ran 862 s on a 30 s limit), so the limit is enforced from outside.
    Processes are reused between calls: starting one costs about a second (the forkserver child imports
    the main module), a small solve a few milliseconds. A process that times out or dies is discarded."""
    md = next((a for a in args if isinstance(a, ModelData)), None)
    keep = md is None or md.A.nnz <= _KEEP_MAX_NNZ
    with _IDLE_LOCK:
        w = _IDLE.pop() if _IDLE else None
    if w is None or not w.proc.is_alive():
        w = _Worker()
    ok = False
    try:
        w.conn.send((fn, args))
        if not w.conn.poll(seconds):
            raise _TimedOut
        try:
            kind, value = w.conn.recv()
        except EOFError:  # the process died without a result (solver crash, killed for memory)
            w.proc.join()
            raise RuntimeError(f"solver process exited with code {w.proc.exitcode} and no result") from None
        ok = True
    finally:
        w.calls += 1
        if ok and keep and w.calls < _MAX_CALLS and w.proc.is_alive():
            with _IDLE_LOCK:
                if len(_IDLE) < _MAX_IDLE:
                    _IDLE.append(w)
                    w = None
        if w is not None:
            w.kill()
    if kind == "err":
        raise value
    return value


def race(md: ModelData, backends: list, time_limit: float = 60.0, start: np.ndarray | None = None):
    """Solve ``md`` with several backends at once; return (winner, SolveResult) for the first conclusive result
    (OPTIMAL, INFEASIBLE, UNBOUNDED, INF_OR_UNBD) and kill the others. When none is conclusive within
    ``time_limit`` (+ grace), returns the first backend with its best result or TIME_LIMIT."""
    import time as _time
    from multiprocessing.connection import wait

    if crossed_bounds(md) != ([], []):
        return backends[0], SolveResult("INFEASIBLE")
    workers = {}
    for b in backends:
        with _IDLE_LOCK:
            w = _IDLE.pop() if _IDLE else None
        if w is None or not w.proc.is_alive():
            w = _Worker()
        w.conn.send((b._solve, (md, time_limit, start)))
        workers[w.conn] = (b, w)
    deadline = _time.time() + time_limit + _GRACE
    winner, fallback = None, None
    pending = dict(workers)
    while pending and winner is None:
        ready = wait(list(pending), timeout=max(0.0, deadline - _time.time()))
        if not ready:
            break
        for conn in ready:
            b, w = pending.pop(conn)
            try:
                kind, value = conn.recv()
            except EOFError:
                continue
            w.calls += 1
            with _IDLE_LOCK:
                keep = len(_IDLE) < _MAX_IDLE and md.A.nnz <= _KEEP_MAX_NNZ
                if keep:
                    _IDLE.append(w)
            if not keep:
                w.kill()
            workers[conn] = (b, None)
            if kind == "ok" and value.status in ("OPTIMAL", "INFEASIBLE", "UNBOUNDED", "INF_OR_UNBD"):
                winner = (b, value)
                break
            if kind == "ok" and fallback is None:
                fallback = (b, value)
    for conn, (b, w) in workers.items():
        if w is not None and (conn in pending or w.proc.is_alive()):
            w.kill()  # still running (lost the race or timed out), or died
    return winner or fallback or (backends[0], SolveResult("TIME_LIMIT"))


class Running:
    """``fn(*args)`` started in its own solver process, for a caller that runs several at once and keeps the first
    answer: ``conn`` for multiprocessing.connection.wait, ``result()`` once it is ready (("ok" | "err", value), or
    ("timeout", None) after ``seconds``), ``kill()`` to stop it. Large models get a fresh process, as in
    _with_deadline."""

    def __init__(self, fn, args, seconds: float):
        import time as _time

        md = next((a for a in args if isinstance(a, ModelData)), None)
        self._keep = md is None or md.A.nnz <= _KEEP_MAX_NNZ
        with _IDLE_LOCK:
            w = _IDLE.pop() if _IDLE and self._keep else None
        self.w = w if w is not None and w.proc.is_alive() else _Worker()
        self.w.conn.send((fn, args))
        self.conn = self.w.conn
        self.deadline = _time.time() + seconds
        self.value = None

    def result(self, timeout: float = 0.0):
        import time as _time

        if self.value is None:
            if self.conn.poll(max(0.0, min(timeout, self.deadline - _time.time()))):
                try:
                    self.value = self.conn.recv()
                except EOFError:  # the process died without a result (solver crash, killed for memory)
                    self.value = ("err", RuntimeError("solver process exited with no result"))
                self.w.calls += 1
                if self._keep and self.value[0] == "ok" and self.w.calls < _MAX_CALLS:
                    with _IDLE_LOCK:
                        if len(_IDLE) < _MAX_IDLE:
                            _IDLE.append(self.w)
                            self.w = None
            elif _time.time() >= self.deadline:
                self.value = ("timeout", None)
            if self.value is not None:
                self.kill()
        return self.value

    def kill(self):
        if self.w is not None:
            self.w.kill()
            self.w = None


class _InProcess:
    """A backend's solve in the calling process, for code that already runs inside a solver process (which may not
    start others); that process's own deadline bounds it."""

    def __init__(self, backend: "Backend"):
        self.backend = backend

    def solve(self, md: ModelData, time_limit: float = 60.0, start=None) -> SolveResult:
        return self.backend._solve(md, time_limit, start)


def plan_then_solve(backend: "Backend", lp_backend: "Backend", md: ModelData, time_limit: float,
                    budget: float) -> tuple[SolveResult, "StartingPlan | None"]:
    """In one solver process: a starting plan for the MIP ``md`` from ``lp_backend`` (starting_plan, within
    ``budget``), then ``backend``'s solve from it with the rest of ``time_limit``. The plan stands in for the result
    when the solver ends at its limit with nothing better. Returns (result, plan or None)."""
    import time as _time

    t0 = _time.time()
    plan = starting_plan(md, _InProcess(lp_backend), budget)
    r = backend._solve(md, max(1.0, time_limit - (_time.time() - t0)), None if plan is None else plan.x)
    if plan is not None and r.status in ("TIME_LIMIT", "OTHER") and (
            r.obj is None or (r.obj > plan.obj if md.minimize else r.obj < plan.obj)):
        r = SolveResult("TIME_LIMIT", obj=plan.obj, x=plan.x, bound=r.bound)
    return r, plan


class Backend:
    name = "base"
    # A licensed solver does every step itself: its user may run it because production or policy requires it, so an
    # answer from another solver is not the one asked for. Open-source backends may stand in for each other.
    licensed = False

    def solve(self, md: ModelData, time_limit: float = 60.0, start: np.ndarray | None = None) -> SolveResult:
        """Solve ``md``. A solve still running ``_GRACE`` seconds past ``time_limit`` is killed and
        reported as TIME_LIMIT. Crossed bounds (lower > upper) are infeasible without a solve: solvers
        reject them as input errors (SCIP) or read them differently. ``start``: a plan (a value per column) the
        solver begins from, for a MIP (see starting_plan); a solver that cannot use it ignores it."""
        if crossed_bounds(md) != ([], []):
            return SolveResult("INFEASIBLE")
        try:
            return _with_deadline(self._solve, (md, time_limit, start), time_limit + _GRACE)
        except _TimedOut:
            return SolveResult("TIME_LIMIT")

    def iis(self, md: ModelData, time_limit: float = 60.0) -> IIS:
        rows, cols = crossed_bounds(md)
        if rows or cols:  # a row or variable whose lower limit is above its upper limit is its own IIS
            return IIS(rows=rows[:1], col_bounds=[] if rows else cols[:1], method="crossed bounds",
                       bounds_reported=True)
        try:
            return _with_deadline(self._iis, (md, time_limit), time_limit + _GRACE)
        except _TimedOut as e:
            raise IISNotSupported(f"{self.name} IIS exceeded {time_limit:.0f}s") from e

    def _solve(self, md: ModelData, time_limit: float, start=None) -> SolveResult:
        raise NotImplementedError

    def _iis(self, md: ModelData, time_limit: float) -> IIS:
        raise IISNotSupported(self.name)


def _with_mps(md: ModelData, fn):
    # Named after the process so the parent can remove it when it kills a solve that ran past its deadline.
    with tempfile.TemporaryDirectory(prefix=f"optlens-{os.getpid()}-") as tmp:
        return fn(md.write_mps(os.path.join(tmp, f"{md.name or 'model'}.mps")))


def _remove_temp_of(pid: int) -> None:
    import glob
    import shutil

    for d in glob.glob(os.path.join(tempfile.gettempdir(), f"optlens-{pid}-*")):
        shutil.rmtree(d, ignore_errors=True)


class HiGHSBackend(Backend):
    name = "highs"
    _STATUS = {
        highspy.HighsModelStatus.kOptimal: "OPTIMAL",
        highspy.HighsModelStatus.kInfeasible: "INFEASIBLE",
        highspy.HighsModelStatus.kUnbounded: "UNBOUNDED",
        highspy.HighsModelStatus.kUnboundedOrInfeasible: "INF_OR_UNBD",
        highspy.HighsModelStatus.kTimeLimit: "TIME_LIMIT",
    }

    def __init__(self, mip_rel_gap: float | None = None):
        # None keeps HiGHS's default (1e-4). An answer key needs a tighter gap: on a $60M objective the default allows
        # about $6,000 of slack, enough to make a reported change wrong.
        self.mip_rel_gap = mip_rel_gap

    def _highs(self, md: ModelData, time_limit: float) -> highspy.Highs:
        if md.is_mip and md.quadratic_objective:
            raise UnsupportedModel("HiGHS does not solve mixed-integer QPs; use SCIP")
        h = md.to_highs()
        h.setOptionValue("time_limit", float(time_limit))
        if md.is_mip and self.mip_rel_gap is not None:
            h.setOptionValue("mip_rel_gap", float(self.mip_rel_gap))
        return h

    def farkas_rows(self, md: ModelData, time_limit: float = 60.0) -> list[int] | None:
        """Rows with a nonzero multiplier in HiGHS's proof that the LP ``md`` is infeasible (its dual ray): together,
        with the variable bounds, they are infeasible on their own. None when the LP is not proven infeasible or no
        ray is returned. Presolve is off, since a model presolve rejects comes back without a ray."""
        try:
            return _with_deadline(self._farkas_rows, (md, time_limit), time_limit + _GRACE)
        except _TimedOut:
            return None

    def _farkas_rows(self, md, time_limit):
        h = self._highs(md, time_limit)
        h.setOptionValue("presolve", "off")
        h.run()
        if h.getModelStatus() != highspy.HighsModelStatus.kInfeasible:
            return None
        status, has_ray, ray = h.getDualRay()
        if status != highspy.HighsStatus.kOk or not has_ray:
            return None
        ray = np.asarray(ray)
        return np.flatnonzero(np.abs(ray) > 1e-9 * max(1.0, float(np.abs(ray).max()))).tolist()

    def _solve(self, md, time_limit, start=None):
        h = self._highs(md, time_limit)
        if start is not None and md.is_mip:
            h.setSolution(md.num_cols, np.arange(md.num_cols, dtype=np.int32), np.asarray(start, float))
        h.run()
        status = self._STATUS.get(h.getModelStatus(), "OTHER")
        if h.getInfo().primal_solution_status != 2:  # 2 = feasible
            return SolveResult(status)
        sol = h.getSolution()
        res = SolveResult(status, obj=h.getInfo().objective_function_value, x=np.array(sol.col_value),
                          bound=h.getInfo().mip_dual_bound if md.is_mip else None)
        if not md.is_mip and sol.dual_valid:
            res.row_dual, res.reduced_cost = np.array(sol.row_dual), np.array(sol.col_dual)
        return res

    def _iis(self, md, time_limit):
        if md.is_mip:
            raise IISNotSupported("HiGHS IIS is LP-only")
        h = self._highs(md, time_limit)
        # The default strategy can return an empty IIS; Irreducible gives a true IIS.
        h.setOptionValue("iis_strategy", int(highspy.IisStrategy.kIisStrategyIrreducible))
        h.run()
        status, iis = h.getIis()
        if status != highspy.HighsStatus.kOk or not iis.valid_:
            raise IISNotSupported(f"HiGHS getIis failed: {status}")
        return IIS(rows=[md.row_names[i] for i in iis.row_index_],
                   col_bounds=[md.col_names[j] for j in iis.col_index_], method="highs:irreducible",
                   bounds_reported=True)


class SCIPBackend(Backend):
    name = "scip"
    _STATUS = {"optimal": "OPTIMAL", "infeasible": "INFEASIBLE", "unbounded": "UNBOUNDED",
               "inforunbd": "INF_OR_UNBD", "timelimit": "TIME_LIMIT"}

    def _model(self, path: str, time_limit: float):
        import pyscipopt

        m = pyscipopt.Model()
        m.hideOutput()
        m.readProblem(path)
        m.setParam("limits/time", float(time_limit))
        return m

    def _solve(self, md, time_limit, start=None):
        def run(path):
            m = self._model(path, time_limit)
            if start is not None and md.is_mip:
                sol, at = m.createSol(), {c: j for j, c in enumerate(md.col_names)}
                for v in m.getVars():
                    m.setSolVal(sol, v, float(start[at[v.name]]))
                m.addSol(sol, free=True)
            m.optimize()
            status = self._STATUS.get(m.getStatus(), "OTHER")
            if m.getNSols() == 0:
                return SolveResult(status)
            best = m.getBestSol()
            by_name = {v.name: m.getSolVal(best, v) for v in m.getVars()}
            x = np.array([by_name[c] for c in md.col_names])
            obj = md.objective_value(x)
            # SCIP's bound is on its own objective, which can leave out md's constant: shift it by the same amount
            bound = m.getDualbound() + (obj - m.getSolObjVal(best)) if md.is_mip else None
            return SolveResult(status, obj=obj, x=x, bound=bound)
        return _with_mps(md, run)

    def _iis(self, md, time_limit):
        def run(path):
            m = self._model(path, time_limit)
            # SCIP's IIS finder has no time limit by default; past the limit it returns an
            # infeasible but possibly reducible subset (labelled scip:reduced below).
            m.setParam("iis/time", float(time_limit))
            m.setParam("iis/greedy/timelimperiter", max(1.0, time_limit / 10))
            iis = m.generateIIS()
            if not iis.isSubscipInfeasible():
                raise IISNotSupported("SCIP IIS finder did not certify infeasibility")
            rows = [c.name for c in iis.getSubscip().getConss()]
            method = "scip:irreducible" if iis.isSubscipIrreducible() else "scip:reduced"
            return IIS(rows=rows, method=method)
        return _with_mps(md, run)


class GurobiBackend(Backend):
    """Reference backend. Runs on whatever license gurobipy finds; here the size-limited one."""
    name = "gurobi"
    licensed = True

    def _run(self, md, time_limit, body):
        import gurobipy as gp

        def run(path):
            with gp.Env(params={"OutputFlag": 0}) as env, gp.read(path, env=env) as m:
                m.Params.TimeLimit = float(time_limit)
                m.Params.DualReductions = 0  # definitive INFEASIBLE vs UNBOUNDED
                try:
                    return body(m)
                except gp.GurobiError as e:
                    if e.errno == gp.GRB.Error.SIZE_LIMIT_EXCEEDED:
                        raise LicenseLimit(str(e)) from e
                    raise
        return _with_mps(md, run)

    def _solve(self, md, time_limit, start=None):
        from gurobipy import GRB

        status_map = {GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE", GRB.UNBOUNDED: "UNBOUNDED",
                      GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.TIME_LIMIT: "TIME_LIMIT"}

        def body(m):
            if start is not None and md.is_mip:
                vs = m.getVars()
                at = {c: j for j, c in enumerate(md.col_names)}
                m.setAttr("Start", vs, [float(start[at[v.VarName]]) for v in vs])
            m.optimize()
            status = status_map.get(m.Status, "OTHER")
            if m.SolCount == 0:
                return SolveResult(status)
            by_name = {v.VarName: v.X for v in m.getVars()}
            res = SolveResult(status, obj=m.ObjVal, x=np.array([by_name[c] for c in md.col_names]),
                              bound=m.ObjBound if md.is_mip else None)
            if not md.is_mip and m.Status == GRB.OPTIMAL and md.convex_objective():
                # Gurobi solves a non-convex QP like a MIP and has no duals for it
                pi = {c.ConstrName: c.Pi for c in m.getConstrs()}
                rc = {v.VarName: v.RC for v in m.getVars()}
                res.row_dual = np.array([pi[r] for r in md.row_names])
                res.reduced_cost = np.array([rc[c] for c in md.col_names])
            return res
        return self._run(md, time_limit, body)

    def ranging(self, md: ModelData, time_limit: float = 60.0) -> dict | None:
        """An LP's solution, shadow prices, reduced costs and the ranges over which its optimal basis stays optimal
        (each row's limit, each column's objective coefficient; None for a QP). None when it is not solved to
        optimality in ``time_limit``."""
        try:
            return _with_deadline(self._ranging, (md, time_limit), time_limit + _GRACE)
        except _TimedOut:
            return None

    def _ranging(self, md, time_limit):
        from gurobipy import GRB

        def body(m):
            m.optimize()
            if m.Status != GRB.OPTIMAL:
                return None
            by_row, by_col = {c.ConstrName: c for c in m.getConstrs()}, {v.VarName: v for v in m.getVars()}
            rows, cols = [by_row[r] for r in md.row_names], [by_col[c] for c in md.col_names]

            def get(objs, attr):  # Gurobi writes an infinite range end as +-1e100
                a = np.array(m.getAttr(attr, objs), dtype=float)
                return np.where(a >= GRB.INFINITY, np.inf, np.where(a <= -GRB.INFINITY, -np.inf, a))
            out = {"x": get(cols, "X"), "shadow_price": get(rows, "Pi"), "reduced_cost": get(cols, "RC"),
                   "rhs_range": None, "cost_range": None}
            if not md.quadratic_objective:
                out["rhs_range"] = (get(rows, "SARHSLow"), get(rows, "SARHSUp"))
                out["cost_range"] = (get(cols, "SAObjLow"), get(cols, "SAObjUp"))
            return out
        return self._run(md, time_limit, body)

    def _iis(self, md, time_limit):
        def body(m):
            m.computeIIS()
            return IIS(rows=[c.ConstrName for c in m.getConstrs() if c.IISConstr],
                       col_bounds=[v.VarName for v in m.getVars() if v.IISLB or v.IISUB],
                       method="gurobi:computeIIS", bounds_reported=True)
        return self._run(md, time_limit, body)


BACKENDS = {b.name: b for b in (HiGHSBackend(), SCIPBackend(), GurobiBackend())}


@dataclass
class StartingPlan:
    x: np.ndarray
    obj: float
    how: str
    seconds: float


def starting_plan(md: ModelData, backend: "Backend", budget: float) -> StartingPlan | None:
    """A feasible plan for the MIP ``md`` built from its LP relaxation within ``budget`` seconds, for a solver to start
    from; None when none is found in time. On large MIPs the solvers can spend their whole limit without a usable plan
    (M08: a 68 % gap at 45 and 300 s where this gives 3 % in 4 s). Two tries:
    - rounded: every integer the LP uses rounded up (a 0/1 used at all becomes 1), the LP solved with them fixed. Works
      when a 1 adds capacity (open a site, start a run);
    - repaired: the integers the LP leaves whole kept, the rest solved as a small MIP. For 0/1s that use something up
      or exclude each other (assignments, at most k)."""
    import time

    t0 = time.time()

    def left() -> float:
        return budget - (time.time() - t0)

    j = np.flatnonzero(md.is_int)
    as_lp = replace(md, is_int=np.zeros(md.num_cols, bool))
    lp = backend.solve(as_lp, max(1.0, left()))
    if lp.status != "OPTIMAL" or left() <= 0:
        return None
    v = lp.x[j]
    rounded = np.clip(np.ceil(v - 1e-6), md.col_lb[j], md.col_ub[j])
    whole = np.abs(v - np.round(v)) < 1e-6
    tries = (("LP relaxation, integers rounded up", rounded, np.ones(len(j), bool), False),
             ("LP relaxation, whole values kept, the rest solved", np.round(v), whole, True))
    for how, values, fix, mip in tries:
        if left() <= 1:
            return None
        lb, ub = md.col_lb.copy(), md.col_ub.copy()
        lb[j[fix]] = ub[j[fix]] = values[fix]
        m = replace(md, col_lb=lb, col_ub=ub, is_int=md.is_int if mip else np.zeros(md.num_cols, bool))
        r = backend.solve(m, max(1.0, left()))
        if r.x is not None and r.status in ("OPTIMAL", "TIME_LIMIT"):
            return StartingPlan(r.x, r.obj, how, round(time.time() - t0, 1))
    return None

