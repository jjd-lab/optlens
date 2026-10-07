"""Solver routing: capability rules, a race on a MIP's first solve, the choice kept for the session.
Run from the repo root: python -m unittest tests.test_solver_routing"""
import unittest
from dataclasses import replace
from unittest import mock

import numpy as np
import scipy.sparse as sp

import optlens as od
import optlens.session as ses
from optlens.session import Session, Version


def lp() -> od.ModelData:
    return od.ModelData(
        name="m", minimize=True, obj=np.array([1.0, 1.0]), obj_offset=0.0,
        A=sp.csr_matrix(np.array([[1.0, 1.0]])), row_lo=np.array([2.0]), row_hi=np.array([od.INF]),
        col_lb=np.zeros(2), col_ub=np.full(2, 10.0), is_int=np.zeros(2, bool), col_names=("x", "y"), row_names=("need",),
    )


class TestSolverRouting(unittest.TestCase):
    def test_race_returns_a_conclusive_result(self):
        b, r = od.race(replace(lp(), is_int=np.ones(2, bool)), [od.BACKENDS["highs"], od.BACKENDS["scip"]], 30)
        self.assertIn(b.name, ("highs", "scip"))
        self.assertEqual(r.status, "OPTIMAL")
        self.assertAlmostEqual(r.obj, 2.0)

    def test_a_large_mip_goes_to_highs_without_a_race(self):
        mip = replace(lp(), is_int=np.ones(2, bool))
        old = ses.RACE_MIN_SIZE, ses.RACE_MAX_NNZ
        ses.RACE_MIN_SIZE, ses.RACE_MAX_NNZ = 0, 1  # this 2-nonzero MIP counts as large
        try:
            s = Session({"v0": Version(mip, None, "original")}, default_solver="scip")
            s.solved("v0")
        finally:
            ses.RACE_MIN_SIZE, ses.RACE_MAX_NNZ = old
        self.assertEqual(s.route, "highs")
        self.assertIn("without a race", s.route_note)

    def test_lp_routes_to_highs(self):
        s = Session({"v0": Version(lp(), None, "original")})
        s.solved("v0")
        self.assertEqual(s.route, "highs")

    def test_no_verdict_is_tried_on_the_other_solver(self):
        # E64: HiGHS ran out its limit on an infeasible LP (OTHER) that SCIP decides in seconds
        bad = replace(lp(), col_ub=np.full(2, 0.5))  # x + y >= 2 with both at most 0.5
        s = Session({"v0": Version(bad, None, "original")})
        with mock.patch.object(od.BACKENDS["highs"], "solve", return_value=od.SolveResult("OTHER")):
            r = s.solved("v0")
            self.assertEqual((r.status, s.route, s.get("v0").solver), ("INFEASIBLE", "highs", "scip"))
            self.assertIn("highs gave no verdict on v0 (OTHER), so scip solved it: INFEASIBLE", s.route_note)
            self.assertIn("IIS (scip", s.compute_iis())  # the IIS tools on v0 use the solver that decided it
        s.modify_and_resolve(changes=[{"action": "set_bounds", "name": "x", "upper": 2.0}])
        self.assertEqual((s.get("v1").solver, s.solved("v1").status), (None, "OPTIMAL"))  # the route, HiGHS
        spent = Session({"v0": Version(bad, None, "original")}, time_limit=45, max_time_limit=48)
        with mock.patch.object(od.BACKENDS["highs"], "solve", return_value=od.SolveResult("OTHER")), \
                mock.patch.object(ses.time, "time", side_effect=[0.0] + [46.0] * 50):
            self.assertEqual(spent.solved("v0").status, "OTHER")
        self.assertIn("pass solver='scip' to try it", spent.route_note)

    def test_a_large_cut_short_iis_points_to_the_relaxation(self):
        s = Session({"v0": Version(replace(lp(), col_ub=np.full(2, 0.5)), None, "original")})
        big = od.IIS(rows=["need"] * 300, method="scip:reduced")
        with mock.patch.object(s, "iis", return_value=([], big)), mock.patch.object(ses, "_conflict_rows", return_value=[]):
            self.assertIn("read better with feasibility_relaxation", s.compute_iis())

    def test_route_note_is_shown_once(self):
        s = Session({"v0": Version(lp(), None, "original")})
        first, _ = s.call("get_model_overview", {})
        second, _ = s.call("get_model_overview", {})
        self.assertIn("solver for this model: highs (LP)", first)
        self.assertNotIn("solver for this model", second)

    def test_call_returns_errors_as_results(self):
        s = Session({"v0": Version(lp(), None, "original")})
        out, err = s.call("get_model_overview", {"version": "v9"})
        self.assertTrue(err)
        self.assertIn("v9", out)

    def test_miqp_needs_scip(self):
        m = replace(lp(), is_int=np.ones(2, bool), Q=sp.csr_matrix(np.eye(2)))
        self.assertEqual(ses._capability(m)[0], "scip")
        self.assertEqual(ses._backend(None, m, "highs").name, "scip")

    def test_explicit_solver_wins(self):
        self.assertEqual(ses._backend("scip", lp(), "highs").name, "scip")


if __name__ == "__main__":
    unittest.main()


class TestSmallerIIS(unittest.TestCase):
    def test_a_huge_iis_is_replaced_by_a_smaller_conflict(self):
        # E66: Gurobi returned 721 rows (720 demand limits and the floor) where a 13-row conflict exists
        n = 10
        A = sp.csr_matrix(np.vstack([np.ones(n), np.eye(n), np.ones(n)]))
        md = od.ModelData(
            name="m", minimize=True, obj=np.ones(n), obj_offset=0.0, A=A,
            row_lo=np.array([20.0] + [-od.INF] * (n + 1)), row_hi=np.array([od.INF] + [1.0] * n + [5.0]),
            col_lb=np.zeros(n), col_ub=np.full(n, od.INF), is_int=np.zeros(n, bool), col_names=tuple(f"x{i}" for i in range(n)),
            row_names=("floor", *(f"cap[{i}]" for i in range(n)), "total"))
        big = od.IIS(rows=["floor", *(f"cap[{i}]" for i in range(n))], method="gurobi:computeIIS")
        real, calls = od.get_iis, []

        def first_big(*args, **kwargs):
            calls.append(1)
            return big if len(calls) == 1 else real(*args, **kwargs)

        s = Session({"v0": Version(md, None, "original")})
        with mock.patch.object(ses, "LARGE_IIS_ROWS", 5), mock.patch.object(od, "get_iis", side_effect=first_big):
            _, iis = s.iis("v0")
        self.assertEqual(sorted(iis.rows), ["floor", "total"])
        self.assertIn("found with cap left out; the first one the solver returned had 11 rows", iis.note)


class TestFixMenuCallAllowance(unittest.TestCase):
    def test_the_iis_search_counts_against_the_calls_allowance(self):
        # E74: through the server fix_menu first spent 48.7 s on the IIS, then took its full limit for the menu
        import time

        s = Session({"v0": Version(replace(lp(), col_ub=np.full(2, 0.5)), None, "original")}, max_time_limit=3.0)

        def slow_iis(*args, **kwargs):
            time.sleep(1.0)
            return [], od.IIS(rows=["need"], method="highs:irreducible")

        with mock.patch.object(s, "iis", side_effect=slow_iis):
            self.assertIn("call fix_menu again", s.fix_menu())
