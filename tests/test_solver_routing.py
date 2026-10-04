"""Solver routing: capability rules, a race on a MIP's first solve, the choice kept for the session.
Run from the repo root: python -m unittest tests.test_solver_routing"""
import unittest
from dataclasses import replace

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
