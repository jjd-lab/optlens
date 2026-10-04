"""compare_versions against the closest optimal plan, not the solver's choice among ties.
Run from the repo root: python -m unittest tests.test_closest"""
import unittest
from dataclasses import replace

import numpy as np
import scipy.sparse as sp

import optlens as od
from optlens.closest import closest_optimal
from optlens.diagnose import default_backend
from optlens.session import Session, Version


def three_sources(demand: float, minimize: bool = True) -> od.ModelData:
    # ship(a), ship(b), ship(c) all cost 1 and together meet demand: every split is optimal
    sign = 1.0 if minimize else -1.0
    return od.ModelData(name="m", minimize=minimize, obj=np.full(3, sign), obj_offset=0.0,
                        A=sp.csr_matrix(np.ones((1, 3))), row_lo=np.array([demand]), row_hi=np.array([od.INF]),
                        col_lb=np.zeros(3), col_ub=np.full(3, od.INF), is_int=np.zeros(3, bool),
                        col_names=("ship(a)", "ship(b)", "ship(c)"), row_names=("demand",))


def session(minimize: bool = True, x1=(0.0, 6.0, 6.0)) -> Session:
    """v0 ships 10 from a; v1 needs 12 and its 'solver' plan (fixed here) is x1."""
    sign = 1.0 if minimize else -1.0
    v0 = Version(three_sources(10, minimize), None, "original")
    v0.result = od.SolveResult("OPTIMAL", sign * 10, np.array([10.0, 0, 0]))
    v1 = Version(three_sources(12, minimize), "v0", "demand 12")
    v1.result = od.SolveResult("OPTIMAL", sign * 12, np.array(x1))
    return Session({"v0": v0, "v1": v1}, race=False)


class Closest(unittest.TestCase):
    # The nearest plans are [12, 0, 0], [10, 2, 0] and [10, 0, 2] (each moves 2 units): any of them keeps 10 on a
    # and changes one variable, where the solver's plan changes all three.
    def assert_nearest(self, x):
        self.assertAlmostEqual(x.sum(), 12, places=6)
        self.assertGreaterEqual(x[0], 10 - 1e-6)
        self.assertEqual(int((np.abs(x - [10, 0, 0]) > 1e-6).sum()), 1)

    def test_closest_plan_keeps_the_old_split(self):
        md = three_sources(12)
        status, x = closest_optimal(md, np.array([10.0, 0, 0]), 12.0, default_backend(md), 30)
        self.assertEqual(status, "OPTIMAL")
        self.assert_nearest(x)

    def test_maximize_holds_the_objective_too(self):
        md = three_sources(12, minimize=False)
        _, x = closest_optimal(md, np.array([10.0, 0, 0]), -12.0, default_backend(md), 30)
        self.assert_nearest(x)

    def test_compare_versions_reports_forced_changes_and_the_ties(self):
        for minimize in (True, False):
            out = session(minimize).compare_versions("v0", "v1")
            self.assertIn("it changes 1 variable. The solver's own plan changes 3: the other 2 are a choice", out)
            self.assertIn("variables changed: 1 of 3", out)

    def test_no_note_when_the_solver_plan_is_already_closest(self):
        out = session(x1=(12.0, 0, 0)).compare_versions("v0", "v1")
        self.assertNotIn("plan compared", out)
        self.assertIn("ship: 10 0 0 -> 12 0 0", out)

    def test_quadratic_models_keep_the_plain_diff(self):
        md = replace(three_sources(12), Q=sp.identity(3, format="csc"))
        self.assertEqual(closest_optimal(md, np.zeros(3), 0.0, default_backend(md), 30), ("QUADRATIC", None))


if __name__ == "__main__":
    unittest.main()
