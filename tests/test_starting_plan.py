"""Starting plans for large MIPs (M08, F46). Run from the repo root: python -m unittest tests.test_starting_plan"""
import importlib.util
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

import optlens as od
from optlens import session as sess
from optlens.session import Session, Version


def lot_sizing() -> od.ModelData:
    """Two weeks of demand 30 and 50; a run costs 100 and allows up to 100 units; holding 1 a unit. The LP runs a
    fraction of each week; rounding the runs up gives a feasible plan."""
    # columns: run1, run2, make1, make2, stock1
    A = np.array([[-100, 0, 1, 0, 0], [0, -100, 0, 1, 0],    # make <= 100 run
                  [0, 0, 1, 0, -1], [0, 0, 0, 1, 1]], float)  # week 1: make1 - stock1 = 30; week 2: make2 + stock1 = 50
    return od.ModelData(name="lots", minimize=True, obj=np.array([100, 100, 0, 0, 1.0]), obj_offset=0.0,
                        A=sp.csr_matrix(A), row_lo=np.array([-od.INF, -od.INF, 30, 50]),
                        row_hi=np.array([0, 0, 30, 50.0]), col_lb=np.zeros(5),
                        col_ub=np.array([1, 1, od.INF, od.INF, od.INF]), is_int=np.array([1, 1, 0, 0, 0], bool),
                        col_names=("run1", "run2", "make1", "make2", "stock1"), row_names=("cap1", "cap2", "dem1", "dem2"))


def odd_cycle() -> od.ModelData:
    """Pick items, no two of three: the LP takes half of each, so rounding up breaks every row; repaired, it solves."""
    A = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]], float)
    return od.ModelData(name="cycle", minimize=False, obj=np.array([1.0, 1.0, 1.0]), obj_offset=0.0,
                        A=sp.csr_matrix(A), row_lo=np.full(3, -od.INF), row_hi=np.ones(3), col_lb=np.zeros(3),
                        col_ub=np.ones(3), is_int=np.ones(3, bool), col_names=("a", "b", "c"), row_names=("ab", "bc", "ca"))


class TestStartingPlan(unittest.TestCase):
    def test_runs_rounded_up_give_a_plan(self):
        plan = od.starting_plan(lot_sizing(), od.HiGHSBackend(), 10)
        self.assertIn("rounded up", plan.how)
        self.assertEqual(plan.obj, 200.0)  # both runs on, each week makes its own demand
        self.assertTrue(np.allclose(plan.x[:2], 1))

    def test_a_rounding_that_breaks_a_limit_is_repaired(self):
        plan = od.starting_plan(odd_cycle(), od.HiGHSBackend(), 10)
        self.assertIn("rest solved", plan.how)
        self.assertEqual(plan.obj, 1.0)

    def test_solvers_take_a_start(self):
        md, start = lot_sizing(), np.array([1, 1, 30, 50, 0.0])
        backends = [od.HiGHSBackend()] + ([od.SCIPBackend()] if importlib.util.find_spec("pyscipopt") else [])
        for bk in backends:
            r = bk.solve(md, 10, start=start)
            self.assertEqual((bk.name, r.status, r.obj), (bk.name, "OPTIMAL", 150.0))  # one run in week 1, 50 held: 100 + 50

    def test_a_large_mip_starts_from_a_plan_and_says_so(self):
        with mock.patch.object(sess, "START_MIN_ROWS", 1):
            s = Session({"v0": Version(lot_sizing(), None, "original")})
            r = s.solved("v0")
        self.assertEqual((r.status, r.obj), ("OPTIMAL", 150.0))
        self.assertIn("started from a plan built from the LP relaxation, integers rounded up (200, ", s.route_note)

    def test_the_routing_note_keeps_the_start(self):
        # the first solve of a large MIP races HiGHS and SCIP; its routing note must not replace the start's
        md = lot_sizing()
        with mock.patch.object(sess, "START_MIN_ROWS", 1), mock.patch.object(sess, "RACE_MIN_SIZE", 1):
            s = Session({"v0": Version(md, None, "original")})
            s.solved("v0")
        self.assertIn("raced on this model", s.route_note)
        self.assertIn("; started from a plan built from the LP relaxation", s.route_note)

    def test_a_small_mip_solves_as_before(self):
        s = Session({"v0": Version(lot_sizing(), None, "original")})
        self.assertEqual(s.solved("v0").obj, 150.0)
        self.assertNotIn("started from", s.route_note)


if __name__ == "__main__":
    unittest.main()
