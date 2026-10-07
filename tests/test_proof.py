"""What a MIP result proves: the solver's bound and gap, and the range an objective change between two plans not proven
optimal must lie in. Run from the repo root: python -m unittest tests.test_proof"""
import importlib.util
import unittest

import numpy as np
import scipy.sparse as sp

import optlens as od
from optlens.session import _change_range, _proof


def knapsack(minimize: bool) -> od.ModelData:
    # pick 2 of 3 items; a constant term checks that the bound is on the same objective as obj
    return od.ModelData(
        name="k", minimize=minimize, obj=np.array([3.0, 5.0, 4.0]), obj_offset=10.0,
        A=sp.csr_matrix(np.ones((1, 3))), row_lo=np.array([2.0]), row_hi=np.array([2.0]),
        col_lb=np.zeros(3), col_ub=np.ones(3), is_int=np.ones(3, bool), col_names=("a", "b", "c"), row_names=("pick",),
    )


class TestBound(unittest.TestCase):
    def test_every_backend_reports_a_mip_bound_on_the_model_objective(self):
        for name in ("highs", "scip", "gurobi"):
            if (name == "scip" and importlib.util.find_spec("pyscipopt") is None) or (name == "gurobi" and not od.gurobi_usable()):
                continue
            for minimize, best in ((True, 17.0), (False, 19.0)):
                with self.subTest(solver=name, minimize=minimize):
                    r = od.BACKENDS[name].solve(knapsack(minimize), 30)
                    self.assertEqual(r.status, "OPTIMAL")
                    self.assertAlmostEqual(r.obj, best)
                    self.assertAlmostEqual(r.bound, best, places=6)
                    self.assertAlmostEqual(r.gap, 0.0, places=6)

    def test_an_lp_has_no_bound(self):
        lp = knapsack(True)
        lp = od.ModelData(**{**lp.__dict__, "is_int": np.zeros(3, bool)})
        self.assertIsNone(od.BACKENDS["highs"].solve(lp, 30).bound)


class TestProofText(unittest.TestCase):
    def test_unproven_plan_shows_bound_and_gap(self):
        r = od.SolveResult("TIME_LIMIT", obj=103.0, x=np.zeros(1), bound=100.0)
        self.assertEqual(_proof(r), " (best plan found, not proven optimal: bound 100, gap 2.91%)")
        self.assertEqual(_proof(od.SolveResult("OPTIMAL", obj=1.0, x=np.zeros(1), bound=1.0)), "")

    def test_a_time_limit_without_a_plan_is_not_infeasibility(self):
        self.assertIn("not proof that none exists", _proof(od.SolveResult("TIME_LIMIT")))
        self.assertEqual(_proof(od.SolveResult("INFEASIBLE")), "")

    def test_change_range_covers_both_plans_and_their_bounds(self):
        base = od.SolveResult("TIME_LIMIT", obj=100.0, x=np.zeros(1), bound=95.0)
        what_if = od.SolveResult("TIME_LIMIT", obj=104.0, x=np.zeros(1), bound=98.0)
        # optimum of base in [95, 100], of what_if in [98, 104]: the change is between 98 - 100 and 104 - 95
        self.assertIn("between -2 and 9", _change_range(base, what_if))
        proven = od.SolveResult("OPTIMAL", obj=100.0, x=np.zeros(1))
        self.assertIn("between -2 and 4", _change_range(proven, what_if))
        self.assertEqual(_change_range(proven, od.SolveResult("OPTIMAL", obj=101.0, x=np.zeros(1))), "")
        self.assertEqual(_change_range(base, od.SolveResult("TIME_LIMIT", obj=99.0, x=np.zeros(1))), "")


if __name__ == "__main__":
    unittest.main()
