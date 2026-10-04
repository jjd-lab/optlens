"""Rationality checks on repairs of a small model. Run from the repo root: python -m unittest tests.test_checks"""
import unittest

import numpy as np
import scipy.sparse as sp

import optlens as od
from optlens.checks import Policy, check_repair


def model() -> od.ModelData:
    # max 3x + 2y  s.t.  cap: x + y <= 4,  demand: x >= 5 (conflict),  x, y integer in [0, 10]
    return od.ModelData(
        name="m", minimize=False, obj=np.array([3.0, 2.0]), obj_offset=0.0,
        A=sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 0.0]])),
        row_lo=np.array([-od.INF, 5.0]), row_hi=np.array([4.0, od.INF]),
        col_lb=np.zeros(2), col_ub=np.full(2, 10.0), is_int=np.array([True, True]),
        col_names=("x", "y"), row_names=("cap", "demand"),
    )


def result(md):
    return od.BACKENDS["scip"].solve(md)


def by_name(checks):
    return {c["check"]: c["passed"] for c in checks}


class TestChecks(unittest.TestCase):
    def test_modest_repair_passes(self):
        md = model()
        fixed = md.set_row_bounds("cap", hi=5.0)
        c = by_name(check_repair(md, fixed, result(fixed), Policy(max_rel_change=0.5)))
        self.assertTrue(all(c.values()), c)

    def test_hard_rule_and_tolerance(self):
        md = model()
        fixed = md.set_row_bounds("cap", hi=10.0)  # +150%
        c = by_name(check_repair(md, fixed, result(fixed), Policy(hard={"cap"}, max_rel_change=0.5)))
        self.assertTrue(c["solves"])
        self.assertFalse(c["hard_rules_untouched"])
        self.assertFalse(c["change_within_tolerance"])

    def test_drop_and_fractional_integer_bound(self):
        md = model()
        dropped = md.drop_rows(["demand"])
        self.assertFalse(by_name(check_repair(md, dropped, result(dropped)))["no_unapproved_drops"])
        frac = md.set_row_bounds("demand", lo=4.0).set_col_bounds("y", ub=0.5)
        self.assertFalse(by_name(check_repair(md, frac, result(frac)))["integers_keep_meaning"])

    def test_objective_bound(self):
        md = model()
        fixed = md.set_row_bounds("demand", lo=0.0)
        r = result(fixed)
        self.assertFalse(by_name(check_repair(md, fixed, r, Policy(max_obj_rel_change=0.1),
                                              reference_objective=100.0))["objective_within_bound"])

    def test_solver_infinity_is_infinite(self):
        # an agent writes "no limit" as 1e30 (Gurobi) or 1e20 (HiGHS, SCIP); it must not count as a huge finite change
        md = model().set_row_bounds("demand", lo=-1e30, hi=5.0)
        self.assertEqual(md.row_lo[1], -od.INF)
        self.assertEqual(model().set_col_bounds("x", ub=1e20).col_ub[0], od.INF)


if __name__ == "__main__":
    unittest.main()
