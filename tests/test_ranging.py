"""Sensitivity ranges: a row with slack can move its limit freely up to its activity; Gurobi gives the same ranges.
Run from the repo root: python -m unittest tests.test_ranging"""
import unittest

import numpy as np
import scipy.sparse as sp

import optlens as od
from optlens.sensitivity import sensitivity


def lp() -> od.ModelData:
    """max x + 2y with x + y <= 4, x <= 3, y <= 10: the optimum is x = 0, y = 4, and only the total binds."""
    return od.ModelData(
        name="m", minimize=False, obj=np.array([1.0, 2.0]), obj_offset=0.0,
        A=sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]])),
        row_lo=np.full(3, -od.INF), row_hi=np.array([4.0, 3.0, 10.0]),
        col_lb=np.zeros(2), col_ub=np.full(2, od.INF), is_int=np.zeros(2, bool), col_names=("x", "y"),
        row_names=("total", "x_cap", "y_cap"),
    )


class TestRanging(unittest.TestCase):
    def test_a_row_with_slack_ranges_from_its_activity(self):
        s = sensitivity(lp())  # optimum y = 4, x = 0: total binds, x_cap and y_cap have slack
        self.assertEqual(s.status, "OPTIMAL")
        np.testing.assert_allclose(s.x, [0.0, 4.0], atol=1e-9)
        self.assertEqual((s.rhs_range[0][1], s.rhs_range[1][1]), (0.0, np.inf))   # x_cap: down to x's 0
        self.assertEqual((s.rhs_range[0][2], s.rhs_range[1][2]), (4.0, np.inf))   # y_cap: down to y's 4
        self.assertAlmostEqual(s.shadow_price[0], 2.0)
        self.assertEqual((s.rhs_range[0][0], s.rhs_range[1][0]), (0.0, 10.0))     # total: y alone carries it, to y_cap

    @unittest.skipUnless(od.gurobi_usable(), "Gurobi cannot run here")
    def test_gurobi_gives_the_same_ranges(self):
        h, g = sensitivity(lp()), sensitivity(lp(), backend=od.BACKENDS["gurobi"])
        pairs = [(h.shadow_price, g.shadow_price), (h.reduced_cost, g.reduced_cost), *zip(h.rhs_range, g.rhs_range),
                 *zip(h.cost_range, g.cost_range)]
        for a, b in pairs:  # HiGHS's cost ranges also have an entry per row after the columns
            np.testing.assert_allclose(a[:len(b)], b, atol=1e-9)


if __name__ == "__main__":
    unittest.main()
