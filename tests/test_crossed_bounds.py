"""Crossed bounds (lower > upper) are infeasible and their own IIS. Run from the repo root: python -m unittest tests.test_crossed_bounds"""
import unittest

import numpy as np
import scipy.sparse as sp

import optlens as od


def model(col_ub: float = -0.8, row_hi: float = 10.0) -> od.ModelData:
    # x in [0, col_ub], cap: 1 <= x + y <= row_hi
    return od.ModelData(
        name="m", minimize=True, obj=np.ones(2), obj_offset=0.0, A=sp.csr_matrix(np.array([[1.0, 1.0]])),
        row_lo=np.array([1.0]), row_hi=np.array([row_hi]), col_lb=np.zeros(2), col_ub=np.array([col_ub, 5.0]),
        is_int=np.zeros(2, bool), col_names=("x", "y"), row_names=("cap",),
    )


class TestCrossedBounds(unittest.TestCase):
    def test_crossed_column_is_infeasible_on_every_backend(self):
        for name in ("highs", "scip"):
            self.assertEqual(od.BACKENDS[name].solve(model()).status, "INFEASIBLE", name)
            iis = od.get_iis(model(), od.BACKENDS[name])
            self.assertEqual((iis.rows, iis.col_bounds, iis.method), ([], ["x"], "crossed bounds"))

    def test_crossed_row(self):
        iis = od.get_iis(model(col_ub=1.0, row_hi=0.5), od.BACKENDS["scip"])
        self.assertEqual(iis.rows, ["cap"])

    def test_consistent_bounds_solve_normally(self):
        self.assertEqual(od.BACKENDS["highs"].solve(model(col_ub=1.0)).status, "OPTIMAL")


if __name__ == "__main__":
    unittest.main()
