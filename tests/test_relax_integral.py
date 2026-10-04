"""Relaxation amounts on integer rows are whole units. Run from the repo root: python -m unittest tests.test_relax_integral"""
import unittest

import numpy as np
import scipy.sparse as sp

import optlens as od
from optlens.diagnose import _snap_integral


def binaries(integer: bool) -> od.ModelData:
    # three 0/1 (or [0, 1] continuous) variables that must sum to at least 4: short by exactly 1
    return od.ModelData(
        name="pick", minimize=True, obj=np.zeros(3), obj_offset=0.0,
        A=sp.csr_matrix(np.ones((1, 3))), row_lo=np.array([4.0]), row_hi=np.array([od.INF]),
        col_lb=np.zeros(3), col_ub=np.ones(3), is_int=np.full(3, integer),
        col_names=("a", "b", "c"), row_names=("need",),
    )


class TestRelaxIntegral(unittest.TestCase):
    def test_solver_tolerance_is_snapped_to_a_whole_unit(self):
        # what HiGHS (0.999999) and Gurobi (0.99999) reported on neos859080: less than the unit the row needs
        for reported in (0.999999, 0.99999, 1.00000001):
            r = _snap_integral(binaries(True), [{"type": "constraint", "name": "need",
                                                 "direction": "RHS must be decreased by", "value": reported}])
            self.assertEqual(r[0]["value"], 1.0)

    def test_continuous_rows_are_left_alone(self):
        r = _snap_integral(binaries(False), [{"type": "constraint", "name": "need",
                                              "direction": "RHS must be decreased by", "value": 0.4}])
        self.assertEqual(r[0]["value"], 0.4)

    def test_snapped_repair_solves(self):
        md = binaries(True)
        for b in ("highs", "scip"):
            relax = od.feas_relax(md, od.BACKENDS[b])
            self.assertEqual(relax.total_violation, 1.0)
            self.assertTrue(od.BACKENDS[b].solve(od.apply_relaxation(md, relax.relaxations)).feasible)


if __name__ == "__main__":
    unittest.main()
