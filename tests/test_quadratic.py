"""Quadratic objectives: a convex QP solves on HiGHS, a mixed-integer or non-convex one goes to SCIP; quadratic
constraints are rejected. Run from the repo root: python -m unittest tests.test_quadratic"""
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import scipy.sparse as sp

import optlens as od
from optlens.session import Session, Version


def qp(q: float, minimize: bool = True) -> od.ModelData:
    """min x + 0.5 q x^2 with 0 <= x <= 3 and x >= 1: convex for q >= 0 (minimizing), concave for q < 0."""
    return od.ModelData(
        name="qp", minimize=minimize, obj=np.array([1.0]), obj_offset=0.0, A=sp.csr_matrix(np.array([[1.0]])),
        row_lo=np.array([1.0]), row_hi=np.array([od.INF]), col_lb=np.zeros(1), col_ub=np.array([3.0]),
        is_int=np.zeros(1, bool), col_names=("x",), row_names=("need",), Q=sp.csc_matrix(np.array([[q]])),
    )


class TestQuadratic(unittest.TestCase):
    def test_convexity_follows_the_sense(self):
        self.assertTrue(qp(2.0).convex_objective())
        self.assertFalse(qp(-2.0).convex_objective())
        self.assertTrue(qp(-2.0, minimize=False).convex_objective())   # maximizing a concave objective
        self.assertTrue(replace(qp(2.0), Q=None).convex_objective())    # linear

    def test_a_convex_qp_solves_on_highs(self):
        s = Session({"v0": Version(qp(2.0), None, "original")})
        r = s.solved("v0")
        self.assertEqual((r.status, s.route), ("OPTIMAL", "highs"))
        self.assertAlmostEqual(r.obj, 2.0)  # x = 1: 1 + 0.5 * 2
        self.assertIn("convex QP", s.route_note)

    def test_a_non_convex_qp_goes_to_scip(self):
        s = Session({"v0": Version(qp(-2.0), None, "original")})
        r = s.solved("v0")
        self.assertEqual((r.status, s.route), ("OPTIMAL", "scip"))
        self.assertAlmostEqual(r.obj, -6.0, places=6)  # x = 3: 3 - 0.5 * 2 * 9
        self.assertIn("non-convex quadratic objective", s.route_note)
        out, err = s.call("sensitivity_report", {"kind": "constraints"})
        self.assertFalse(err, out)
        self.assertIn("non-convex quadratic objective", out)

    def test_a_mixed_integer_qp_goes_to_scip(self):
        s = Session({"v0": Version(replace(qp(2.0), is_int=np.ones(1, bool)), None, "original")})
        self.assertEqual((s.solved("v0").status, s.route), ("OPTIMAL", "scip"))

    @unittest.skipUnless(od.gurobi_usable(), "Gurobi cannot run here")
    def test_gurobi_solves_a_non_convex_qp(self):
        r = od.BACKENDS["gurobi"].solve(qp(-2.0))
        self.assertEqual(r.status, "OPTIMAL")
        self.assertAlmostEqual(r.obj, -6.0, places=6)
        self.assertIsNone(r.row_dual)

    def test_a_quadratic_constraint_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "qcon.lp"
            path.write_text("Minimize\n obj: x\nSubject To\n q: [ x ^ 2 ] <= 4\nBounds\n x >= -10\nEnd\n")
            with self.assertRaisesRegex(od.UnsupportedModel, "quadratic constraint"):
                od.load(path)


if __name__ == "__main__":
    unittest.main()
