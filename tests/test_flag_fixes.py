"""suspicious_values solves the undo of its flags, alone and together. Run from the repo root: python -m unittest tests.test_flag_fixes"""
import unittest

import numpy as np
import scipy.sparse as sp

import optlens as od
from optlens.outliers import flag_fix
from optlens.session import Session, Version


def network() -> od.ModelData:
    # flow f0 -> f1 + f2 (conservation); min-flow rows lb_* (>=) and cap rows ub_* (<=), five of each; every flow at
    # most 7. Two senses are flipped, each causing its own conflict: lb_f0 reads f0 <= 3 against f1 + f2 >= 1 + 3,
    # and ub_f3 reads f3 >= 8 against f3's bound 7. Undoing either alone leaves the other, so only both clear it.
    cols = ("f0", "f1", "f2", "f3", "f4")
    rows, lo, hi, A = [], [], [], []
    def row(name, coefs, low, high):
        rows.append(name)
        lo.append(low)
        hi.append(high)
        A.append([coefs.get(c, 0.0) for c in cols])
    row("conserve", {"f0": 1, "f1": -1, "f2": -1}, 0, 0)
    for c, v in zip(cols, (3, 1, 3, 1, 1)):
        row(f"lb_{c}", {c: 1}, v, od.INF)
    for c in cols:
        row(f"ub_{c}", {c: 1}, -od.INF, 8)
    lo[rows.index("lb_f0")], hi[rows.index("lb_f0")] = -od.INF, 3   # flipped: f0 <= 3
    lo[rows.index("ub_f3")], hi[rows.index("ub_f3")] = 8, od.INF    # flipped: f3 >= 8
    return od.ModelData(name="m", minimize=True, obj=np.ones(5), obj_offset=0.0, A=sp.csr_matrix(np.array(A, float)),
                        row_lo=np.array(lo, float), row_hi=np.array(hi, float), col_lb=np.zeros(5),
                        col_ub=np.full(5, 7.0), is_int=np.zeros(5, bool), col_names=cols, row_names=tuple(rows))


class FlagFixes(unittest.TestCase):
    def test_a_sense_flip_keeps_its_limit(self):
        md = network()
        self.assertEqual(flag_fix(md, {"kind": "constraint sense", "name": "ub_f3"}),
                         {"action": "set_rhs", "name": "ub_f3", "lower": None, "upper": 8.0})
        self.assertEqual(flag_fix(md, {"kind": "constraint sense", "name": "lb_f0"}),
                         {"action": "set_rhs", "name": "lb_f0", "lower": 3.0, "upper": None})

    def test_an_integer_bound_goes_back_to_a_whole_number(self):
        md = network()
        md = od.ModelData(**{**md.__dict__, "is_int": np.ones(5, bool)})
        fix = flag_fix(md, {"kind": "variable upper bound", "name": "f0", "typical": 28.5})
        self.assertEqual(fix, {"action": "set_bounds", "name": "f0", "upper": 28.0})

    def test_two_flips_are_found_together(self):
        out = Session({"v0": Version(network(), None, "o")}).suspicious_values()
        self.assertIn("lb_f0: <= 3 -> >= 3: still INFEASIBLE", out)
        self.assertIn("ub_f3: >= 8 -> <= 8: still INFEASIBLE", out)
        self.assertIn("together: lb_f0: <= 3 -> >= 3; ub_f3: >= 8 -> <= 8: OPTIMAL", out)


if __name__ == "__main__":
    unittest.main()
