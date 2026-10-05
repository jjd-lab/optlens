"""Suspicious-value flags. Run from the repo root: python -m unittest tests.test_outliers"""
import unittest
from dataclasses import replace

import numpy as np
import scipy.sparse as sp

import optlens as od
from optlens.outliers import family_outliers, pattern_breaks


def model(rhs):
    n = len(rhs)
    return od.ModelData(
        name="m", minimize=True, obj=np.zeros(n), obj_offset=0.0, A=sp.identity(n, format="csr"),
        row_lo=np.full(n, -od.INF), row_hi=np.array(rhs, float), col_lb=np.zeros(n), col_ub=np.full(n, 10.0),
        is_int=np.zeros(n, bool), col_names=tuple(f"x[{i}]" for i in range(n)),
        row_names=tuple(f"cap[{i}]" for i in range(n)),
    )


class TestOutliers(unittest.TestCase):
    def test_lone_value_among_equal_siblings(self):
        flags = family_outliers(model([0, 0, 0, 0, -1, 0]))
        self.assertEqual([f["name"] for f in flags], ["cap[4]"])

    def test_three_members_flag_only_a_near_miss(self):
        # the tutorial's 0.99 among two 1s; a genuinely different third member is not a pattern break
        self.assertEqual([f["name"] for f in family_outliers(model([1, 1, 0.99]))], ["cap[2]"])
        self.assertEqual([f["name"] for f in family_outliers(model([1, 10, 1]))], ["cap[1]"])
        self.assertEqual(family_outliers(model([100, 100, 150])), [])

    def test_repeated_values_are_a_pattern(self):
        self.assertEqual(family_outliers(model([0, 0, 0, 0, 0, 0, 100, 100, 200, 200])), [])

    def test_iis_rows_rank_first(self):
        flags = family_outliers(model([0, 0, 0, 0, 0, 0, 0, 0, 5, 7]), iis_rows={"cap[9]"})
        self.assertEqual(flags[0]["name"], "cap[9]")
        self.assertTrue(flags[0]["in_iis"])

    def test_the_farthest_value_ranks_first_and_survives_the_cut(self):
        rhs = [100, 101, 99, 100, 102, 98, 100, 103, 97, 300, 104, 30000]
        self.assertEqual([f["name"] for f in family_outliers(model(rhs))], ["cap[11]", "cap[9]"])
        self.assertEqual([f["name"] for f in family_outliers(model(rhs), max_flags=1)], ["cap[11]"])

    def test_lone_flipped_sense_is_flagged(self):
        md = model([1] * 6)
        row_lo, row_hi = md.row_lo.copy(), md.row_hi.copy()
        row_lo[3], row_hi[3] = 1.0, od.INF  # <= 1 turned into >= 1
        flags = family_outliers(replace(md, row_lo=row_lo, row_hi=row_hi))
        self.assertEqual([(f["kind"], f["name"]) for f in flags], [("constraint sense", "cap[3]")])

    def test_mixed_or_tiny_groups_are_not_flipped(self):
        md = model([1] * 6)
        row_lo, row_hi = md.row_lo.copy(), md.row_hi.copy()
        row_lo[:3], row_hi[:3] = 1.0, od.INF  # three and three: a mix, not a flip
        self.assertEqual(family_outliers(replace(md, row_lo=row_lo, row_hi=row_hi)), [])
        row_lo, row_hi = md.row_lo[:3].copy(), md.row_hi[:3].copy()
        row_lo[0], row_hi[0] = 1.0, od.INF  # too few siblings to define a pattern
        tiny = replace(model([1] * 3), row_lo=row_lo, row_hi=row_hi)
        self.assertEqual(family_outliers(tiny), [])


def anonymous(rows, senses, rhs):
    """Rows as coefficient lists over shared columns, with names that carry no pattern (as in some anonymous MPS)."""
    n = max(len(r) for r in rows)
    A = sp.csr_matrix(np.array([r + [0] * (n - len(r)) for r in rows], float))
    lo = np.array([b if s in (">=", "=") else -od.INF for s, b in zip(senses, rhs)], float)
    hi = np.array([b if s in ("<=", "=") else od.INF for s, b in zip(senses, rhs)], float)
    names = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel", "india", "juliet")[:len(rows)]
    return od.ModelData(name="anon", minimize=True, obj=np.zeros(n), obj_offset=0.0, A=A, row_lo=lo, row_hi=hi,
                        col_lb=np.zeros(n), col_ub=np.full(n, 10.0), is_int=np.zeros(n, bool),
                        col_names=tuple(f"c{j}" for j in range(n)), row_names=names)


class TestNameFreeSenses(unittest.TestCase):
    # on anonymous models the flipped row was never compared with the caps written next to it
    CAPS = [[1, -1], [1, 1, 0, 1], [1], [0, 1, 1], [1, 0, 0, 1, 1], [0, 0, 1], [1, 1, 1, 1, 1]]

    def test_flip_in_a_run_of_consecutive_caps(self):
        md = anonymous(self.CAPS, ["<=", "<=", "<=", "<=", ">=", "<=", "<="], [0, 5, 3, 2, 10, 4, 9])
        flags = [f for f in family_outliers(md) if f["kind"] == "constraint sense"]
        self.assertEqual([f["name"] for f in flags], ["echo"])
        self.assertIn("consecutive", flags[0]["group"])

    def test_an_equality_among_inequalities_is_not_a_flip(self):
        md = anonymous(self.CAPS, ["<=", "<=", "<=", "<=", "=", "<=", "<="], [0, 5, 3, 2, 10, 4, 9])
        self.assertEqual([f for f in family_outliers(md) if f["kind"] == "constraint sense"], [])

    def test_named_models_keep_their_name_groups(self):
        md = replace(anonymous(self.CAPS, ["<=", "<=", "<=", "<=", ">=", "<=", "<="], [0, 5, 3, 2, 10, 4, 9]),
                     row_names=tuple(f"cap[{i}]" for i in range(7)))  # one family: too mixed in shape to judge
        self.assertNotIn("consecutive", " ".join(f["group"] for f in family_outliers(md)))


class TestTightestGroup(unittest.TestCase):
    def test_the_smallest_flagging_group_gives_the_reason(self):
        # ROW# holds all 10 rows (9 of 10 at 0); the one-term rows are a tighter group (4 of 5 at 0)
        rows = [[1]] * 5 + [[1, 1]] * 5
        md = anonymous(rows, ["<="] * 10, [0, 0, -5, 0, 0, 0, 0, 0, 0, 0])
        md = replace(md, row_names=tuple(f"ROW{i:05d}" for i in range(10)))
        [flag] = family_outliers(md)
        self.assertEqual(flag["name"], "ROW00002")
        self.assertIn("4 of 5", flag["reason"])

    def test_a_budget_its_terms_can_reach_is_not_out_of_scale(self):
        # a budget of 3900 next to exclusivity limits of 1 was flagged as 3900x the model's limits
        n = 40
        A = sp.vstack([sp.identity(n, format="csr"), sp.csr_matrix(np.full((1, n), 500.0))]).tocsr()
        md = od.ModelData(name="m", minimize=True, obj=np.zeros(n), obj_offset=0.0, A=A,
                          row_lo=np.full(n + 1, -od.INF), row_hi=np.array([1.0] * n + [3900.0]),
                          col_lb=np.zeros(n), col_ub=np.ones(n), is_int=np.ones(n, bool),
                          col_names=tuple(f"x[{i}]" for i in range(n)), row_names=tuple(f"excl[{i}]" for i in range(n)) + ("budget",))
        self.assertEqual(family_outliers(md), [])
        md = replace(md, row_hi=np.array([1.0] * n + [39000.0]))  # beyond what the terms can reach (20000)
        self.assertEqual([f["name"] for f in family_outliers(md)], ["budget"])


class TestPatternBreaks(unittest.TestCase):
    # models that solve with a wrong optimum; the error is in the objective, a coefficient or a missing row
    def test_objective_sign_flip_in_a_run_of_equal_costs(self):
        md = replace(model([1] * 10), obj=np.array([0.242] * 6 + [-0.242] + [0.3] * 3),
                     col_names=tuple(f"A{361 + j}" for j in range(10)))
        flags = [f for f in pattern_breaks(md) if f["kind"] == "objective coefficient"]
        self.assertEqual([f["name"] for f in flags], ["A367"])

    def test_power_of_ten_coefficient_among_identical_rows(self):
        rows = [[1, 1, -1, 0]] * 4 + [[10, 1, -1, 0]]
        md = anonymous(rows, ["="] * 5, [0] * 5)
        [flag] = [f for f in pattern_breaks(md) if f["kind"] == "coefficient"]
        self.assertEqual((flag["name"], flag["column"], flag["value"], flag["typical"]), ("echo", "c0", 10.0, 1.0))

    def test_a_sign_difference_in_a_coefficient_is_not_flagged(self):
        rows = [[1, 1, -1]] * 4 + [[1, 1, 1]]  # balance rows mix +1 and -1 by design
        self.assertEqual(pattern_breaks(anonymous(rows, ["="] * 5, [0] * 5)), [])

    def test_gap_in_a_numbered_series_and_an_indexed_family(self):
        names = [f"ROW{i:05d}" for i in range(1, 16) if i != 7]
        md = replace(model([1] * len(names)), row_names=tuple(names))
        self.assertEqual([f["name"] for f in pattern_breaks(md)], ["ROW00007"])
        grid = [f"x[{a},{b}]" for a in range(3) for b in range(3) if (a, b) != (1, 2)]
        md = replace(model([1] * len(grid)), row_names=tuple(grid))
        self.assertEqual([f["name"] for f in pattern_breaks(md)], ["x[1,2]"])

    def test_infeasible_keeps_only_coefficients_in_the_iis(self):
        # on an infeasible model a coefficient flag outside the conflict led the first fix astray
        rows = [[1, 1, -1, 0]] * 4 + [[10, 1, -1, 0]]
        md = anonymous(rows, ["="] * 5, [0] * 5)
        self.assertEqual(pattern_breaks(md, infeasible=True), [])
        self.assertEqual([f["name"] for f in pattern_breaks(md, {"echo"}, infeasible=True)], ["echo"])
        md = replace(model([1] * 10), obj=np.array([0.242] * 6 + [-0.242] + [0.3] * 3))
        self.assertEqual(pattern_breaks(md, infeasible=True), [])  # the objective cannot make a model infeasible

    def test_many_breaks_of_one_kind_are_how_the_model_is_written(self):
        rows = [[1, 1, 1, -1]] * 3 + [[10, 1, 1, -1], [1, 1, 1, -10], [1, 1, 1, -0.1], [0.1, 1, 1, -1]]
        self.assertEqual([f for f in pattern_breaks(anonymous(rows, ["="] * 7, [0] * 7)) if f["kind"] == "coefficient"], [])


if __name__ == "__main__":
    unittest.main()
