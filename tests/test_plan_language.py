"""Engine output in a planner's terms: schedules as before -> after patterns, linking rows marked in a relaxation.
Run from the repo root: python -m unittest tests.test_plan_language"""
import unittest

import numpy as np
import scipy.sparse as sp

import optlens as od
from optlens.session import Session, Version


def schedule() -> od.ModelData:
    # open[a_0..3], open[b_0..3] in {0,1}; maximize openings weighted by day, at most 2 open per product
    names = tuple(f"open({p}_{d})" for p in "ab" for d in range(4))
    A = sp.csr_matrix(np.array([[1, 1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 0, 1, 1, 1, 1]], float))
    return od.ModelData(name="m", minimize=False, obj=np.array([1, 2, 3, 4, 1, 2, 3, 4], float), obj_offset=0.0, A=A,
                        row_lo=np.full(2, -od.INF), row_hi=np.full(2, 2.0), col_lb=np.zeros(8), col_ub=np.ones(8),
                        is_int=np.ones(8, bool), col_names=names, row_names=("cap(a)", "cap(b)"))


def linked() -> od.ModelData:
    # make - ship = 0 links the two; ship >= 10 is the business demand; make <= 5 the capacity: infeasible
    A = sp.csr_matrix(np.array([[1.0, -1.0], [0.0, 1.0], [1.0, 0.0]]))
    return od.ModelData(name="m", minimize=True, obj=np.ones(2), obj_offset=0.0, A=A,
                        row_lo=np.array([0.0, 10.0, -od.INF]), row_hi=np.array([0.0, od.INF, 5.0]),
                        col_lb=np.zeros(2), col_ub=np.full(2, od.INF), is_int=np.zeros(2, bool),
                        col_names=("make", "ship"), row_names=("link", "demand", "capacity"))


def two_ways() -> od.ModelData:
    # x + y >= 10 is needed, but cap_x: x <= 4 and cap_y: y <= 4: short by 2. Raising either cap by 2 is the same
    # minimal total change; x earns 3 and y earns 1 (maximize), so the best fix raises cap_x.
    A = sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 0.0], [0.0, 1.0]]))
    return od.ModelData(name="m", minimize=False, obj=np.array([3.0, 1.0]), obj_offset=0.0, A=A,
                        row_lo=np.array([10.0, -od.INF, -od.INF]), row_hi=np.array([od.INF, 4.0, 4.0]),
                        col_lb=np.zeros(2), col_ub=np.full(2, od.INF), is_int=np.zeros(2, bool),
                        col_names=("x", "y"), row_names=("need", "cap_x", "cap_y"))


class PlanLanguage(unittest.TestCase):
    def test_a_schedule_change_reads_as_a_pattern(self):
        s = Session({"v0": Version(schedule(), None, "original")})
        s.modify_and_resolve("v0", "a on day 0", [{"action": "set_bounds", "name": "open(a_0)", "lower": 1}])
        out = s.compare_versions("v0", "v1")
        self.assertIn("open over its last index (0, 1, 2, 3), before -> after (O = 1, . = 0, - = none):", out)
        self.assertIn("  a: ..OO -> O..O", out)
        self.assertNotIn("largest variable changes", out)  # every changed family is shown as a pattern

    def test_a_sparse_family_keeps_each_value_under_its_position(self):
        # b has no day 0, and the file lists its columns out of order: values must stay under their own day
        md = schedule()
        keep = [0, 1, 2, 3, 7, 6, 5]
        md = od.ModelData(name="m", minimize=False, obj=md.obj[keep], obj_offset=0.0, A=md.A[:, keep].tocsr(),
                          row_lo=md.row_lo, row_hi=md.row_hi, col_lb=md.col_lb[keep], col_ub=md.col_ub[keep],
                          is_int=md.is_int[keep], col_names=tuple(md.col_names[j] for j in keep), row_names=md.row_names)
        s = Session({"v0": Version(md, None, "original")})
        s.modify_and_resolve("v0", "b on day 1", [{"action": "set_bounds", "name": "open(b_1)", "lower": 1}])
        out = s.compare_versions("v0", "v1")
        self.assertIn("  b: -.OO -> -O.O", out)

    def test_underscores_that_are_not_indices_stay_one_name(self):
        A = sp.csr_matrix(np.ones((1, 2)))
        md = od.ModelData(name="m", minimize=False, obj=np.array([1.0, 2.0]), obj_offset=0.0, A=A,
                          row_lo=np.array([-od.INF]), row_hi=np.array([10.0]), col_lb=np.zeros(2), col_ub=np.full(2, 10.0),
                          is_int=np.zeros(2, bool), col_names=("spend[north_east]", "spend[north_west]"), row_names=("budget",))
        s = Session({"v0": Version(md, None, "original")})
        s.modify_and_resolve("v0", "east first", [{"action": "set_bounds", "name": "spend[north_east]", "lower": 4}])
        out = s.compare_versions("v0", "v1")
        self.assertIn("spend over its last index (north_east, north_west), before -> after:", out)
        self.assertIn("  spend: 0 10 -> 4 6", out)

    def test_relaxation_marks_linking_rows_and_lists_business_first(self):
        s = Session({"v0": Version(linked(), None, "original")})
        out = s.feasibility_relaxation()
        lines = [ln for ln in out.splitlines() if ln.startswith("  ")]
        self.assertTrue(all("linking" not in ln for ln in lines[:1]), out)
        self.assertTrue(all(("link:" in ln) == ("[linking row, not a business lever]" in ln) for ln in lines), out)

    def test_relaxation_picks_the_best_plan_among_minimal_fixes(self):
        r = od.feas_relax(two_ways(), od.BACKENDS["highs"], only={"cap_x", "cap_y"}, min_objective=True)
        self.assertAlmostEqual(r.total_violation, 2.0)
        self.assertEqual([c["name"] for c in r.relaxations], ["cap_x"])
        self.assertAlmostEqual(r.objective, 3 * 6 + 1 * 4)

    def test_scaling_back_respects_the_penalty(self):
        r = od.feas_relax(two_ways(), od.BACKENDS["highs"], only={"cap_x", "cap_y"}, penalty=0.5, min_objective=True)
        self.assertAlmostEqual(sum(c["value"] for c in r.relaxations), 2.0)  # the change itself, not halved

    def test_fix_menu_reports_the_best_plan_for_each_lever(self):
        out = Session({"v0": Version(two_ways(), None, "original")}).fix_menu(families=["cap_x", "cap_y"])
        self.assertIn("cap_x: sufficient; total change 2 over 1 bounds; best plan with it: objective 22", out)
        self.assertIn("cap_y: sufficient; total change 2 over 1 bounds; best plan with it: objective 18", out)

    def test_an_unrestricted_relaxation_names_the_other_levers_in_the_conflict(self):
        s = Session({"v0": Version(two_ways(), None, "original")})
        self.assertIn("this is the smallest fix in total; other limits may each fix it alone", s.feasibility_relaxation())
        s.compute_iis()
        out = s.feasibility_relaxation()
        self.assertIn("best plan at this minimal change: objective", out)
        self.assertRegex(out, r"also in the conflict, unchanged by this fix: (cap_y|need)")

    def test_linking_families_are_rows_whose_limits_are_all_zero(self):
        self.assertEqual(od.linking_families(linked()), {"link"})


if __name__ == "__main__":
    unittest.main()
