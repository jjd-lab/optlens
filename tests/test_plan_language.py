"""Engine output in a planner's terms: schedules as before -> after patterns, linking rows marked in a relaxation.
Run from the repo root: python -m unittest tests.test_plan_language"""
import unittest

import numpy as np
import scipy.sparse as sp

import optlens as od
from optlens.session import TOOLS, Session, Version


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


def furniture() -> od.ModelData:
    # two plants (north 400 h, south 150 h) and contracts for 120 chairs (1 h), 60 tables (3 h) and 70 desks (5 h,
    # north only): 650 h needed, 550 available. The one minimal fix within the contracts is 20 fewer desks.
    products, plants = [("chairs", 1.0, 20.0, 120.0), ("tables", 3.0, 70.0, 60.0), ("desks", 5.0, 130.0, 70.0)], ["north", "south"]
    cols = [(p, k) for p, *_ in products for k in plants]
    cap = [[h if k == kk else 0.0 for (p, kk) in cols for (pp, h, *_) in products if pp == p] for k in plants]
    con = [[1.0 if pp == p else 0.0 for (pp, _) in cols] for p, *_ in products]
    A = sp.csr_matrix(np.array(cap + con))
    return od.ModelData(name="m", minimize=False, obj=np.array([v for _, _, v, _ in products for _ in plants]), obj_offset=0.0,
                        A=A, row_lo=np.array([-od.INF, -od.INF] + [c for *_, c in products]),
                        row_hi=np.array([400.0, 150.0, od.INF, od.INF, od.INF]), col_lb=np.zeros(6),
                        col_ub=np.array([od.INF] * 5 + [0.0]), is_int=np.zeros(6, bool),
                        col_names=tuple(f"make[{p},{k}]" for p, k in cols),
                        row_names=("capacity[north]", "capacity[south]", "contract[chairs]", "contract[tables]", "contract[desks]"))


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

    def test_the_infeasibility_proof_reads_as_family_totals(self):
        # the contracts need 120 x 1 + 60 x 3 + 70 x 5 = 650 hours, the plants allow 400 + 150 = 550
        c = od.certificate(furniture())
        self.assertEqual({k: n for k, (n, _) in c.required.items()}, {"contract": 3})
        self.assertEqual({k: n for k, (n, _) in c.limits.items()}, {"capacity": 2})
        self.assertAlmostEqual(c.need, 650)
        self.assertAlmostEqual(c.have, 550)
        self.assertAlmostEqual(c.forced, 0)

    def test_the_proof_text_weights_each_row(self):
        out = Session({"v0": Version(furniture(), None, "week")}).infeasibility_proof()
        self.assertIn("require at least 650:", out)
        self.assertIn("5 x contract[desks] >= 70", out)
        self.assertIn("capacity (2 rows): 550", out)
        self.assertIn("short by 100", out)

    def test_the_proof_is_a_tool_of_its_own(self):
        # E91-E92: the proof inside compute_iis cost answers their business chain, so it is offered on request only
        s = Session({"v0": Version(furniture(), None, "week")})
        self.assertNotIn("require at least", s.compute_iis())
        out, err = s.call("infeasibility_proof", {})
        self.assertFalse(err)
        self.assertIn("short by 100", out)
        self.assertIn("infeasibility_proof", [t["name"] for t in TOOLS])

    def test_a_crossed_bound_is_its_own_proof(self):
        md = furniture()
        md = md.set_col_bounds("make[chairs,north]", lb=5.0, ub=2.0)
        out = Session({"v0": Version(md, None, "week")}).infeasibility_proof()
        self.assertEqual(out, "no plan exists: make[chairs,north] must be >= 5 and <= 2")

    def test_the_optimum_has_a_receipt(self):
        # with 250 south hours the contracts' 650 hours fit exactly: the prices of the limits add up to the profit
        md = furniture().set_row_bounds("capacity[south]", hi=250.0)
        c = od.optimality_certificate(md)
        self.assertAlmostEqual(c.bound, 15700)
        self.assertTrue(c.exact)
        self.assertFalse(c.relaxation)
        total = sum(t for _, t in c.rows.values()) + sum(t for _, t in c.bounds.values()) + c.offset
        self.assertAlmostEqual(total, 15700)

    def test_the_optimality_proof_is_a_tool_on_request(self):
        md = furniture().set_row_bounds("capacity[south]", hi=250.0)
        out, err = Session({"v0": Version(md, None, "week")}).call("optimality_proof", {})
        self.assertFalse(err)
        self.assertIn("no plan earns more than 15,700, and this plan reaches it", out)
        self.assertIn("capacity (2 rows): 100% (16,900)", out)
        self.assertIn("contract (2 rows): counts against the bound (-1,200)", out)
        self.assertIn("optimality_proof", [t["name"] for t in TOOLS])
        out, _ = Session({"v0": Version(furniture(), None, "week")}).call("optimality_proof", {})
        self.assertIn("not solved to optimality", out)  # 150 south hours: infeasible

    def test_a_milps_receipt_comes_from_its_relaxation(self):
        def one(limit):  # max x, x whole in [0, 10], 2x <= limit
            return od.ModelData(name="m", minimize=False, obj=np.ones(1), obj_offset=0.0,
                                A=sp.csr_matrix(np.array([[2.0]])), row_lo=np.array([-od.INF]),
                                row_hi=np.array([limit]), col_lb=np.zeros(1), col_ub=np.full(1, 10.0),
                                is_int=np.ones(1, bool), col_names=("x",), row_names=("cap",))
        c = od.optimality_certificate(one(4.0))  # the relaxation's x = 2 is whole: the sum proves the optimum
        self.assertTrue(c.relaxation)
        self.assertTrue(c.exact)
        self.assertAlmostEqual(c.bound, 2)
        c = od.optimality_certificate(one(3.0))  # relaxation 1.5, best whole plan 1: a bound with a gap
        self.assertFalse(c.exact)
        self.assertAlmostEqual(c.bound, 1.5)
        self.assertAlmostEqual(c.plan, 1)
        self.assertAlmostEqual(c.gap, 0.5)

    def test_no_proof_when_only_integrality_conflicts(self):
        # 2x = 1 with x whole: the relaxation solves (x = 0.5), so no weighted sum of rows proves it infeasible
        md = od.ModelData(name="m", minimize=True, obj=np.ones(1), obj_offset=0.0, A=sp.csr_matrix(np.array([[2.0]])),
                          row_lo=np.array([1.0]), row_hi=np.array([1.0]), col_lb=np.zeros(1), col_ub=np.full(1, 10.0),
                          is_int=np.ones(1, bool), col_names=("x",), row_names=("half",))
        self.assertIsNone(od.certificate(md))

    def test_the_best_plan_phase_leaves_no_solver_noise_in_the_fix(self):
        # the best-plan solve returned 19.9999999685 on contract[desks] plus 5e-8 on contract[tables], which the
        # scale-back then spread over both: the fix must be the exact minimum on the one row that moves
        r = od.feas_relax(furniture(), od.BACKENDS["highs"], only={"contract"}, min_objective=True)
        self.assertEqual([(c["name"], c["value"]) for c in r.relaxations], [("contract[desks]", 20.0)])
        self.assertEqual(r.total_violation, 20.0)

    def test_scaling_back_respects_the_penalty(self):
        r = od.feas_relax(two_ways(), od.BACKENDS["highs"], only={"cap_x", "cap_y"}, penalty=0.5, min_objective=True)
        self.assertAlmostEqual(sum(c["value"] for c in r.relaxations), 2.0)  # the change itself, not halved

    def test_a_second_phase_without_a_plan_keeps_the_first(self):
        # E80: HiGHS ended the best-plan phase OPTIMAL with no plan, and feas_relax read its objective (None)
        highs = od.BACKENDS["highs"]

        class NoPlanSecond:
            calls = 0

            def solve(self, md, time_limit):
                self.calls += 1
                return highs.solve(md, time_limit) if self.calls == 1 else od.SolveResult("OPTIMAL")

        r = od.feas_relax(two_ways(), NoPlanSecond(), only={"cap_x", "cap_y"}, min_objective=True)
        self.assertAlmostEqual(r.total_violation, 2.0)
        self.assertIsNone(r.objective)

    def test_highs_optimal_without_a_plan_is_no_verdict(self):
        import highspy

        class Info:
            primal_solution_status = 1  # infeasible

        class Fake:
            def run(self):
                pass

            def getModelStatus(self):
                return highspy.HighsModelStatus.kOptimal

            def getInfo(self):
                return Info()

        bk = od.HiGHSBackend()
        bk._highs = lambda md, time_limit: Fake()
        res = bk._solve(two_ways(), 10.0)
        self.assertEqual(res.status, "OTHER")
        self.assertFalse(res.feasible)

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

    def test_a_relaxation_is_one_minimal_fix_not_the_only_one(self):
        # E66: Gurobi spread 4,031 over the floor and three demand rows, where the floor alone, by the same total, was
        # enough; "all required together" read as if every change were needed
        out = Session({"v0": Version(two_ways(), None, "original")}).feasibility_relaxation()
        self.assertNotIn("required together", out)
        self.assertIn("a minimal fix; other minimal fixes of the same size may exist (fix_menu lists single-family "
                      "fixes)", out)

    def test_linking_families_are_rows_whose_limits_are_all_zero(self):
        self.assertEqual(od.linking_families(linked()), {"link"})

    def test_a_zero_limit_ban_is_a_business_rule_not_a_link(self):
        # E83: the hotel's rate fence (last-minute discounted sales <= 0) was labelled a linking row
        md = linked().add_row("fence", {"ship": 1.0}, hi=0.0)
        self.assertEqual(od.linking_families(md), {"link"})


if __name__ == "__main__":
    unittest.main()
