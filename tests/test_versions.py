"""Edited versions keep their changes, not a model copy; only the most recently used few stay built (dozens of
837k-row copies ran the worker out of memory). Run from the repo root: python -m unittest tests.test_versions"""
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

import optlens as od
from optlens import session as S
from optlens.session import Session, Version


def model():
    # max a + b + c with a + b <= 4 (cap), b + c <= 5 (fence), each in [0, 3]
    return od.ModelData(
        name="m", minimize=False, obj=np.ones(3), obj_offset=0.0,
        A=sp.csr_matrix(np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])), row_lo=np.full(2, -od.INF),
        row_hi=np.array([4.0, 5.0]), col_lb=np.zeros(3), col_ub=np.full(3, 3.0), is_int=np.zeros(3, bool),
        col_names=("a", "b", "c"), row_names=("cap", "fence"))


class TestVersionsAsChanges(unittest.TestCase):
    def setUp(self):
        self.s = Session({"v0": Version(model(), None, "original")})

    def test_only_recent_models_stay_built_and_older_ones_rebuild(self):
        with mock.patch.object(S, "MODELS_KEPT", 1):
            self.s.modify_and_resolve("v0", "cap 5", [{"action": "set_rhs", "name": "cap", "upper": 5}])
            self.s.modify_and_resolve("v1", "no fence", [{"action": "drop_constraint", "name": "fence"}])
            self.assertEqual(list(self.s._models), ["v2"])
            self.assertIsNone(self.s.versions["v1"]._md)
            v1 = self.s.get("v1").md  # rebuilt from v0 and its change
            self.assertEqual(v1.row_hi[v1.row_index("cap")], 5.0)
            self.assertEqual(list(self.s._models), ["v1"])
            v2 = self.s.get("v2").md  # rebuilt through v1
            self.assertEqual(v2.row_names, ("cap",))
            self.assertEqual(v2.row_hi[0], 5.0)
        self.assertEqual(self.s.solved("v2").obj, 8.0)  # results are kept with the version, not rebuilt
        self.assertIn("v2", self.s.compare_versions("v1", "v2"))

    def test_try_options_versions_rebuild_when_needed(self):
        with mock.patch.object(S, "MODELS_KEPT", 1):
            out = self.s.try_options("v0", [{"label": "cap 5", "changes": [{"action": "set_rhs", "name": "cap",
                                                                             "upper": 5}]},
                                            {"label": "no fence", "changes": [{"action": "drop_constraint",
                                                                                "name": "fence"}]},
                                            {"label": "same", "changes": [{"action": "set_rhs", "name": "cap",
                                                                           "upper": 4}]}])
        self.assertIn("v1 cap 5: OPTIMAL, objective 8", out)
        self.assertIn("v2 no fence: OPTIMAL, objective 7", out)
        self.assertIn("same: no change from v0", out)
        self.assertEqual(self.s.get("v1").changes, ({"action": "set_rhs", "name": "cap", "upper": 5},))
        self.assertEqual(len(self.s._models), 1)

    def test_roots_keep_their_model(self):
        self.s.add_model(model(), "other")
        self.assertIsNotNone(self.s.versions["other"]._md)
        self.assertIsNotNone(self.s.versions["v0"]._md)


class TestChainsAndBranches(unittest.TestCase):
    """A planner keeps changes one by one, then goes back to the original and tries something else."""

    def setUp(self):
        self.s = Session({"v0": Version(model(), None, "original")})

    def step(self, base, changes, label="step"):
        return self.s._modify(base, label, changes)[0]

    def test_a_long_chain_rebuilds_in_one_pass_and_keeps_only_what_is_used(self):
        with mock.patch.object(S, "MODELS_KEPT", 2):
            v = "v0"
            for k in range(8):  # cap 4 -> 5 -> ... -> 12, one step at a time
                v = self.step(v, [{"action": "set_rhs", "name": "cap", "upper": 5 + k}])
            branch = self.step("v0", [{"action": "drop_constraint", "name": "fence"}], "back to the original")
            self.s._models.clear()
            with mock.patch.object(Session, "_apply", wraps=Session._apply) as apply:
                tip = self.s.get(v).md
            self.assertEqual(apply.call_count, 1)  # one pass from v0, not one per step
            self.assertEqual(len(apply.call_args.args[1]), 8)
            self.assertEqual(tip.row_hi[tip.row_index("cap")], 12.0)
            self.assertEqual(list(self.s._models), [v])  # the steps in between are not kept
            self.s.get("v4").md  # a middle step: rebuilt from v0 (v8 is further down the chain, not an ancestor)
            self.assertEqual(list(self.s._models), [v, "v4"])
            self.assertEqual(self.s.get("v6").md.row_hi[0], 10.0)  # rebuilt from the kept v4, two steps on
            self.assertEqual(self.s.get(branch).md.row_names, ("cap",))
            self.assertEqual(self.s.get(branch).md.row_hi[0], 4.0)  # the branch has none of the chain's changes
        self.assertEqual(self.s.lineage(v), ["v0"] + [f"v{k}" for k in range(1, 9)])
        self.assertEqual(self.s.lineage(branch), ["v0", branch])

    def test_version_history_says_what_a_plan_contains_and_what_it_leaves_out(self):
        v1 = self.step("v0", [{"action": "set_rhs", "name": "cap", "upper": 5}], "cap 5")
        v2 = self.step(v1, [{"action": "drop_constraint", "name": "fence"}], "no fence")
        v3 = self.step("v0", [{"action": "set_bounds", "name": "c", "upper": 1}], "c at most 1")
        out = self.s.version_history(v2)
        self.assertIn(f"{v2}: 2 step(s) and 2 change(s) from v0", out)
        self.assertIn(f"{v1} from v0: cap 5 [OPTIMAL, objective 8]", out)
        self.assertIn("  - limit of cap: upper 5", out)
        self.assertIn("  - dropped fence", out)
        self.assertIn(f"not in {v2}: {v3} (other branches)", out)

    def test_consecutive_drops_are_one_copy_and_a_repeated_drop_still_fails(self):
        md = model()
        with mock.patch.object(od.ModelData, "drop_rows", autospec=True, wraps=od.ModelData.drop_rows) as drop:
            out = Session._apply(md, [{"action": "drop_constraint", "name": "cap"},
                                      {"action": "drop_constraint", "name": "fence"}])
        self.assertEqual(drop.call_count, 1)
        self.assertEqual(out.row_names, ())
        with self.assertRaises(ValueError):
            Session._apply(md, [{"action": "drop_constraint", "name": "cap"},
                                {"action": "drop_constraint", "name": "cap"}])


class TestBatchedChanges(unittest.TestCase):
    """Runs of limit or objective changes are applied in one copy, with the same result and errors as one by one."""

    def test_batch_equals_one_by_one_and_last_wins(self):
        md = model()
        changes = [{"action": "set_rhs", "name": "cap", "upper": 5}, {"action": "set_rhs", "name": "fence", "upper": 6},
                   {"action": "set_rhs", "name": "cap", "upper": 7}, {"action": "set_objective_coef", "name": "a", "value": 3},
                   {"action": "set_objective_coef", "name": "b", "value": 0}, {"action": "drop_constraint", "name": "fence"},
                   {"action": "set_rhs", "name": "cap", "lower": 1}]
        out = Session._apply(md, changes)
        self.assertEqual((out.row_names, list(out.row_hi), list(out.row_lo), list(out.obj)), (("cap",), [7.0], [1.0], [3.0, 0.0, 1.0]))
        self.assertEqual(S._apply_batch(md, changes[:3]).row_hi.tolist(), [7.0, 6.0])

    def test_errors_in_a_batch_are_still_reported(self):
        md = model()
        with self.assertRaises(ValueError) as e:
            Session._apply(md, [{"action": "set_rhs", "name": "cap", "upper": 5}, {"action": "set_rhs", "name": "nope", "upper": 1},
                                {"action": "set_objective_coef", "name": "a", "value": "x"},
                                {"action": "set_rhs", "name": "fence", "lower": 9, "upper": 2}])
        msg = str(e.exception)
        self.assertIn("change 2 (set_rhs nope)", msg)
        self.assertIn("change 3 (set_objective_coef a)", msg)
        self.assertIn("lower 9 is above upper 2", msg)


if __name__ == "__main__":
    unittest.main()


class TestExportAndRestore(unittest.TestCase):
    """A resumed conversation rebuilds its versions in a new process from what export_versions wrote."""

    def test_versions_come_back_under_their_ids_and_solve_again(self):
        import json
        import tempfile
        from pathlib import Path
        s = Session({"v0": Version(model(), None, "original")})
        v1 = s._modify("v0", "cap 5", [{"action": "set_rhs", "name": "cap", "upper": np.float64(5)}])[0]
        v2 = s._modify(v1, "no fence", [{"action": "drop_constraint", "name": "fence"}])[0]
        v3 = s._modify("v0", "c at most 1", [{"action": "set_bounds", "name": "c", "upper": 1}])[0]
        with tempfile.TemporaryDirectory() as d:
            lp = Path(d) / "other.lp"
            lp.write_text("Maximize\n obj: x\nSubject To\n c1: x <= 2\nEnd\n")
            s.add_model(str(lp), "other")
            s.carried = {"other": v3}
            state = json.loads(json.dumps(s.export_versions()))  # stored as JSON
            fresh = Session({"v0": Version(model(), None, "original")})
            out = fresh.restore_versions(state)
            self.assertIn("restored 4 versions (v1, v2, v3, other)", out)
            self.assertEqual(list(fresh.versions), ["v0", v1, v2, v3, "other"])
            self.assertIsNone(fresh.versions[v2].result)  # not solved until used
            self.assertEqual(fresh.solved(v2).obj, s.solved(v2).obj)
            self.assertEqual(fresh.solved(v3).obj, s.solved(v3).obj)
            self.assertEqual(fresh.solved("other").obj, 2.0)
            self.assertEqual(fresh.lineage(v2), ["v0", v1, v2])
            self.assertEqual(fresh.carried, {"other": v3})
            self.assertEqual(fresh._new_version(None, None, "next"), "v5")  # new ids continue after the restored ones
        state["versions"].append({"id": "v9", "parent": "v8", "changes": [], "description": ""})
        self.assertIn("not restored: other (its file is gone), v9 (its parent v8 is missing)", Session({"v0": Version(model(), None, "o")})
                      .restore_versions(state))
