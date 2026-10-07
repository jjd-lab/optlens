"""fix_menu on a model with two independent conflicts. Run from the repo root: python -m unittest tests.test_fix_menu"""
import unittest

import numpy as np
import scipy.sparse as sp

import optlens as od
from optlens.diagnose import base_name


def toy(two_conflicts: bool) -> od.ModelData:
    # rows: min_x: x >= 5, max_x: x <= 3, min_y: y >= 6 (or >= 2), max_y: y <= 4
    return od.ModelData(
        name="toy", minimize=True, obj=np.zeros(2), obj_offset=0.0,
        A=sp.csr_matrix(np.array([[1.0, 0], [1, 0], [0, 1], [0, 1]])),
        row_lo=np.array([5.0, -od.INF, 6.0 if two_conflicts else 2.0, -od.INF]),
        row_hi=np.array([od.INF, 3.0, od.INF, 4.0]),
        col_lb=np.zeros(2), col_ub=np.full(2, 100.0), is_int=np.zeros(2, bool),
        col_names=("x", "y"), row_names=("min_x", "max_x", "min_y", "max_y"),
    )


class TestFixMenu(unittest.TestCase):
    def test_single_conflict_each_side_is_a_lever(self):
        menu = od.fix_menu(toy(False), od.BACKENDS["highs"], families=["min_x", "max_x"])
        by = {e["family"]: e for e in menu}
        self.assertTrue(by["min_x"]["sufficient"] and by["max_x"]["sufficient"])
        self.assertAlmostEqual(by["min_x"]["total_change"], 2.0, places=6)
        self.assertAlmostEqual(menu[-1]["total_change"], 2.0, places=6)

    def test_no_family_alone_fixes_two_conflicts(self):
        menu = od.fix_menu(toy(True), od.BACKENDS["highs"], families=["min_x", "max_x", "min_y", "max_y"])
        self.assertFalse(any(e["sufficient"] for e in menu[:-1]))
        self.assertTrue(menu[-1]["sufficient"])
        self.assertAlmostEqual(menu[-1]["total_change"], 4.0, places=6)

    def test_default_families_come_from_iis(self):
        fams = [e["family"] for e in od.fix_menu(toy(False), od.BACKENDS["highs"])[:-1]]
        self.assertEqual(sorted(fams), ["max_x", "min_x"])

    def test_suggested_values_are_the_exact_minimum(self):
        by = {e["family"]: e for e in od.fix_menu(toy(False), od.BACKENDS["highs"], families=["min_x", "max_x"])}
        lo, hi = by["min_x"]["changes"][0], by["max_x"]["changes"][0]
        self.assertEqual((lo["side"], lo["current"]), ("lower", 5.0))
        self.assertAlmostEqual(lo["new"], 3.0, places=9)   # x >= 5 relaxed down to the x <= 3 cap
        self.assertEqual((hi["side"], hi["current"]), ("upper", 3.0))
        self.assertAlmostEqual(hi["new"], 5.0, places=9)


class SlowBackend(od.HiGHSBackend):
    """Every solve runs to its time limit without an answer, like a remote Gurobi on a hard relaxation."""

    def solve(self, md, time_limit=60.0, start=None):
        import time

        time.sleep(time_limit)
        return od.SolveResult("TIME_LIMIT")


class TestFixMenuDeadline(unittest.TestCase):
    def test_queued_relaxations_share_the_menus_time_limit(self):
        # E66: with fewer solver slots than families, each queued relaxation got the full limit (356 s against 60)
        import time
        from unittest import mock

        md = toy(two_conflicts=True)
        families = sorted({base_name(r) for r in md.row_names})
        with mock.patch("optlens.diagnose.parallel_solves", return_value=1):
            t0 = time.monotonic()
            menu = od.fix_menu(md, SlowBackend(), families, time_limit=1.5)
        self.assertLess(time.monotonic() - t0, 3.0)  # was (families + 1) x 1.5 s
        self.assertTrue(all(e["status"] == "TIME_LIMIT" for e in menu))
