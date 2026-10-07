"""Row and column lookups by name. Run from the repo root: python -m unittest tests.test_model_data"""
import time
import unittest

import numpy as np
import scipy.sparse as sp

import optlens as od


def model(rows: int) -> od.ModelData:
    # one column, `rows` rows x <= i: big enough that a linear scan per name shows
    return od.ModelData(
        name="m", minimize=True, obj=np.array([1.0]), obj_offset=0.0,
        A=sp.csr_matrix(np.ones((rows, 1))), row_lo=np.full(rows, -od.INF), row_hi=np.arange(rows, dtype=float),
        col_lb=np.zeros(1), col_ub=np.full(1, od.INF), is_int=np.array([False]),
        col_names=("x",), row_names=tuple(f"r{i}" for i in range(rows)),
    )


class TestLookups(unittest.TestCase):
    def test_row_index_finds_the_first_row_of_a_name_and_rejects_unknown_ones(self):
        md = od.ModelData(
            name="m", minimize=True, obj=np.array([1.0]), obj_offset=0.0, A=sp.csr_matrix(np.ones((3, 1))),
            row_lo=np.full(3, -od.INF), row_hi=np.array([1.0, 2.0, 3.0]), col_lb=np.zeros(1),
            col_ub=np.full(1, od.INF), is_int=np.array([False]), col_names=("x",), row_names=("a", "b", "a"),
        )
        self.assertEqual(md.row_index("a"), 0)
        self.assertEqual(md.row_index("b"), 1)
        with self.assertRaises(ValueError):
            md.row_index("c")

    def test_an_edited_model_looks_up_its_own_rows(self):
        md = model(5).drop_rows(["r1", "r3"])
        self.assertEqual([md.row_index(n) for n in ("r0", "r2", "r4")], [0, 1, 2])
        with self.assertRaises(ValueError):
            md.row_index("r1")

    def test_dropping_a_large_family_is_fast(self):
        # a 262,800-row family took ~560 s to drop from 279k rows when each name was found by tuple.index
        md = model(100_000)
        t = time.perf_counter()
        rest = md.drop_rows([f"r{i}" for i in range(0, 100_000, 2)])
        self.assertLess(time.perf_counter() - t, 5.0)
        self.assertEqual(rest.num_rows, 50_000)
        self.assertEqual(rest.row_names[:2], ("r1", "r3"))


if __name__ == "__main__":
    unittest.main()
