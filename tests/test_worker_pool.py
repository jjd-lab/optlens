"""Reused solver processes keep the hard time limit. Run from the repo root: python -m unittest tests.test_worker_pool"""
import os
import time
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

import optlens as od
from optlens import backends


def _pid_with(md):
    return os.getpid()


class TestWorkerPool(unittest.TestCase):
    def test_timeout_kills_and_next_call_works(self):
        t = time.time()
        with self.assertRaises(backends._TimedOut):
            backends._with_deadline(time.sleep, (30,), 1.0)
        self.assertLess(time.time() - t, 10)
        self.assertEqual(backends._with_deadline(abs, (-3,), 30), 3)

    def test_processes_are_reused(self):
        first = backends._with_deadline(os.getpid, (), 30)
        self.assertEqual(backends._with_deadline(os.getpid, (), 30), first)

    def test_errors_come_back(self):
        with self.assertRaises(ValueError):
            backends._with_deadline(int, ("x",), 30)
        self.assertEqual(backends._with_deadline(abs, (-1,), 30), 1)

    def test_a_process_that_solved_a_large_model_is_not_kept(self):
        # it would hold that model's memory while idle (1.5 GB at 837k rows)
        md = od.ModelData(name="m", minimize=True, obj=np.ones(2), obj_offset=0.0, A=sp.csr_matrix(np.ones((1, 2))),
                          row_lo=np.ones(1), row_hi=np.full(1, od.INF), col_lb=np.zeros(2), col_ub=np.ones(2),
                          is_int=np.zeros(2, bool), col_names=("a", "b"), row_names=("r",))
        with mock.patch.object(backends, "_KEEP_MAX_NNZ", 1):
            first = backends._with_deadline(_pid_with, (md,), 30)
            self.assertNotEqual(backends._with_deadline(_pid_with, (md,), 30), first)
        first = backends._with_deadline(_pid_with, (md,), 30)
        self.assertEqual(backends._with_deadline(_pid_with, (md,), 30), first)

    def test_parallel_solves_fit_the_memory_budget(self):
        small = sp.csr_matrix(np.ones((1, 2)))
        md = od.ModelData(name="m", minimize=True, obj=np.ones(2), obj_offset=0.0, A=small, row_lo=np.ones(1),
                          row_hi=np.full(1, od.INF), col_lb=np.zeros(2), col_ub=np.ones(2), is_int=np.zeros(2, bool),
                          col_names=("a", "b"), row_names=("r",))
        self.assertEqual(backends.parallel_solves(md), 4)
        with mock.patch.object(backends, "SOLVE_BYTES_PER_NZ", backends.SOLVE_MEMORY / 2 / 2):
            self.assertEqual(backends.parallel_solves(md), 2)  # 2 nonzeros: half the budget each
        with mock.patch.object(backends, "SOLVE_BYTES_PER_NZ", backends.SOLVE_MEMORY):
            self.assertEqual(backends.parallel_solves(md), 1)  # never fewer than one


if __name__ == "__main__":
    unittest.main()
