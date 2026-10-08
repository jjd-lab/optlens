"""Solver workers start where forkserver cannot: Windows, and a sandbox that forbids Unix sockets.
Run from the repo root: python -m unittest tests.test_start_method"""
import multiprocessing as mp
import unittest
from multiprocessing import forkserver
from unittest import mock

from optlens import backends


@unittest.skipUnless("forkserver" in mp.get_all_start_methods(), "no forkserver on this platform")
class TestStartMethod(unittest.TestCase):
    def setUp(self):
        self.saved, backends._CTX = backends._CTX, None

    def tearDown(self):
        backends._CTX = self.saved

    def test_a_sandbox_that_blocks_the_forkserver_socket_gets_spawn(self):
        with mock.patch.object(forkserver, "ensure_running", side_effect=PermissionError(1, "Operation not permitted")):
            self.assertEqual(backends._context().get_start_method(), "spawn")

    def test_forkserver_where_it_starts(self):
        with mock.patch.object(forkserver, "ensure_running"):
            self.assertEqual(backends._context().get_start_method(), "forkserver")


if __name__ == "__main__":
    unittest.main()
