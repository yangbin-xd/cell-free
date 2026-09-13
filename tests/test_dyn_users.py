"""Unit tests for dyn_users.py (pure numpy helpers for moving UEs).

Run:  python3 -m unittest discover tests
Slow real-data checks are gated behind DYN_SLOW=1 (they import process.py,
which loads the full dataset).
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dyn_users as du  # noqa: E402


def _grid3x3():
    xs, ys = np.meshgrid([0.0, 10.0, 20.0], [0.0, 10.0, 20.0])
    return np.stack([xs.ravel(), ys.ravel()], axis=1).astype(np.float32)


class NearestGridTests(unittest.TestCase):
    def test_exact_grid_point_snaps_to_itself(self):
        g = _grid3x3()
        idx, dist = du.nearest_grid(g[[4, 8]], g)
        self.assertEqual(idx.tolist(), [4, 8])
        np.testing.assert_allclose(dist, 0.0, atol=1e-6)

    def test_off_grid_point_returns_closest_index_and_distance(self):
        g = _grid3x3()
        idx, dist = du.nearest_grid(np.array([[11.0, 9.0]]), g)
        self.assertEqual(int(idx[0]), 4)          # (10,10)
        self.assertAlmostEqual(float(dist[0]), np.sqrt(2.0), places=5)

    def test_snap_indices_matches_nearest_grid(self):
        g = _grid3x3()
        pts = np.array([[1.0, 1.0], [19.0, 21.0]])
        np.testing.assert_array_equal(du.snap_indices(pts, g), du.nearest_grid(pts, g)[0])


class RegionTests(unittest.TestCase):
    def test_in_region_true_inside_false_outside(self):
        g = _grid3x3()
        pts = np.array([[1.0, 1.0], [25.0, 25.0]])
        self.assertEqual(du.in_region(pts, g, r_max=3.0).tolist(), [True, False])

    def test_clamp_leaves_inside_points_unchanged(self):
        g = _grid3x3()
        pts = np.array([[1.0, 1.0], [10.5, 12.0]], dtype=np.float32)
        np.testing.assert_allclose(du.clamp_to_region(pts, g, r_max=3.0), pts)

    def test_clamp_pulls_outside_point_to_r_max_along_same_direction(self):
        g = _grid3x3()
        pts = np.array([[25.0, 20.0]])                 # 5 m east of (20,20)
        out = du.clamp_to_region(pts, g, r_max=3.0)
        np.testing.assert_allclose(out, [[23.0, 20.0]], atol=1e-5)
        self.assertFalse(np.isnan(out).any())


class SpeedToSigmaTests(unittest.TestCase):
    def test_zero_speed_gives_zero_sigma(self):
        self.assertEqual(du.speed_to_sigma(0.0, 100), 0.0)

    def test_negative_speed_clamped_to_zero(self):
        self.assertEqual(du.speed_to_sigma(-2.0, 100), 0.0)

    def test_mean_step_length_matches_speed_times_dt(self):
        # E|step| for a 2-D isotropic Gaussian with per-axis σ is σ·sqrt(pi/2)
        v, tick_ms = 1.4, 100
        sigma = du.speed_to_sigma(v, tick_ms)
        rng = np.random.default_rng(0)
        steps = rng.normal(0.0, sigma, size=(200000, 2))
        mean_len = float(np.linalg.norm(steps, axis=1).mean())
        self.assertAlmostEqual(mean_len, v * tick_ms / 1000.0, delta=0.002)


class BrownianTests(unittest.TestCase):
    def test_sigma_zero_does_not_move(self):
        g = _grid3x3()
        pts = np.array([[1.0, 1.0], [10.0, 10.0]], dtype=np.float32)
        out = du.brownian_step(pts, 0.0, g, 3.0, np.random.default_rng(0))
        np.testing.assert_array_equal(out, pts)

    def test_steps_stay_inside_region(self):
        g = _grid3x3()
        pts = np.tile([[10.0, 10.0]], (50, 1)).astype(np.float32)
        rng = np.random.default_rng(1)
        for _ in range(20):
            pts = du.brownian_step(pts, 1.0, g, 3.0, rng)
        self.assertTrue(du.in_region(pts, g, 3.0).all())

    def test_isolated_point_with_huge_sigma_stays_put(self):
        g = np.array([[0.0, 0.0]])
        pts = np.array([[0.0, 0.0]])
        out = du.brownian_step(pts, 1000.0, g, 0.5, np.random.default_rng(2))
        np.testing.assert_array_equal(out, pts)

    def test_seeded_rng_is_reproducible(self):
        g = _grid3x3()
        pts = np.array([[10.0, 10.0]])
        a = du.brownian_step(pts, 1.0, g, 3.0, np.random.default_rng(7))
        b = du.brownian_step(pts, 1.0, g, 3.0, np.random.default_rng(7))
        np.testing.assert_array_equal(a, b)
        self.assertFalse(np.array_equal(a, pts))


class RebuildLocNormTests(unittest.TestCase):
    def test_only_ue_rows_change(self):
        base = np.arange(42 * 2, dtype=np.float32).reshape(42, 2)
        mean = np.array([600.0, 250.0], dtype=np.float32)
        std = np.array([60.0, 30.0], dtype=np.float32)
        ue = np.array([[660.0, 280.0], [540.0, 220.0]], dtype=np.float32)
        out = du.rebuild_loc_norm(base, ap_num=3, ue_num=2, ue_loc=ue,
                                  loc_mean=mean, loc_std=std)
        np.testing.assert_array_equal(out[:3], base[:3])
        np.testing.assert_array_equal(out[5:], base[5:])
        np.testing.assert_allclose(out[3:5], [[1.0, 1.0], [-1.0, -1.0]], atol=1e-6)
        self.assertEqual(out.dtype, np.float32)
        self.assertFalse(np.shares_memory(out, base))

    def test_grid_bounds(self):
        xmin, xmax, ymin, ymax = du.grid_bounds(_grid3x3())
        self.assertEqual((xmin, xmax, ymin, ymax), (0.0, 20.0, 0.0, 20.0))


@unittest.skipUnless(os.environ.get("DYN_SLOW"), "set DYN_SLOW=1 to run real-data checks")
class RealDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from main import UE_loc
        from generate import loc_mean, loc_std, idx_test, UE_test
        from process import loc_test_norm, AP_num_test, UE_num_test
        cls.grid = UE_loc[:, :2]
        cls.UE_test = UE_test[:, :2]
        cls.idx_test = idx_test
        cls.mean, cls.std = loc_mean, loc_std
        cls.loc_test_norm = loc_test_norm
        cls.AP_num_test, cls.UE_num_test = AP_num_test, UE_num_test

    def test_grid_points_snap_to_themselves(self):
        idx = du.snap_indices(self.grid[:100], self.grid)
        np.testing.assert_array_equal(idx, np.arange(100))

    def test_test_split_points_snap_to_global_index(self):
        idx = du.snap_indices(self.UE_test[:50], self.grid)
        np.testing.assert_array_equal(idx, self.idx_test[:50])

    def test_rebuild_with_original_positions_reproduces_loc_test_norm(self):
        i = 0
        base = self.loc_test_norm[i].numpy()
        ap_num, ue_num = int(self.AP_num_test[i]), int(self.UE_num_test[i])
        xy = base * self.std + self.mean
        out = du.rebuild_loc_norm(base, ap_num, ue_num, xy[ap_num:ap_num + ue_num],
                                  self.mean, self.std)
        np.testing.assert_allclose(out, base, atol=1e-5)


if __name__ == "__main__":
    unittest.main()


class ReassociateTests(unittest.TestCase):
    """Handover rule during motion, mirroring main.CellFree.improved_random_selection:
    a UE may only be served by its top_l nearest APs (cap max_k UEs per AP);
    links that stay valid are kept, invalid ones dropped, at least one serving
    AP is guaranteed, and the total edge count is topped back up randomly."""
    def setUp(self):
        # 5 APs on a line at x = 0, 100, 200, 300, 400
        self.ap = np.array([[0.0, 0.0], [100.0, 0.0], [200.0, 0.0], [300.0, 0.0], [400.0, 0.0]], np.float32)

    def test_links_kept_when_still_among_nearest(self):
        ue = np.array([[150.0, 10.0]], np.float32)         # nearest 4: APs 1,2,0,3
        A = np.zeros((5, 1), np.float32); A[1, 0] = 1; A[2, 0] = 1
        A2, added, removed = du.reassociate(A, self.ap, ue, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(A2, A)
        self.assertEqual((added, removed), (set(), set()))

    def test_far_ap_dropped_and_replaced_keeping_edge_count(self):
        ue = np.array([[390.0, 0.0]], np.float32)          # nearest 4: APs 4,3,2,1 -> AP0 invalid
        A = np.zeros((5, 1), np.float32); A[0, 0] = 1; A[4, 0] = 1
        A2, added, removed = du.reassociate(A, self.ap, ue, rng=np.random.default_rng(0))
        self.assertEqual(removed, {(0, 0)})
        self.assertEqual(A2[0, 0], 0)
        self.assertEqual(A2[4, 0], 1)
        self.assertEqual(int(A2.sum()), 2)                  # count preserved
        (l, k), = added
        self.assertIn(l, (1, 2, 3))

    def test_ue_never_left_unserved(self):
        ue = np.array([[400.0, 0.0]], np.float32)
        A = np.zeros((5, 1), np.float32); A[0, 0] = 1       # only served by the far AP
        A2, added, removed = du.reassociate(A, self.ap, ue, rng=np.random.default_rng(0))
        self.assertEqual(removed, {(0, 0)})
        self.assertGreaterEqual(int(A2[:, 0].sum()), 1)
        self.assertEqual(A2[4, 0], 1)                       # nearest AP takes over first

    def test_ap_capacity_respected(self):
        ue = np.tile([[0.0, 5.0]], (10, 1)).astype(np.float32)   # 10 UEs on top of AP0
        A = np.zeros((5, 10), np.float32); A[0, :] = 1           # AP0 serves 10 > max_k
        A2, added, removed = du.reassociate(A, self.ap, ue, max_k=8, rng=np.random.default_rng(1))
        self.assertLessEqual(int(A2[0].sum()), 8)
        self.assertTrue((A2.sum(axis=0) >= 1).all())

    def test_fewer_aps_than_top_l(self):
        ap = self.ap[:2]
        ue = np.array([[50.0, 0.0]], np.float32)
        A = np.zeros((2, 1), np.float32); A[0, 0] = 1
        A2, _, _ = du.reassociate(A, ap, ue, rng=np.random.default_rng(0))
        np.testing.assert_array_equal(A2, A)

    def test_replacement_link_goes_to_the_ue_that_lost_one(self):
        # UE0 moves away from AP0 (loses it); UE1 sits still with one spare candidate.
        ue = np.array([[390.0, 0.0], [150.0, 0.0]], np.float32)
        A = np.zeros((5, 2), np.float32)
        A[0, 0] = 1; A[4, 0] = 1          # UE0: AP0 (now invalid) + AP4
        A[1, 1] = 1                       # UE1: AP1 only
        for seed in range(5):
            A2, added, removed = du.reassociate(A, self.ap, ue, rng=np.random.default_rng(seed))
            self.assertEqual(removed, {(0, 0)})
            self.assertEqual(len(added), 1)
            (l, k), = added
            self.assertEqual(k, 0, "replacement should go to UE0, which lost the link")


class RatchetRangeTests(unittest.TestCase):
    """Frozen bar-chart axes: fixed from the first values with a margin, kept
    unchanged while the data stays inside, widened only when data leaves."""
    def test_initial_range_has_margin(self):
        lo, hi = du.ratchet_range(None, [-90.0, -80.0], pad_abs=3.0, pad_rel=0.25)
        self.assertAlmostEqual(lo, -95.5); self.assertAlmostEqual(hi, -74.5)

    def test_unchanged_while_values_inside(self):
        prev = [-95.5, -74.5]
        self.assertEqual(du.ratchet_range(prev, [-89.0, -75.0], 3.0, 0.25), prev)

    def test_widens_only_the_exceeded_side(self):
        prev = [-95.5, -74.5]
        lo, hi = du.ratchet_range(prev, [-90.0, -70.0], 3.0, 0.25)
        self.assertEqual(lo, -95.5)
        self.assertGreater(hi, -70.0)

    def test_floor_and_nan_handling(self):
        lo, hi = du.ratchet_range(None, [0.5, float("nan"), float("-inf"), 2.0], 0.5, 0.4, lo_floor=0.0)
        self.assertEqual(lo, 0.0); self.assertAlmostEqual(hi, 3.1)
        self.assertIsNone(du.ratchet_range(None, [float("nan")], 0.5, 0.4))
