"""Unit tests for replacement utilities, specifically testing foreground alpha channel blending."""
import unittest
import numpy as np

from mimicmotion.utils.replacement import foreground_alpha


class TestForegroundAlpha(unittest.TestCase):

    def test_radius_zero(self):
        """When radius is 0, foreground_alpha should return raw mask scaled to [0, 1]."""
        mask = np.array([[0, 127, 255], [51, 102, 204]], dtype=np.uint8)
        expected = mask.astype(np.float32) / 255.0
        result = foreground_alpha(mask, radius=0)

        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_outside_mask_remains_zero(self):
        """Blurred foreground alpha must never extend outside the original mask support (raw == 0)."""
        mask = np.zeros((30, 30), dtype=np.uint8)
        mask[10:20, 10:20] = 255

        result = foreground_alpha(mask, radius=3)

        self.assertEqual(result.dtype, np.float32)
        # Check that pixels outside the original mask are strictly 0.0
        outside_pixels = result[mask == 0]
        np.testing.assert_array_equal(outside_pixels, 0.0)

    def test_feathering_inside_edge(self):
        """Feathering reduces alpha on interior edges while deep interior stays 1.0."""
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255

        result = foreground_alpha(mask, radius=3)

        # Inside edge pixels (e.g. at [10, 10]) should be feathered (0 < alpha < 1.0)
        self.assertGreater(result[10, 10], 0.0)
        self.assertLess(result[10, 10], 1.0)

        # Deep interior pixels (e.g. at [25, 25]) should remain 1.0
        self.assertAlmostEqual(result[25, 25], 1.0, places=4)

    def test_all_zeros_and_all_255s(self):
        """Edge cases: all-zero mask and all-255 mask."""
        zeros_mask = np.zeros((20, 20), dtype=np.uint8)
        result_zeros = foreground_alpha(zeros_mask, radius=5)
        np.testing.assert_array_equal(result_zeros, 0.0)

        ones_mask = np.full((20, 20), 255, dtype=np.uint8)
        result_ones = foreground_alpha(ones_mask, radius=5)
        np.testing.assert_allclose(result_ones, 1.0, atol=1e-5)

    def test_radius_comparison(self):
        """Larger radius produces lower alpha values near inner edges than smaller radius, but <= raw."""
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[10:30, 10:30] = 255
        raw = mask.astype(np.float32) / 255.0

        res_r1 = foreground_alpha(mask, radius=1)
        res_r5 = foreground_alpha(mask, radius=5)

        # Both results must be element-wise <= raw
        self.assertTrue(np.all(res_r1 <= raw + 1e-6))
        self.assertTrue(np.all(res_r5 <= raw + 1e-6))

        # At an inside edge pixel, radius=5 should feather more (lower alpha) than radius=1
        self.assertLess(res_r5[10, 10], res_r1[10, 10])

    def test_output_properties(self):
        """Output should maintain mask shape, float32 type, and range in [0, 1]."""
        mask = np.random.randint(0, 256, size=(35, 45), dtype=np.uint8)
        result = foreground_alpha(mask, radius=2)

        self.assertEqual(result.shape, (35, 45))
        self.assertEqual(result.dtype, np.float32)
        self.assertGreaterEqual(result.min(), 0.0)
        self.assertLessEqual(result.max(), 1.0)


if __name__ == '__main__':
    unittest.main()
