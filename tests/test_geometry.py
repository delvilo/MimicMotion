"""Unit tests for geometry calculations in mimicmotion.utils.replacement."""
import unittest

from mimicmotion.utils.replacement import geometry


class TestGeometry(unittest.TestCase):
    def test_square_aspect_ratio(self):
        """Test geometry calculation for a square image."""
        res = geometry(width=1000, height=1000, resolution=512)
        expected = {
            'width': 1000,
            'height': 1000,
            'resized_width': 512,
            'resized_height': 512,
            'canvas_width': 512,
            'canvas_height': 512,
            'x': 0,
            'y': 0,
        }
        self.assertEqual(res, expected)

    def test_landscape_aspect_ratio(self):
        """Test geometry calculation for a standard landscape video (1080p)."""
        res = geometry(width=1920, height=1080, resolution=576)
        expected = {
            'width': 1920,
            'height': 1080,
            'resized_width': 1024,
            'resized_height': 576,
            'canvas_width': 1024,
            'canvas_height': 576,
            'x': 0,
            'y': 0,
        }
        self.assertEqual(res, expected)

    def test_portrait_aspect_ratio(self):
        """Test geometry calculation for a standard portrait video."""
        res = geometry(width=1080, height=1920, resolution=576)
        expected = {
            'width': 1080,
            'height': 1920,
            'resized_width': 576,
            'resized_height': 1024,
            'canvas_width': 576,
            'canvas_height': 1024,
            'x': 0,
            'y': 0,
        }
        self.assertEqual(res, expected)

    def test_canvas_padding_and_centering(self):
        """Test dimensions where scaled dimensions require padding to align with 64-pixel multiples."""
        res = geometry(width=1280, height=720, resolution=512)
        # scale = 512 / 720 = 0.7111111...
        # rw = round(1280 * 512 / 720) = round(910.222) = 910
        # rh = round(720 * 512 / 720) = 512
        # cw = ceil(910 / 64) * 64 = 15 * 64 = 960
        # ch = ceil(512 / 64) * 64 = 512
        # x = (960 - 910) // 2 = 25
        # y = (512 - 512) // 2 = 0
        expected = {
            'width': 1280,
            'height': 720,
            'resized_width': 910,
            'resized_height': 512,
            'canvas_width': 960,
            'canvas_height': 512,
            'x': 25,
            'y': 0,
        }
        self.assertEqual(res, expected)

    def test_odd_padding_offset(self):
        """Test centering when canvas padding is odd."""
        # Consider width=500, height=300, resolution=300
        # scale = 300 / 300 = 1.0
        # rw = 500, rh = 300
        # cw = ceil(500/64)*64 = 8 * 64 = 512
        # ch = ceil(300/64)*64 = 5 * 64 = 320
        # x = (512 - 500) // 2 = 6
        # y = (320 - 300) // 2 = 10
        res = geometry(width=500, height=300, resolution=300)
        self.assertEqual(res['canvas_width'], 512)
        self.assertEqual(res['canvas_height'], 320)
        self.assertEqual(res['x'], 6)
        self.assertEqual(res['y'], 10)

    def test_geometry_invariants(self):
        """Test structural invariants of geometry output across multiple aspect ratios."""
        test_cases = [
            (640, 480, 512),
            (800, 600, 768),
            (1920, 1080, 512),
            (1080, 1920, 512),
            (1234, 567, 512),
            (300, 700, 256),
        ]
        for w, h, target_res in test_cases:
            with self.subTest(w=w, h=h, target_res=target_res):
                res = geometry(w, h, target_res)

                # Check canvas alignment to 64
                self.assertEqual(res['canvas_width'] % 64, 0)
                self.assertEqual(res['canvas_height'] % 64, 0)

                # Check canvas bounds
                self.assertGreaterEqual(res['canvas_width'], res['resized_width'])
                self.assertGreaterEqual(res['canvas_height'], res['resized_height'])

                # Check minimum dimension scaling
                self.assertEqual(min(w, h), w if w <= h else h)
                min_dim_scale = target_res / min(w, h)
                self.assertEqual(res['resized_width'], round(w * min_dim_scale))
                self.assertEqual(res['resized_height'], round(h * min_dim_scale))

                # Check centering offsets
                self.assertEqual(res['x'], (res['canvas_width'] - res['resized_width']) // 2)
                self.assertEqual(res['y'], (res['canvas_height'] - res['resized_height']) // 2)
                self.assertGreaterEqual(res['x'], 0)
                self.assertGreaterEqual(res['y'], 0)


if __name__ == '__main__':
    unittest.main()
