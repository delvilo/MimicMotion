import unittest
import math
from mimicmotion.utils.replacement import geometry


class GeometryTests(unittest.TestCase):
    def test_geometry_landscape(self):
        """Test standard landscape video scaling and padding calculation."""
        width, height, resolution = 1920, 1080, 576
        res = geometry(width, height, resolution)

        # min(1920, 1080) = 1080
        # scale = 576 / 1080 = 0.5333333...
        # rw = round(1920 * 0.5333333...) = 1024
        # rh = round(1080 * 0.5333333...) = 576
        # cw = ceil(1024 / 64) * 64 = 1024
        # ch = ceil(576 / 64) * 64 = 576
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

    def test_geometry_portrait(self):
        """Test portrait video scaling and padding calculation."""
        width, height, resolution = 1080, 1920, 576
        res = geometry(width, height, resolution)

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

    def test_geometry_square(self):
        """Test square video aspect ratio scaling."""
        width, height, resolution = 1000, 1000, 512
        res = geometry(width, height, resolution)

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

    def test_geometry_padding_offset_and_64_alignment(self):
        """Test cases where resized dimensions are not multiples of 64 and require canvas padding and centering."""
        # width = 1000, height = 700, resolution = 500
        # min(1000, 700) = 700
        # scale = 500 / 700 = 0.7142857142857143
        # rw = round(1000 * 500 / 700) = round(714.2857) = 714
        # rh = round(700 * 500 / 700) = round(500) = 500
        # cw = ceil(714 / 64) * 64 = 12 * 64 = 768
        # ch = ceil(500 / 64) * 64 = 8 * 64 = 512
        # x = (768 - 714) // 2 = 54 // 2 = 27
        # y = (512 - 500) // 2 = 12 // 2 = 6
        width, height, resolution = 1000, 700, 500
        res = geometry(width, height, resolution)

        expected = {
            'width': 1000,
            'height': 700,
            'resized_width': 714,
            'resized_height': 500,
            'canvas_width': 768,
            'canvas_height': 512,
            'x': 27,
            'y': 6,
        }
        self.assertEqual(res, expected)

    def test_geometry_invariants(self):
        """Verify mathematical invariants on various inputs."""
        test_inputs = [
            (1920, 1080, 576),
            (1080, 1920, 512),
            (1280, 720, 600),
            (800, 600, 300),
            (1234, 567, 400),
        ]
        for w, h, res in test_inputs:
            with self.subTest(w=w, h=h, res=res):
                g = geometry(w, h, res)
                # Dictionary structure
                self.assertSetEqual(
                    set(g.keys()),
                    {
                        'width',
                        'height',
                        'resized_width',
                        'resized_height',
                        'canvas_width',
                        'canvas_height',
                        'x',
                        'y',
                    },
                )
                self.assertEqual(g['width'], w)
                self.assertEqual(g['height'], h)

                # Canvas dimensions must be multiples of 64 and >= resized dimensions
                self.assertEqual(g['canvas_width'] % 64, 0)
                self.assertEqual(g['canvas_height'] % 64, 0)
                self.assertGreaterEqual(g['canvas_width'], g['resized_width'])
                self.assertGreaterEqual(g['canvas_height'], g['resized_height'])

                # Canvas dimensions should be the smallest multiple of 64 >= resized dimension
                self.assertEqual(
                    g['canvas_width'], math.ceil(g['resized_width'] / 64) * 64
                )
                self.assertEqual(
                    g['canvas_height'], math.ceil(g['resized_height'] / 64) * 64
                )

                # Offsets should center the resized image on the canvas
                self.assertEqual(g['x'], (g['canvas_width'] - g['resized_width']) // 2)
                self.assertEqual(
                    g['y'], (g['canvas_height'] - g['resized_height']) // 2
                )


if __name__ == '__main__':
    unittest.main()
