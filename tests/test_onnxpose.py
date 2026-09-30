import unittest
from unittest.mock import MagicMock

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    np = None
    HAS_NUMPY = False

if HAS_NUMPY:
    from mimicmotion.dwpose.onnxpose import inference
    from mimicmotion.dwpose.onnxdet import _get_grid_and_strides, demo_postprocess


@unittest.skipUnless(HAS_NUMPY, "numpy is required for ONNX pose/det tests")
class TestONNXPose(unittest.TestCase):
    def test_inference(self):
        session = MagicMock()
        input_mock = MagicMock()
        input_mock.name = "input"
        output_mock = MagicMock()
        output_mock.name = "output"
        session.get_inputs.return_value = [input_mock]
        session.get_outputs.return_value = [output_mock]
        session.run.return_value = ["mock_output"]

        img = [np.zeros((256, 192, 3), dtype=np.uint8), np.ones((256, 192, 3), dtype=np.uint8)]
        outputs = inference(session, img)
        self.assertEqual(len(outputs), 2)
        self.assertEqual(session.run.call_count, 2)
        self.assertEqual(outputs[0], ["mock_output"])
        self.assertEqual(outputs[1], ["mock_output"])

    def test_get_grid_and_strides_caching_and_shapes(self):
        # Test grid and stride calculation
        img_size = (640, 640)
        grids1, strides1 = _get_grid_and_strides(img_size, p6=False)
        grids2, strides2 = _get_grid_and_strides(img_size, p6=False)

        # Verify caching (same reference)
        self.assertIs(grids1, grids2)
        self.assertIs(strides1, strides2)

        # Verify shapes for 640x640: (640/8)^2 + (640/16)^2 + (640/32)^2 = 6400 + 1600 + 400 = 8400
        self.assertEqual(grids1.shape, (1, 8400, 2))
        self.assertEqual(strides1.shape, (1, 8400, 1))

    def test_demo_postprocess(self):
        # Create dummy prediction tensor [1, 8400, 6]
        img_size = (640, 640)
        outputs = np.zeros((1, 8400, 6), dtype=np.float32)
        processed = demo_postprocess(outputs, img_size, p6=False)

        self.assertEqual(processed.shape, (1, 8400, 6))
        # First grid point at (0, 0) with stride 8 -> center (0, 0) * 8 = 0
        np.testing.assert_allclose(processed[0, 0, :2], [0.0, 0.0])
        # exp(0) * stride 8 = 8.0 for width and height
        np.testing.assert_allclose(processed[0, 0, 2:4], [8.0, 8.0])


if __name__ == "__main__":
    unittest.main()
