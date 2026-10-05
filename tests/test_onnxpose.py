import unittest
from unittest.mock import MagicMock
import numpy as np

from mimicmotion.dwpose.onnxpose import inference


class TestONNXPose(unittest.TestCase):
    def test_inference(self):
        session = MagicMock()
        input_mock = MagicMock()
        input_mock.name = "input"
        output_mock1 = MagicMock()
        output_mock1.name = "simcc_x"
        output_mock2 = MagicMock()
        output_mock2.name = "simcc_y"
        session.get_inputs.return_value = [input_mock]
        session.get_outputs.return_value = [output_mock1, output_mock2]

        # Mock batched model output: shape (N, K, W) for N=2 samples
        mock_simcc_x = np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
        mock_simcc_y = np.array([[[5.0, 6.0]], [[7.0, 8.0]]])
        session.run.return_value = [mock_simcc_x, mock_simcc_y]

        img = [np.zeros((256, 192, 3), dtype=np.uint8), np.ones((256, 192, 3), dtype=np.uint8)]
        outputs = inference(session, img)

        self.assertEqual(len(outputs), 2)
        # Batched inference must execute in a single session.run call
        self.assertEqual(session.run.call_count, 1)

        # Check that outputs are sliced correctly per sample
        np.testing.assert_array_equal(outputs[0][0], mock_simcc_x[0:1])
        np.testing.assert_array_equal(outputs[0][1], mock_simcc_y[0:1])
        np.testing.assert_array_equal(outputs[1][0], mock_simcc_x[1:2])
        np.testing.assert_array_equal(outputs[1][1], mock_simcc_y[1:2])


if __name__ == "__main__":
    unittest.main()
