import unittest
from unittest.mock import MagicMock
import numpy as np

from mimicmotion.dwpose.onnxpose import inference


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


if __name__ == "__main__":
    unittest.main()
