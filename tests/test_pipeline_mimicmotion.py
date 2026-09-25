import unittest
from unittest.mock import MagicMock
import numpy as np
import torch
from PIL import Image

from mimicmotion.pipelines.pipeline_mimicmotion import tensor2vid


class TestTensor2Vid(unittest.TestCase):
    def test_tensor2vid_np(self):
        # Shape: (batch_size=2, channels=3, num_frames=4, height=16, width=16)
        video = torch.randn(2, 3, 4, 16, 16)
        mock_processor = MagicMock()
        # postprocess returns numpy array per batch element
        mock_processor.postprocess.side_effect = [
            np.zeros((4, 16, 16, 3), dtype=np.float32),
            np.ones((4, 16, 16, 3), dtype=np.float32),
        ]

        output = tensor2vid(video, mock_processor, output_type="np")

        self.assertEqual(mock_processor.postprocess.call_count, 2)
        # Check that the tensor passed to postprocess was permuted from (C, F, H, W) to (F, C, H, W)
        first_call_args = mock_processor.postprocess.call_args_list[0][0]
        self.assertEqual(first_call_args[0].shape, torch.Size([4, 3, 16, 16]))
        self.assertEqual(first_call_args[1], "np")

        self.assertIsInstance(output, np.ndarray)
        self.assertEqual(output.shape, (2, 4, 16, 16, 3))
        np.testing.assert_array_equal(output[0], np.zeros((4, 16, 16, 3)))
        np.testing.assert_array_equal(output[1], np.ones((4, 16, 16, 3)))

    def test_tensor2vid_pt(self):
        video = torch.randn(1, 3, 5, 8, 8)
        mock_processor = MagicMock()
        mock_processor.postprocess.return_value = torch.zeros(5, 3, 8, 8)

        output = tensor2vid(video, mock_processor, output_type="pt")

        mock_processor.postprocess.assert_called_once()
        call_args = mock_processor.postprocess.call_args[0]
        self.assertEqual(call_args[0].shape, torch.Size([5, 3, 8, 8]))
        self.assertEqual(call_args[1], "pt")

        self.assertTrue(torch.is_tensor(output))
        self.assertEqual(output.shape, torch.Size([1, 5, 3, 8, 8]))

    def test_tensor2vid_pil(self):
        video = torch.randn(1, 3, 2, 8, 8)
        mock_processor = MagicMock()
        pil_images = [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]
        mock_processor.postprocess.return_value = pil_images

        output = tensor2vid(video, mock_processor, output_type="pil")

        mock_processor.postprocess.assert_called_once()
        call_args = mock_processor.postprocess.call_args[0]
        self.assertEqual(call_args[0].shape, torch.Size([2, 3, 8, 8]))
        self.assertEqual(call_args[1], "pil")

        self.assertIsInstance(output, list)
        self.assertEqual(len(output), 1)
        self.assertEqual(output[0], pil_images)

    def test_tensor2vid_invalid_output_type(self):
        video = torch.randn(1, 3, 2, 8, 8)
        mock_processor = MagicMock()
        mock_processor.postprocess.return_value = "invalid"

        with self.assertRaises(ValueError) as ctx:
            tensor2vid(video, mock_processor, output_type="invalid_type")

        self.assertIn("invalid_type does not exist", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
