import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from PIL import Image

from mimicmotion.utils.segmenter import VideoSegmenter


class TestVideoSegmenter(unittest.TestCase):

    def test_segmenter_with_mock_model(self):
        with tempfile.TemporaryDirectory() as temp:
            temp_path = Path(temp)
            img = Image.new("RGB", (64, 64), color="red")
            img.save(temp_path / "000000.jpg")
            img.save(temp_path / "000001.jpg")

            out_dir = temp_path / "masks"

            mock_model = MagicMock()
            # return shape [1, 1, 1024, 1024]
            mock_model.return_value = torch.ones((1, 1, 1024, 1024), dtype=torch.float32) * 5.0

            segmenter = VideoSegmenter(model_name_or_path="mock", device="cpu")
            segmenter.model = mock_model

            mask = segmenter.segment_image(img)
            self.assertIsInstance(mask, np.ndarray)
            self.assertEqual(mask.shape, (64, 64))
            self.assertEqual(mask.dtype, np.uint8)
            self.assertTrue((mask == 255).all())

            # Test directory segmentation
            mask_paths = segmenter.segment_directory(temp_path, out_dir)
            self.assertEqual(len(mask_paths), 2)
            self.assertTrue(all(p.is_file() for p in mask_paths))


if __name__ == "__main__":
    unittest.main()
