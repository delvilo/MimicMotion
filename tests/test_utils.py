import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import torch

from mimicmotion.utils.utils import save_to_mp4


class UtilsTests(unittest.TestCase):

    def test_save_to_mp4_default_fps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "sub_dir" / "output.mp4"
            # frames tensor with shape (frames=5, channels=3, height=10, width=20)
            frames = torch.zeros((5, 3, 10, 20), dtype=torch.uint8)

            with patch("mimicmotion.utils.video_codec.HAS_TORCHCODEC", True), \
                 patch("mimicmotion.utils.video_codec.VideoEncoder") as mock_encoder:
                save_to_mp4(frames, str(save_path))

                # Ensure parent directory was created
                self.assertTrue(save_path.parent.exists())

                # Ensure VideoEncoder was called with expected arguments
                mock_encoder.assert_called_once()
                call_args, call_kwargs = mock_encoder.call_args
                self.assertEqual(call_kwargs.get("frame_rate"), 7.0)
                mock_encoder.return_value.to_file.assert_called_once_with(str(save_path), codec="h264")

    def test_save_to_mp4_custom_fps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "nested" / "deep" / "video.mp4"
            frames = torch.zeros((2, 3, 8, 8), dtype=torch.uint8)

            with patch("mimicmotion.utils.video_codec.HAS_TORCHCODEC", True), \
                 patch("mimicmotion.utils.video_codec.VideoEncoder") as mock_encoder:
                save_to_mp4(frames, save_path, fps=24)

                self.assertTrue(save_path.parent.exists())
                mock_encoder.assert_called_once()
                call_args, call_kwargs = mock_encoder.call_args
                self.assertEqual(call_kwargs.get("frame_rate"), 24.0)
                mock_encoder.return_value.to_file.assert_called_once_with(str(save_path), codec="h264")


if __name__ == "__main__":
    unittest.main()
