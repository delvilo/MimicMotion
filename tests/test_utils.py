import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mimicmotion.utils.utils import save_to_mp4


class UtilsTests(unittest.TestCase):

    def test_save_to_mp4_default_fps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "sub_dir" / "output.mp4"
            # frames tensor with shape (frames=5, channels=3, height=10, width=20)
            frames = torch.zeros((5, 3, 10, 20), dtype=torch.uint8)

            with patch("mimicmotion.utils.utils.write_video") as mock_write_video:
                save_to_mp4(frames, str(save_path))

                # Ensure parent directory was created
                self.assertTrue(save_path.parent.exists())

                # Ensure write_video was called with expected arguments
                mock_write_video.assert_called_once()
                call_args, call_kwargs = mock_write_video.call_args
                self.assertEqual(call_args[0], str(save_path))

                written_frames = call_args[1]
                # Check permuted shape (f, h, w, c) -> (5, 10, 20, 3)
                self.assertEqual(written_frames.shape, (5, 10, 20, 3))
                self.assertEqual(call_kwargs.get("fps"), 7)

    def test_save_to_mp4_custom_fps(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_path = Path(temp_dir) / "nested" / "deep" / "video.mp4"
            frames = torch.zeros((2, 3, 8, 8), dtype=torch.uint8)

            with patch("mimicmotion.utils.utils.write_video") as mock_write_video:
                save_to_mp4(frames, save_path, fps=24)

                self.assertTrue(save_path.parent.exists())
                mock_write_video.assert_called_once()
                call_args, call_kwargs = mock_write_video.call_args
                self.assertEqual(call_args[0], save_path)

                written_frames = call_args[1]
                self.assertEqual(written_frames.shape, (2, 8, 8, 3))
                self.assertEqual(call_kwargs.get("fps"), 24)


if __name__ == "__main__":
    unittest.main()
