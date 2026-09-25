import unittest
from unittest.mock import MagicMock, patch
import numpy as np

from mimicmotion.dwpose.preprocess import _get_video_pose, get_video_pose


class TestPreprocessVideoErrorHandling(unittest.TestCase):
    def setUp(self):
        self.ref_image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.mock_processor = MagicMock()
        valid_subset = np.ones((1, 18), dtype=float)
        valid_candidate = np.ones((18, 2), dtype=float)
        self.mock_processor.return_value = {
            'bodies': {
                'subset': valid_subset,
                'candidate': valid_candidate
            }
        }

    @patch('mimicmotion.dwpose.preprocess.decord.VideoReader')
    def test_empty_video_raises_value_error(self, mock_vr_cls):
        mock_vr = MagicMock()
        mock_vr.__len__.return_value = 0
        mock_vr.get_avg_fps.return_value = 30.0
        mock_vr_cls.return_value = mock_vr

        with self.assertRaisesRegex(ValueError, "Video is empty or has an invalid frame rate"):
            _get_video_pose("dummy_path.mp4", self.ref_image, processor=self.mock_processor)

    @patch('mimicmotion.dwpose.preprocess.decord.VideoReader')
    def test_invalid_fps_raises_value_error(self, mock_vr_cls):
        invalid_fps_values = [0, -10.0, float('nan'), float('inf'), float('-inf')]
        for fps in invalid_fps_values:
            with self.subTest(fps=fps):
                mock_vr = MagicMock()
                mock_vr.__len__.return_value = 100
                mock_vr.get_avg_fps.return_value = fps
                mock_vr_cls.return_value = mock_vr

                with self.assertRaisesRegex(ValueError, "Video is empty or has an invalid frame rate"):
                    _get_video_pose("dummy_path.mp4", self.ref_image, processor=self.mock_processor)

    @patch('mimicmotion.dwpose.preprocess.decord.VideoReader')
    def test_too_few_sampled_frames_raises_value_error(self, mock_vr_cls):
        mock_vr = MagicMock()
        mock_vr.__len__.return_value = 5
        mock_vr.get_avg_fps.return_value = 24.0
        mock_vr_cls.return_value = mock_vr

        with self.assertRaisesRegex(ValueError, "Too few sampled frames; reduce num_frames or sample_stride"):
            _get_video_pose("dummy_path.mp4", self.ref_image, sample_stride=10, processor=self.mock_processor, min_frames=5)

    @patch('mimicmotion.dwpose.preprocess.dwprocessor')
    @patch('mimicmotion.dwpose.preprocess._get_video_pose')
    def test_get_video_pose_releases_default_processor_memory(self, mock_get_video_pose, mock_dwprocessor):
        mock_get_video_pose.return_value = "pose_result"
        res = get_video_pose("dummy_path.mp4", self.ref_image)
        self.assertEqual(res, "pose_result")
        mock_dwprocessor.release_memory.assert_called_once()

    @patch('mimicmotion.dwpose.preprocess.dwprocessor')
    @patch('mimicmotion.dwpose.preprocess._get_video_pose')
    def test_get_video_pose_releases_memory_on_exception(self, mock_get_video_pose, mock_dwprocessor):
        mock_get_video_pose.side_effect = ValueError("Video is empty")
        with self.assertRaises(ValueError):
            get_video_pose("dummy_path.mp4", self.ref_image)
        mock_dwprocessor.release_memory.assert_called_once()

    @patch('mimicmotion.dwpose.preprocess._get_video_pose')
    def test_get_video_pose_does_not_release_custom_processor_memory(self, mock_get_video_pose):
        mock_get_video_pose.return_value = "pose_result"
        res = get_video_pose("dummy_path.mp4", self.ref_image, processor=self.mock_processor)
        self.assertEqual(res, "pose_result")
        self.mock_processor.release_memory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
