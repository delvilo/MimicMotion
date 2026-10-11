import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from mimicmotion.utils.video_codec import TorchVideoReader, save_to_mp4, HAS_TORCHCODEC


class TestVideoCodec(unittest.TestCase):

    def test_save_and_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as temp:
            save_path = Path(temp) / "test_video.mp4"
            # 8 frames, 3 channels, 64x64
            frames = torch.randint(0, 256, (8, 3, 64, 64), dtype=torch.uint8)
            save_to_mp4(frames, save_path, fps=10)

            self.assertTrue(save_path.is_file())
            self.assertGreater(save_path.stat().st_size, 0)

            reader = TorchVideoReader(save_path, ctx="cpu")
            self.assertEqual(len(reader), 8)
            self.assertAlmostEqual(reader.get_avg_fps(), 10.0, delta=1.0)

            # Test single frame access
            f0 = reader[0]
            f0_np = f0.asnumpy()
            self.assertIsInstance(f0_np, np.ndarray)
            self.assertEqual(f0_np.shape, (64, 64, 3))

            # Test batch frame access
            batch = reader.get_batch([0, 2, 4])
            batch_np = batch.asnumpy()
            self.assertEqual(batch_np.shape, (3, 64, 64, 3))

            # Test timestamps
            ts = reader.get_frame_timestamp([0, 1, 2])
            self.assertEqual(ts.shape, (3, 2))
            self.assertAlmostEqual(ts[0, 0], 0.0)

            reader.close()


if __name__ == "__main__":
    unittest.main()
