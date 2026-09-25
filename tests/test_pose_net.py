import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from mimicmotion.modules.pose_net import PoseNet


class TestPoseNet(unittest.TestCase):
    def test_from_pretrained_uses_weights_only(self):
        model = PoseNet(noise_latent_channels=320)
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "posenet.pth"
            torch.save(model.state_dict(), ckpt_path)

            with patch("torch.load", wraps=torch.load) as mock_torch_load:
                loaded_model = PoseNet.from_pretrained(str(ckpt_path))
                self.assertIsInstance(loaded_model, PoseNet)
                mock_torch_load.assert_called_once()
                _, kwargs = mock_torch_load.call_args
                self.assertTrue(kwargs.get("weights_only"), "torch.load must be called with weights_only=True")


if __name__ == "__main__":
    unittest.main()
