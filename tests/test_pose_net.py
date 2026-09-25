import tempfile
import unittest
from pathlib import Path

import torch
from mimicmotion.modules.pose_net import PoseNet


class PoseNetTests(unittest.TestCase):
    def test_from_pretrained_loads_weights_safely(self):
        model = PoseNet(noise_latent_channels=320)
        with tempfile.TemporaryDirectory() as temp_dir:
            weights_path = Path(temp_dir) / "posenet.pth"
            torch.save(model.state_dict(), weights_path)

            loaded_model = PoseNet.from_pretrained(str(weights_path))
            self.assertIsInstance(loaded_model, PoseNet)

            # Test forward pass with dummy input
            dummy_input = torch.randn(1, 3, 64, 64)
            output = loaded_model(dummy_input)
            self.assertEqual(output.shape, (1, 320, 8, 8))


if __name__ == '__main__':
    unittest.main()
