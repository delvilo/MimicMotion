import sys
import unittest
from unittest.mock import MagicMock, patch
from types import SimpleNamespace

# Create mock objects for heavy dependencies if torch is not installed
if 'torch' not in sys.modules:
    mock_torch = MagicMock()
    mock_torch.float16 = 'float16'
    mock_torch.nn = SimpleNamespace(Module=object)
    mock_torch.utils = SimpleNamespace(checkpoint=MagicMock())
    sys.modules['torch'] = mock_torch
    sys.modules['torch.utils'] = mock_torch.utils
    sys.modules['torch.utils.checkpoint'] = mock_torch.utils.checkpoint

if 'diffusers' not in sys.modules:
    mock_diffusers = MagicMock()
    sys.modules['diffusers'] = mock_diffusers
    sys.modules['diffusers.models'] = mock_diffusers.models
    sys.modules['diffusers.schedulers'] = mock_diffusers.schedulers

if 'transformers' not in sys.modules:
    mock_transformers = MagicMock()
    sys.modules['transformers'] = mock_transformers

# Mock internal modules that might import torch
sys.modules['mimicmotion.modules.unet'] = MagicMock()
sys.modules['mimicmotion.modules.pose_net'] = MagicMock()
sys.modules['mimicmotion.pipelines.pipeline_mimicmotion'] = MagicMock()

import mimicmotion.utils.loader as loader_module


class TestLoader(unittest.TestCase):
    @patch("mimicmotion.utils.loader.MimicMotionPipeline")
    @patch("mimicmotion.utils.loader.MimicMotionModel")
    @patch("sys.modules", sys.modules)
    def test_checkpoint_key_validation(self, mock_model_cls, mock_pipeline_cls):
        checkpoint_data = {
            "unet.weight": "tensor1",
            "vae.weight": "tensor2",
            "image_encoder.weight": "tensor3",
            "pose_net.weight": "tensor4",
            "unexpected_prefix.weight": "tensor5",
        }

        with patch("torch.load", return_value=checkpoint_data):
            mock_model_instance = MagicMock()
            mock_model_cls.return_value = mock_model_instance
            mock_model_instance.load_state_dict.return_value = SimpleNamespace(
                missing_keys=[], unexpected_keys=[]
            )

            infer_config = SimpleNamespace(base_model_path="fake/path", ckpt_path="fake/ckpt.pt")

            with patch("mimicmotion.utils.loader.logger") as mock_logger:
                loader_module.create_pipeline(infer_config, device="cpu")

                # Check that warning was logged for the unexpected key
                mock_logger.warning.assert_called_with("Unexpected key in checkpoint: unexpected_prefix.weight")


if __name__ == "__main__":
    unittest.main()
