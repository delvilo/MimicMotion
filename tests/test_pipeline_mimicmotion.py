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


class TestPoseLatentsPrecomputation(unittest.TestCase):
    def test_pose_net_called_once_and_sliced_correctly(self):
        """Test that pose_net is called exactly once outside the timesteps loop."""
        from mimicmotion.modules.pose_net import PoseNet
        from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline

        num_frames = 4
        tile_size = 4
        num_inference_steps = 3

        # Create mock components for MimicMotionPipeline
        mock_vae = MagicMock()
        mock_vae.config.block_out_channels = [32, 64]
        mock_vae.config.scaling_factor = 0.18215
        mock_vae.dtype = torch.float32
        mock_vae.encode.return_value.latent_dist.mode.return_value = torch.zeros(1, 4, 8, 8)
        mock_vae.decode.return_value.sample = torch.zeros(num_frames, 3, 16, 16)

        mock_image_encoder = MagicMock()
        mock_image_encoder.parameters.return_value = iter([torch.tensor(0.0)])
        mock_image_encoder.return_value.image_embeds = torch.zeros(1, 768)

        mock_unet = MagicMock()
        mock_unet.config.sample_size = 2
        mock_unet.config.in_channels = 8
        mock_unet.config.num_frames = num_frames
        mock_unet.config.addition_time_embed_dim = 256
        mock_unet.add_embedding.linear_1.in_features = 768
        mock_unet.side_effect = None
        # Return dummy noise prediction with shape (1, 4, 8, 8)
        mock_unet.return_value = (torch.zeros(1, 4, 8, 8),)

        mock_scheduler = MagicMock()
        mock_scheduler.init_noise_sigma = 1.0
        mock_scheduler.timesteps = torch.tensor([100, 50, 0])
        mock_scheduler.scale_model_input.side_effect = lambda sample, t: sample
        mock_scheduler.step.return_value = (torch.zeros(1, num_frames, 4, 8, 8),)

        mock_feature_extractor = MagicMock()
        mock_pose_net = MagicMock(spec=PoseNet)
        # PoseNet precomputed latents for all 4 frames
        expected_pose_latents = torch.randn(num_frames, 320, 2, 2)
        mock_pose_net.return_value = expected_pose_latents

        pipe = MimicMotionPipeline(
            vae=mock_vae,
            image_encoder=mock_image_encoder,
            unet=mock_unet,
            scheduler=mock_scheduler,
            feature_extractor=mock_feature_extractor,
            pose_net=mock_pose_net,
        )

        dummy_image = Image.new("RGB", (16, 16))
        dummy_pose = torch.randn(num_frames, 3, 16, 16)

        pipe(
            image=dummy_image,
            image_pose=dummy_pose,
            height=16,
            width=16,
            num_frames=num_frames,
            tile_size=tile_size,
            tile_overlap=1,
            num_inference_steps=num_inference_steps,
            max_guidance_scale=1.0, # disable CFG to simplify calls
            device="cpu",
        )

        # Confirm pose_net was called exactly ONCE outside the loop with full pose tensor
        mock_pose_net.assert_called_once()
        args, _ = mock_pose_net.call_args
        self.assertTrue(torch.equal(args[0], dummy_pose))

        # Check unet calls received the precomputed pose latents slice
        # mock_unet was called 2 * num_inference_steps times (uncond pass with pose_latents=None, cond pass with pose_latents)
        self.assertEqual(mock_unet.call_count, num_inference_steps * 2)
        for i, call in enumerate(mock_unet.call_args_list):
            passed_pose_latents = call[1]["pose_latents"]
            if i % 2 == 0:
                self.assertIsNone(passed_pose_latents)
            else:
                self.assertTrue(torch.equal(passed_pose_latents, expected_pose_latents[:tile_size]))


if __name__ == "__main__":
    unittest.main()
