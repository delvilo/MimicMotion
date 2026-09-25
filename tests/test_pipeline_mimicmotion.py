import unittest
from unittest.mock import MagicMock, patch
import torch
from PIL import Image

from mimicmotion.pipelines.pipeline_mimicmotion import MimicMotionPipeline, MimicMotionPipelineOutput


class MockScheduler:
    def __init__(self):
        self.config = MagicMock()
        self.init_noise_sigma = 1.0
        self.timesteps = torch.tensor([1000, 500])

    def set_timesteps(self, num_inference_steps, device=None):
        self.timesteps = torch.tensor([1000, 500])

    def scale_model_input(self, sample, timestep):
        return sample

    def step(self, model_output, timestep, sample, **kwargs):
        return (sample - model_output * 0.1,)


class PipelineMimicMotionTests(unittest.TestCase):
    def setUp(self):
        self.vae = MagicMock()
        self.vae.config.block_out_channels = [64, 128]
        self.vae.config.scaling_factor = 0.18215
        self.vae.dtype = torch.float32

        self.image_encoder = MagicMock()
        mock_param = MagicMock()
        mock_param.dtype = torch.float32
        self.image_encoder.parameters.return_value = iter([mock_param])

        self.unet = MagicMock()
        self.unet.config.sample_size = 32
        self.unet.config.num_frames = 16
        self.unet.config.in_channels = 8
        self.unet.config.addition_time_embed_dim = 64
        self.unet.add_embedding.linear_1.in_features = 192

        self.scheduler = MockScheduler()
        self.feature_extractor = MagicMock()
        self.pose_net = MagicMock()

        self.pipeline = MimicMotionPipeline(
            vae=self.vae,
            image_encoder=self.image_encoder,
            unet=self.unet,
            scheduler=self.scheduler,
            feature_extractor=self.feature_extractor,
            pose_net=self.pose_net,
        )

    def test_get_tile_indices_valid(self):
        indices = self.pipeline._get_tile_indices(num_frames=25, tile_size=16, tile_overlap=4)
        self.assertTrue(len(indices) > 0)
        self.assertEqual(indices[0][0], 0)
        self.assertEqual(indices[-1][-1], 24)

    def test_get_tile_indices_invalid(self):
        with self.assertRaises(ValueError):
            self.pipeline._get_tile_indices(num_frames=25, tile_size=1, tile_overlap=0)

        with self.assertRaises(ValueError):
            self.pipeline._get_tile_indices(num_frames=25, tile_size=16, tile_overlap=16)

        with self.assertRaises(ValueError):
            self.pipeline._get_tile_indices(num_frames=10, tile_size=16, tile_overlap=4)

    def test_prepare_guidance_scale(self):
        guidance = self.pipeline._prepare_guidance_scale(
            min_guidance_scale=1.0,
            max_guidance_scale=3.0,
            num_frames=16,
            batch_size=1,
            num_videos_per_prompt=1,
            device="cpu",
            dtype=torch.float32,
            ndim=5,
        )
        self.assertEqual(guidance.shape[0], 1)
        self.assertEqual(guidance.shape[1], 16)
        self.assertEqual(guidance.ndim, 5)

    def test_prepare_vae_latents(self):
        self.pipeline._guidance_scale = 3.0
        image = Image.new("RGB", (64, 64))
        mock_mode = torch.zeros((1, 4, 32, 32))
        self.vae.encode.return_value.latent_dist.mode.return_value = mock_mode

        with patch.object(self.pipeline.image_processor, "preprocess", return_value=torch.zeros((1, 3, 64, 64))):
            latents = self.pipeline._prepare_vae_latents(
                image=image,
                height=64,
                width=64,
                num_frames=8,
                noise_aug_strength=0.02,
                num_videos_per_prompt=1,
                generator=None,
                device="cpu",
                dtype=torch.float32,
            )
            self.assertEqual(latents.shape, (2, 8, 4, 32, 32))

    def test_predict_noise(self):
        self.pipeline._guidance_scale = 3.0
        latent_model_input = torch.zeros((2, 16, 8, 16, 16))
        image_embeddings = torch.zeros((2, 1, 768))
        added_time_ids = torch.zeros((2, 3))
        image_pose = torch.zeros((16, 3, 64, 64))
        image_latents = torch.zeros((1, 16, 4, 16, 16))
        indices = [[0, *range(1, 16)]]

        self.pose_net.return_value = torch.zeros((16, 32, 16, 16))
        self.unet.return_value = (torch.zeros((1, 16, 4, 16, 16)),)

        progress_bar = MagicMock()

        noise_pred = self.pipeline._predict_noise(
            latent_model_input=latent_model_input,
            t=1000,
            image_embeddings=image_embeddings,
            added_time_ids=added_time_ids,
            image_pose=image_pose,
            image_latents=image_latents,
            indices=indices,
            tile_size=16,
            num_frames=16,
            image_only_indicator=False,
            device="cpu",
            progress_bar=progress_bar,
        )
        self.assertEqual(noise_pred.shape, (1, 16, 4, 16, 16))

    def test_call_latent_output(self):
        image = Image.new("RGB", (64, 64))
        image_pose = torch.zeros((16, 3, 64, 64))

        self.image_encoder.return_value.image_embeds = torch.zeros((1, 768))
        mock_mode = torch.zeros((1, 4, 32, 32))
        self.vae.encode.return_value.latent_dist.mode.return_value = mock_mode
        self.pose_net.return_value = torch.zeros((16, 32, 32, 32))
        self.unet.return_value = (torch.zeros((1, 16, 4, 32, 32)),)

        with patch.object(self.pipeline.image_processor, "preprocess", return_value=torch.zeros((1, 3, 64, 64))), \
             patch("mimicmotion.pipelines.pipeline_mimicmotion.retrieve_timesteps", return_value=(torch.tensor([1000]), 1)):
            output = self.pipeline(
                image=image,
                image_pose=image_pose,
                height=64,
                width=64,
                num_frames=16,
                tile_size=16,
                tile_overlap=4,
                output_type="latent",
                return_dict=True,
                device="cpu",
            )
            self.assertIsInstance(output, MimicMotionPipelineOutput)
            self.assertEqual(output.frames.ndim, 5)


if __name__ == "__main__":
    unittest.main()
