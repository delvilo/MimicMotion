import logging

import torch
import torch.utils.checkpoint
from diffusers.models import AutoencoderKLTemporalDecoder
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import CLIPImageProcessor, CLIPVisionModelWithProjection

from ..modules.unet import UNetSpatioTemporalConditionModel
from ..modules.pose_net import PoseNet
from ..pipelines.pipeline_mimicmotion import MimicMotionPipeline

logger = logging.getLogger(__name__)

class MimicMotionModel(torch.nn.Module):
    def __init__(self, base_model_path, dtype=torch.float16):
        """construnct base model components and load pretrained svd model except pose-net
        Args:
            base_model_path (str): pretrained svd model path
        """
        super().__init__()
        self.unet = UNetSpatioTemporalConditionModel.from_config(
            UNetSpatioTemporalConditionModel.load_config(base_model_path, subfolder="unet"))
        self.unet.to(dtype=dtype)
        self.vae = AutoencoderKLTemporalDecoder.from_pretrained(
            base_model_path, subfolder="vae", torch_dtype=dtype, variant="fp16")
        self.image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            base_model_path, subfolder="image_encoder", torch_dtype=dtype, variant="fp16")
        self.noise_scheduler = EulerDiscreteScheduler.from_pretrained(
            base_model_path, subfolder="scheduler")
        self.feature_extractor = CLIPImageProcessor.from_pretrained(
            base_model_path, subfolder="feature_extractor")
        # pose_net
        self.pose_net = PoseNet(noise_latent_channels=self.unet.config.block_out_channels[0]).to(dtype=dtype)

def create_pipeline(infer_config, device, dtype=torch.float16):
    """create mimicmotion pipeline and load pretrained weight

    Args:
        infer_config (str): 
        device (str or torch.device): "cpu" or "cuda:{device_id}"
    """
    mimicmotion_models = MimicMotionModel(infer_config.base_model_path, dtype=dtype)
    # State dictionaries contain tensors; do not allow arbitrary checkpoint globals.
    checkpoint = torch.load(infer_config.ckpt_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError("Checkpoint must be a non-empty state dictionary")
    # Load model checkpoint
    incompatible = mimicmotion_models.load_state_dict(checkpoint, strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        logger.warning("Checkpoint mismatch: %d missing keys, %d unexpected keys",
                       len(incompatible.missing_keys), len(incompatible.unexpected_keys))
    # Explicitly cast every component; never change the process-wide default dtype.
    mimicmotion_models.to(dtype=dtype)
    # Validate checkpoint structure before loading
    for key in checkpoint:
        if not any(key.startswith(expected_prefix) for expected_prefix in ['unet', 'vae', 'image_encoder', 'pose_net']):
            logger.warning(f"Unexpected key in checkpoint: {key}")
    pipeline = MimicMotionPipeline(
        vae=mimicmotion_models.vae, 
        image_encoder=mimicmotion_models.image_encoder, 
        unet=mimicmotion_models.unet, 
        scheduler=mimicmotion_models.noise_scheduler,
        feature_extractor=mimicmotion_models.feature_extractor, 
        pose_net=mimicmotion_models.pose_net
    )
    return pipeline

