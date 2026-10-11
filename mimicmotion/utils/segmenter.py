"""High-performance person and object segmentation module replacing SAM 2.

Uses pure PyTorch / Transformers (RMBG-1.4 / BiRefNet) without torchvision or torchaudio.
Runs directly in-process on GPU or CPU.
"""
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


def patch_transformers_compatibility():
    """Ensure compatibility with transformers 5.x for custom remote models."""
    try:
        from transformers import PreTrainedModel
        if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
            PreTrainedModel.all_tied_weights_keys = {}
    except ImportError:
        pass


class VideoSegmenter:
    """In-process image/video segmentation using pure PyTorch models (e.g. RMBG-1.4)."""

    def __init__(
        self,
        model_name_or_path: str = "briaai/RMBG-1.4",
        device: Optional[str] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        patch_transformers_compatibility()
        self.model_name_or_path = model_name_or_path
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.float16 if "cuda" in str(self.device) else torch.float32)
        self.model = None

    def _load_model(self):
        if self.model is not None:
            return
        from transformers import AutoModelForImageSegmentation
        logger.info("Loading segmentation model %s on %s...", self.model_name_or_path, self.device)
        try:
            model = AutoModelForImageSegmentation.from_pretrained(
                self.model_name_or_path, trust_remote_code=True
            )
            model = model.to(self.device).eval()
            self.model = model
        except Exception as e:
            logger.error("Failed to load segmentation model %s: %s", self.model_name_or_path, e)
            raise

    def segment_image(
        self,
        image: Union[Image.Image, np.ndarray],
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Segment a single RGB image and return a binary uint8 mask of shape (H, W)."""
        self._load_model()
        if isinstance(image, np.ndarray):
            orig_h, orig_w = image.shape[:2]
            pil_img = Image.fromarray(image)
        else:
            orig_w, orig_h = image.size
            pil_img = image

        # Model expects (1024, 1024)
        input_size = (1024, 1024)
        resized = pil_img.resize(input_size, Image.Resampling.BILINEAR)
        img_np = np.array(resized, dtype=np.float32) / 255.0
        # Normalize: (x - 0.5) / 1.0
        img_np = (img_np - 0.5) / 1.0
        tensor = torch.from_numpy(img_np).permute(2, 0, 1).unsqueeze(0).to(self.device).to(self.dtype)

        with torch.no_grad():
            out = self.model(tensor)
            if isinstance(out, (list, tuple)):
                pred = out[0][0] if isinstance(out[0], (list, tuple)) else out[0]
            elif hasattr(out, "logits"):
                pred = out.logits
            else:
                pred = out

            if pred.ndim == 4:
                pred = pred[0, 0]
            elif pred.ndim == 3:
                pred = pred[0]

            prob = torch.sigmoid(pred.float()).unsqueeze(0).unsqueeze(0)
            # Resize back to original dimensions
            prob_orig = F.interpolate(prob, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
            prob_map = prob_orig[0, 0].cpu().numpy()

        mask = (prob_map > threshold).astype(np.uint8) * 255
        return mask

    def segment_directory(
        self,
        input_dir: Union[str, Path],
        output_dir: Union[str, Path],
        threshold: float = 0.5,
    ) -> List[Path]:
        """Segment all images in input_dir and save corresponding masks in output_dir."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        image_files = sorted(list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png")))
        if not image_files:
            raise ValueError(f"No image files found in {input_dir}")

        mask_paths = []
        for img_path in image_files:
            with Image.open(img_path) as img:
                rgb_img = img.convert("RGB")
                mask_arr = self.segment_image(rgb_img, threshold=threshold)
            out_mask_path = output_dir / f"{img_path.stem}.png"
            Image.fromarray(mask_arr).save(out_mask_path)
            mask_paths.append(out_mask_path)

        return mask_paths
