"""Video I/O utilities powered by TorchCodec, replacing decord and torchvision.io."""
import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

try:
    from torchcodec.decoders import VideoDecoder
    from torchcodec.encoders import VideoEncoder
    HAS_TORCHCODEC = True
except ImportError:
    HAS_TORCHCODEC = False
    VideoDecoder = None
    VideoEncoder = None


class FrameWrapper:
    """Wraps a single frame tensor to provide .asnumpy() matching decord's API."""

    def __init__(self, tensor: torch.Tensor):
        # Tensor is expected to be [H, W, C] uint8
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4) and tensor.shape[0] < tensor.shape[-1]:
            tensor = tensor.permute(1, 2, 0)
        self._tensor = tensor

    def asnumpy(self) -> np.ndarray:
        if isinstance(self._tensor, np.ndarray):
            return self._tensor
        return self._tensor.detach().cpu().numpy()

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._tensor.shape)

    @property
    def data(self) -> torch.Tensor:
        return self._tensor


class FrameBatchWrapper:
    """Wraps a batch of frame tensors to provide .asnumpy() matching decord's API."""

    def __init__(self, tensor: torch.Tensor):
        # Tensor is expected to be [N, H, W, C] uint8
        if tensor.ndim == 4 and tensor.shape[1] in (1, 3, 4) and tensor.shape[1] < tensor.shape[-1]:
            tensor = tensor.permute(0, 2, 3, 1)
        self._tensor = tensor

    def asnumpy(self) -> np.ndarray:
        if isinstance(self._tensor, np.ndarray):
            return self._tensor
        return self._tensor.detach().cpu().numpy()

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self._tensor.shape)

    @property
    def data(self) -> torch.Tensor:
        return self._tensor


class TorchVideoReader:
    """A drop-in video reader replacement for decord.VideoReader using TorchCodec.

    Supports indexing `reader[i].asnumpy()`, batch slicing `reader.get_batch(indices).asnumpy()`,
    `len(reader)`, and `reader.get_avg_fps()`.
    """

    def __init__(self, video_path: Union[str, Path], ctx: Optional[Any] = None):
        self.video_path = str(video_path)
        self.ctx = ctx
        self._device = "cpu"
        if ctx is not None:
            ctx_str = str(ctx).lower()
            if "cuda" in ctx_str or "gpu" in ctx_str:
                self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

        if HAS_TORCHCODEC:
            try:
                self.decoder = VideoDecoder(self.video_path, device=self._device, dimension_order="NHWC")
            except Exception as e:
                logger.warning("TorchCodec failed with device %s, falling back to CPU: %s", self._device, e)
                self.decoder = VideoDecoder(self.video_path, device="cpu", dimension_order="NHWC")

            meta = self.decoder.metadata
            self.fps = float(meta.average_fps if meta.average_fps and meta.average_fps > 0 else 24.0)
            self.num_frames = int(meta.num_frames if meta.num_frames and meta.num_frames > 0 else len(self.decoder))
            self.width = int(meta.width) if meta.width else 0
            self.height = int(meta.height) if meta.height else 0
            self._fallback_cv2 = None
        else:
            # Fallback using OpenCV if TorchCodec is unavailable
            import cv2
            self.decoder = None
            self._fallback_cv2 = cv2.VideoCapture(self.video_path)
            self.fps = float(self._fallback_cv2.get(cv2.CAP_PROP_FPS) or 24.0)
            self.num_frames = int(self._fallback_cv2.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            self.width = int(self._fallback_cv2.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            self.height = int(self._fallback_cv2.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    def get_avg_fps(self) -> float:
        return self.fps

    def __len__(self) -> int:
        return self.num_frames

    def __getitem__(self, idx: int) -> FrameWrapper:
        if idx < 0:
            idx += self.num_frames
        if idx < 0 or idx >= self.num_frames:
            raise IndexError(f"Index {idx} out of range for video with {self.num_frames} frames")

        if self.decoder is not None:
            frame = self.decoder[idx]
            tensor = frame.data if hasattr(frame, "data") else frame
            return FrameWrapper(tensor)
        else:
            import cv2
            self._fallback_cv2.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, bgr = self._fallback_cv2.read()
            if not ret:
                raise RuntimeError(f"Failed to read frame {idx} from {self.video_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            return FrameWrapper(torch.from_numpy(rgb))

    def get_batch(self, indices: List[int]) -> FrameBatchWrapper:
        if not indices:
            return FrameBatchWrapper(torch.empty((0, self.height, self.width, 3), dtype=torch.uint8))

        if self.decoder is not None:
            batch = self.decoder.get_frames_at(indices)
            tensor = batch.data if hasattr(batch, "data") else batch
            return FrameBatchWrapper(tensor)
        else:
            frames = [self[i].asnumpy() for i in indices]
            return FrameBatchWrapper(torch.from_numpy(np.stack(frames)))

    def get_frame_timestamp(self, indices: List[int]) -> np.ndarray:
        duration = 1.0 / self.fps if self.fps > 0 else 0.04
        ts = []
        for i in indices:
            start = i * duration
            end = start + duration
            ts.append([start, end])
        return np.asarray(ts, dtype=np.float64)

    def close(self):
        if self._fallback_cv2 is not None:
            self._fallback_cv2.release()


def save_to_mp4(
    frames: Union[torch.Tensor, np.ndarray],
    save_path: Union[str, Path],
    fps: Union[int, float] = 7,
    codec: str = "h264",
) -> None:
    """Save video frames to MP4 file using TorchCodec VideoEncoder.

    Args:
        frames: Video frames tensor of shape (N, C, H, W) or (N, H, W, C), uint8 or float [0, 1].
        save_path: Path to target MP4 file.
        fps: Target frame rate.
        codec: Video codec (default: 'h264').
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(frames, np.ndarray):
        frames = torch.from_numpy(frames)

    # Convert floating point frames in [0, 1] to uint8 [0, 255]
    if frames.is_floating_point():
        frames = (frames.clamp(0, 1) * 255.0).to(torch.uint8)
    elif frames.dtype != torch.uint8:
        frames = frames.to(torch.uint8)

    # Bring to CPU if needed for encoder
    frames = frames.detach().cpu()

    # Ensure format is (N, C, H, W) or (N, H, W, C)
    if frames.ndim != 4:
        raise ValueError(f"Expected 4D video tensor (N, C, H, W) or (N, H, W, C), got {frames.shape}")

    # Standardize to (N, C, H, W) if it was (N, H, W, C) where C=3
    if frames.shape[-1] in (1, 3) and frames.shape[1] not in (1, 3):
        frames = frames.permute(0, 3, 1, 2)

    if HAS_TORCHCODEC and VideoEncoder is not None:
        encoder = VideoEncoder(frames, frame_rate=float(fps))
        encoder.to_file(str(save_path), codec=codec)
    else:
        # Fallback using OpenCV if TorchCodec is not installed
        import cv2
        frames_np = frames.permute(0, 2, 3, 1).numpy()
        h, w = frames_np.shape[1:3]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_path), fourcc, float(fps), (w, h))
        for frame in frames_np:
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)
        writer.release()
