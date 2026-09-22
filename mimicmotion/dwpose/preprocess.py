from tqdm import tqdm
import decord
import numpy as np

from .util import draw_pose
from .dwpose_detector import dwpose_detector as dwprocessor


def _get_video_pose(
        video_path: str, 
        ref_image: np.ndarray, 
        sample_stride: int=1, processor=None, min_frames=2):
    """preprocess ref image pose and video pose

    Args:
        video_path (str): video pose path
        ref_image (np.ndarray): reference image 
        sample_stride (int, optional): Defaults to 1.

    Returns:
        np.ndarray: sequence of video pose
    """
    processor = processor if processor is not None else dwprocessor
    # select ref-keypoint from reference pose for pose rescale
    ref_pose = processor(ref_image)
    ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ref_keypoint_id = [i for i in ref_keypoint_id \
        if len(ref_pose['bodies']['subset']) > 0 and ref_pose['bodies']['subset'][0][i] >= .0]
    if len(ref_keypoint_id) < 2:
        raise ValueError("Reference image has fewer than two usable body keypoints")
    ref_body = ref_pose['bodies']['candidate'][ref_keypoint_id]
    if not np.isfinite(ref_body).all():
        raise ValueError("Reference image has invalid pose coordinates")

    height, width, _ = ref_image.shape

    # read input video
    vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
    source_fps = float(vr.get_avg_fps())
    if len(vr) == 0 or not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("Video is empty or has an invalid frame rate")
    sample_stride *= max(1, int(source_fps / 24))
    if len(range(0, len(vr), sample_stride)) + 1 < min_frames:
        raise ValueError("Too few sampled frames; reduce num_frames or sample_stride")

    frames = vr.get_batch(list(range(0, len(vr), sample_stride))).asnumpy()
    detected_poses = [processor(frm) for frm in tqdm(frames, desc="DWPose")]


    valid_bodies = [p['bodies']['candidate'] for p in detected_poses
                    if p['bodies']['candidate'].shape[0] == 18
                    and len(p['bodies']['subset']) == 1
                    and np.all(p['bodies']['subset'][0][ref_keypoint_id] >= 0)
                    and np.isfinite(p['bodies']['candidate']).all()]
    if not valid_bodies:
        raise ValueError("No video frame has one person with the required visible body keypoints")
    detected_bodies = np.stack(valid_bodies)[:, ref_keypoint_id]
    if np.ptp(detected_bodies[:, :, 1]) < 1e-6:
        raise ValueError("Video pose has insufficient vertical variation for alignment")
    # compute linear-rescale params
    ay, by = np.polyfit(detected_bodies[:, :, 1].flatten(), np.tile(ref_body[:, 1], len(detected_bodies)), 1)
    fh, fw, _ = vr[0].shape
    ax = ay / (fh / fw / height * width)
    bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
    if not np.isfinite([ax, ay, bx, by]).all() or ay <= 0:
        raise ValueError("Could not compute a valid pose alignment")
    a = np.array([ax, ay])
    b = np.array([bx, by])
    output_pose = []
    # pose rescale 
    for detected_pose in detected_poses:
        detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] * a + b
        detected_pose['faces'] = detected_pose['faces'] * a + b
        detected_pose['hands'] = detected_pose['hands'] * a + b
        im = draw_pose(detected_pose, height, width)
        output_pose.append(np.array(im))
    return np.stack(output_pose)


def get_video_pose(video_path, ref_image, sample_stride=1, processor=None, min_frames=2):
    """Extract aligned poses; callers supplying a detector own its lifecycle."""
    detector = processor if processor is not None else dwprocessor
    try:
        return _get_video_pose(video_path, ref_image, sample_stride, detector, min_frames)
    finally:
        if processor is None:
            detector.release_memory()


def get_image_pose(ref_image, processor=None):
    """process image pose

    Args:
        ref_image (np.ndarray): reference image pixel value

    Returns:
        np.ndarray: pose visual image in RGB-mode
    """
    processor = processor if processor is not None else dwprocessor
    height, width, _ = ref_image.shape
    ref_pose = processor(ref_image)
    subset = ref_pose['bodies']['subset']
    if len(subset) == 0 or np.count_nonzero(subset[0] >= 0) < 2:
        raise ValueError("Reference image has no usable person pose")
    pose_img = draw_pose(ref_pose, height, width)
    return np.array(pose_img)
