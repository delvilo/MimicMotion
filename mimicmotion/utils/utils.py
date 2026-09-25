import inspect
import logging
from pathlib import Path

from torchvision.io import write_video

logger = logging.getLogger(__name__)

def save_to_mp4(frames, save_path, fps=7):
    frames = frames.permute((0, 2, 3, 1))  # (f, c, h, w) to (f, h, w, c)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    write_video(save_path, frames, fps=fps)


def prepare_extra_step_kwargs(scheduler, generator, eta=0.0):
    # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
    # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
    # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
    # and should be between [0, 1]

    accepts_eta = "eta" in set(inspect.signature(scheduler.step).parameters.keys())
    extra_step_kwargs = {}
    if accepts_eta:
        extra_step_kwargs["eta"] = eta

    # check if the scheduler accepts generator
    accepts_generator = "generator" in set(inspect.signature(scheduler.step).parameters.keys())
    if accepts_generator:
        extra_step_kwargs["generator"] = generator
    return extra_step_kwargs
