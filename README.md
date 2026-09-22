# MimicMotion [ICML 2025]

<a href='http://tencent.github.io/MimicMotion'><img src='https://img.shields.io/badge/Project-Page-Green'></a> <a href='https://arxiv.org/abs/2406.19680'><img src='https://img.shields.io/badge/Paper-Arxiv-red'></a>

<p align="center">
<b>MimicMotion: High-Quality Human Motion Video Generation with Confidence-aware Pose Guidance</b>
<br/>
<i>Yuang Zhang<sup>1,2</sup>, Jiaxi Gu<sup>1</sup>, Li-Wen Wang<sup>1</sup>, Han Wang<sup>1,2</sup>, Junqi Cheng<sup>1</sup>, Yuefeng Zhu<sup>1</sup>, Fangyuan Zou<sup>1</sup></i>
<br/>
[<sup>1</sup>Tencent  <sup>2</sup>Shanghai Jiao Tong University]
</p>

<p align="center">
  <img src="assets/figures/preview_1.gif" width="100" />
  <img src="assets/figures/preview_2.gif" width="100" />
  <img src="assets/figures/preview_3.gif" width="100" />
  <img src="assets/figures/preview_4.gif" width="100" />
  <img src="assets/figures/preview_5.gif" width="100" />
  <img src="assets/figures/preview_6.gif" width="100" />
  <br/>
  <span>Highlights: <b>rich details</b>, <b> good temporal smoothness</b>, and <b>long video length</b>. </span>
</p>

## Overview

<p align="center">
  <img src="assets/figures/model_structure.png" alt="model architecture" width="640"/>
  </br>
  <i>An overview of the framework of MimicMotion.</i>
</p>

In recent years, generative artificial intelligence has achieved significant advancements in the field of image generation, spawning a variety of applications. However, video generation still faces considerable challenges in various aspects such as controllability, video length, and richness of details, which hinder the application and popularization of this technology. In this work, we propose a controllable video generation framework, dubbed *MimicMotion*, which can generate high-quality videos of arbitrary length with any motion guidance. Comparing with previous methods, our approach has several highlights. Firstly, with confidence-aware pose guidance, temporal smoothness can be achieved so model robustness can be enhanced with large-scale training data. Secondly, regional loss amplification based on pose confidence significantly eases the distortion of image significantly. Lastly, for generating long smooth videos, a progressive latent fusion strategy is proposed. By this means, videos of arbitrary length can be generated with acceptable resource consumption. With extensive experiments and user studies, MimicMotion demonstrates significant improvements over previous approaches in multiple aspects.

## News

* `[2025-05-03]`: &#x1F389; Our paper is accepted by ICML 2025. Congratulations and many thanks to the co-authors!
* `[2024-07-08]`: 🔥 [A superior model checkpoint](https://huggingface.co/tencent/MimicMotion/blob/main/MimicMotion_1-1.pth) has been released as version 1.1. The maximum number of video frames has now been expanded from 16 to 72, significantly enhancing the video quality!
* `[2024-07-01]`: Project page, code, technical report and [a basic model checkpoint](https://huggingface.co/tencent/MimicMotion/blob/main/MimicMotion_1.pth) are released. A better checkpoint supporting higher quality video generation will be released very soon. Stay tuned!

## Quickstart

For the initial released version of the model checkpoint, it supports generating videos with a maximum of 72 frames at a 576x1024 resolution. If you encounter insufficient memory issues, you can appropriately reduce the number of frames.

### Environment setup

Recommend python 3+ with torch 2.x are validated with an Nvidia V100 GPU. Follow the command below to install all the dependencies of python:

```
conda env create -f environment.yaml
conda activate mimicmotion
```

### Download weights
If you experience connection issues with Hugging Face, you can utilize the mirror endpoint by setting the environment variable: `export HF_ENDPOINT=https://hf-mirror.com`.
Please download weights manually as follows:
```
cd MimicMotions/
mkdir models
```
1. Download DWPose pretrained model: [dwpose](https://huggingface.co/yzd-v/DWPose/tree/main)
    ```
    mkdir -p models/DWPose
    wget https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx?download=true -O models/DWPose/yolox_l.onnx
    wget https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx?download=true -O models/DWPose/dw-ll_ucoco_384.onnx
    ```
2. Download the pre-trained checkpoint of MimicMotion from [Huggingface](https://huggingface.co/tencent/MimicMotion)
    ```
    wget -P models/ https://huggingface.co/tencent/MimicMotion/resolve/main/MimicMotion_1-1.pth
    ```
3. The SVD model [stabilityai/stable-video-diffusion-img2vid-xt-1-1](https://huggingface.co/stabilityai/stable-video-diffusion-img2vid-xt-1-1) will be automatically downloaded.

Finally, all the weights should be organized in models as follows

```
models/
├── DWPose
│   ├── dw-ll_ucoco_384.onnx
│   └── yolox_l.onnx
└── MimicMotion_1-1.pth
```

### Model inference

The default mode is now **source-first actor replacement** (`--mode replace`): keep the source shot, generate only the replacement actor for compositing, recover uncovered background from registered original frames, and use LaMa only for remaining holes. Full setup, model dependencies, commands, and limitations are in [REPLACEMENT.md](REPLACEMENT.md).

```bash
python inference.py --mode replace \
  --ref_video_path dance.mp4 --ref_image_path new_actor.jpg \
  --sam2_python /path/to/sam2-env/bin/python \
  --sam2_checkpoint models/sam2.1_hiera_small.pt \
  --lama_checkpoint models/big-lama.pt \
  --device cuda:0 --output_file outputs/replaced.mp4
```

Replacement preserves source frame count and FPS: omit `--fps` and use stride 1. It requires a separate SAM 2 environment, SAM 2 / LaMa weights and FFmpeg. Neural quality has not been validated on real footage in the delivery environment; it does not guarantee perfect motion or matting.

#### Previous full-frame generation mode

Pass all parameters on the command line; no YAML is loaded. `--help` works without importing model dependencies.

```bash
python inference.py --mode generate \
  --ref_video_path assets/example_data/videos/pose1.mp4 \
  --ref_image_path assets/example_data/images/demo1.jpg \
  --device cuda:0 --dtype float16 \
  --num_frames 72 --frames_overlap 6 \
  --decode_chunk_size 8 --conditioning_fps 7 --fps 15 \
  --output_file outputs/demo.mp4
```

For a batch, pass videos and images in corresponding order. One image may be reused for all videos. The diffusion model is loaded once; pose sessions are released after extraction and recreated for the next task to conserve GPU memory.

```bash
python inference.py --mode generate \
  --ref_video_path clips/a.mp4 clips/b.mp4 \
  --ref_image_path images/a.jpg images/b.jpg \
  --device cuda:0 --dtype float16 \
  --output_dir outputs/batch --continue_on_error
```

| Option | Behavior / default |
| --- | --- |
| `--base_model_path` | Local directory or `stabilityai/stable-video-diffusion-img2vid-xt-1-1` |
| `--ckpt_path` | `models/MimicMotion_1-1.pth` |
| `--device` | `auto`: CUDA when available, otherwise CPU; also accepts `cuda:N` |
| `--dtype` | CUDA: `float16`; CPU: `float32`. `--no_use_float16` is an alias for `--dtype float32` |
| `--num_frames`, `--frames_overlap` | Temporal tile size `72`, overlap `6`; tile size is not output duration |
| `--resolution`, `--sample_stride` | `576` (multiple of 64), `2` |
| `--num_inference_steps`, `--guidance_scale` | `25`, `2.0`; guidance must exceed 1 for the current pipeline |
| `--noise_aug_strength`, `--seed` | `0`, `42` |
| `--decode_chunk_size` | `8`; lowering it reduces decoding memory demand |
| `--conditioning_fps` | Model conditioning FPS: `7`; separate from export FPS |
| `--fps` | Export FPS: `15`; changing it changes playback duration |
| `--output_file` | Exact MP4 path, single task only |
| `--output_dir` | `outputs/`; also stores run logs and summary |
| `--overwrite` | Explicitly allow replacing output MP4 and sidecar JSON |
| `--continue_on_error` | Process remaining tasks after failure; exit status remains nonzero |
| `--log_file`, `--log_level` | Optional custom log path; default level `INFO` |

Inputs and media are checked before loading the model. A video must provide at least `num_frames - 1` sampled frames (the reference adds one). The effective sampling stride is `sample_stride * max(1, int(source_fps / 24))`. Short clips are rejected with guidance to lower the tile size or stride; they are not silently padded. Missing or degenerate pose data is also rejected.

The loader uses explicit component dtypes, without changing the global PyTorch default. It retains the original pretrained `fp16` weight variant, including when computing in FP32. CUDA pose extraction requires ONNX Runtime's CUDA provider and uses the selected GPU index. CPU mode uses FP32; actual hardware inference must be validated in your installed environment.

Each successful MP4 has a JSON sidecar containing effective arguments, source media properties, checkpoint SHA-256, model path/ID, environment versions, and generation timings. A run summary records successes, failures, and unprocessed task counts. A model ID is not an immutable remote model revision; these records help trace runs but do not guarantee bit-identical results across hardware or dependency versions.

Output names preserve multi-dot stems and include a unique run ID. Videos are encoded to a temporary file in the destination directory before publication. Without `--overwrite`, publication uses a hard link to prevent overwriting a concurrently created target (the filesystem must support hard links). MP4 and JSON are published separately: if JSON publication fails, the completed MP4 may remain, and the summary reports the output failure.

Exit codes: `0` success, `2` invalid input/options, `3` model/runtime failure, `4` GPU out of memory, `5` file/output failure, `130` interruption. Batch processing returns the first task failure code. Initialization failures stop the entire batch.

Tips: if your GPU memory is limited, try set env `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256`.

### VRAM requirement and Runtime

For the 35s demo video, the 72-frame model requires 16GB VRAM (4060ti) and finishes in 20 minutes on a 4090 GPU.

The minimum VRAM requirement for the 16-frame U-Net model is 8GB; however, the VAE decoder demands 16GB. You have the option to run the VAE decoder on CPU.

## Citation	
```bib
@inproceedings{zhang2025mimicmotion,
  title={MimicMotion: High-Quality Human Motion Video Generation with Confidence-aware Pose Guidance},
  author={Yuang Zhang and Jiaxi Gu and Li-Wen Wang and Han Wang and Junqi Cheng and Yuefeng Zhu and Fangyuan Zou},
  booktitle={International Conference on Machine Learning},
  year={2025}
}
```
