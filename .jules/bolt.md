## 2026-09-29 - Precompute Tile Weights and Normalization Maps in Denoising Loops

**Learning:** In diffusion pipelines that employ temporal/spatial tiling (e.g. `MimicMotionPipeline`), calculating tile blending weights (`weight`), reshaping them to 4D (`weight[:, None, None, None]`), and accumulating normalization weights (`noise_pred_cnt[idx] += weight`) inside the per-timestep denoising loop creates significant Python and PyTorch overhead (repeated tensor creations, CPU-GPU memory allocations, and reshaping). Since tile indexing, tile sizes, and frame counts remain invariant across all denoising timesteps, these tensors can be computed once outside the loop.

**Action:** Always inspect the inner denoising / sampling loops of diffusion pipelines for invariant tensor allocations, reshaping, or accumulation steps, and factor them out before the timesteps loop.

## 2026-09-30 - Memoize Grid and Stride Arrays in Video Frame Detection Post-Processing

**Learning:** In frame-by-frame pose detection modules (e.g., `DWPose` detector in `mimicmotion/dwpose/onnxdet.py`), decoding bounding boxes uses `np.meshgrid` and array concatenation (`np.concatenate`) in `demo_postprocess` for every video frame. Since image input shapes (`img_size`) remain invariant across all frames of a video, allocating and concatenating these grid/stride arrays repeatedly creates significant overhead. Wrapping array creation in an `@functools.lru_cache`-backed helper function avoids thousands of unnecessary NumPy allocations across video processing tasks.

**Action:** Look for per-frame post-processing or pre-processing steps in video pipelines where array shapes are invariant across frames and memoize constant grid, stride, or mask allocations.
