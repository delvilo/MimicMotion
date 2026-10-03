## 2026-09-29 - Precompute Tile Weights and Normalization Maps in Denoising Loops

**Learning:** In diffusion pipelines that employ temporal/spatial tiling (e.g. `MimicMotionPipeline`), calculating tile blending weights (`weight`), reshaping them to 4D (`weight[:, None, None, None]`), and accumulating normalization weights (`noise_pred_cnt[idx] += weight`) inside the per-timestep denoising loop creates significant Python and PyTorch overhead (repeated tensor creations, CPU-GPU memory allocations, and reshaping). Since tile indexing, tile sizes, and frame counts remain invariant across all denoising timesteps, these tensors can be computed once outside the loop.

**Action:** Always inspect the inner denoising / sampling loops of diffusion pipelines for invariant tensor allocations, reshaping, or accumulation steps, and factor them out before the timesteps loop.

## 2026-09-30 - Factor Out Static Conditioning Model Forward Passes Outside Sampling Loops

**Learning:** In video diffusion pipelines with condition injection networks (such as `PoseNet` in `MimicMotionPipeline`), evaluating conditioning model forward passes (`self.pose_net(image_pose[idx].to(device))`) inside the per-timestep and per-tile loop causes massive, redundant compute and memory transfer overhead ($T \times K$ forward passes instead of 1). Since condition inputs and conditioning networks remain invariant throughout all sampling steps, computing `pose_latents = self.pose_net(image_pose.to(device))` once outside the timesteps loop eliminates $T \times K - 1$ redundant neural network forward passes and CPU-to-GPU memory copies.

**Action:** When working with conditional diffusion pipelines, check if conditioning network outputs (e.g., pose encodings, controlnet features, reference image encodings) depend on timestep $t$ or noisy latents. If they are invariant across sampling timesteps, precompute them once before starting the sampling loop.
