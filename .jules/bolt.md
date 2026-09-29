## 2026-09-29 - Precompute Tile Weights and Normalization Maps in Denoising Loops

**Learning:** In diffusion pipelines that employ temporal/spatial tiling (e.g. `MimicMotionPipeline`), calculating tile blending weights (`weight`), reshaping them to 4D (`weight[:, None, None, None]`), and accumulating normalization weights (`noise_pred_cnt[idx] += weight`) inside the per-timestep denoising loop creates significant Python and PyTorch overhead (repeated tensor creations, CPU-GPU memory allocations, and reshaping). Since tile indexing, tile sizes, and frame counts remain invariant across all denoising timesteps, these tensors can be computed once outside the loop.

**Action:** Always inspect the inner denoising / sampling loops of diffusion pipelines for invariant tensor allocations, reshaping, or accumulation steps, and factor them out before the timesteps loop.
