## 2026-09-29 - Precompute Tile Weights and Normalization Maps in Denoising Loops

**Learning:** In diffusion pipelines that employ temporal/spatial tiling (e.g. `MimicMotionPipeline`), calculating tile blending weights (`weight`), reshaping them to 4D (`weight[:, None, None, None]`), and accumulating normalization weights (`noise_pred_cnt[idx] += weight`) inside the per-timestep denoising loop creates significant Python and PyTorch overhead (repeated tensor creations, CPU-GPU memory allocations, and reshaping). Since tile indexing, tile sizes, and frame counts remain invariant across all denoising timesteps, these tensors can be computed once outside the loop.

**Action:** Always inspect the inner denoising / sampling loops of diffusion pipelines for invariant tensor allocations, reshaping, or accumulation steps, and factor them out before the timesteps loop.

## 2026-09-29 - Precompute Invariant Color Conversions and Index Arrays in Pose Drawing Loops

**Learning:** In pose visualization functions (e.g., `draw_bodypose`, `draw_handpose`), performing repeated NumPy array index offset conversions (`np.array(...) - 1`) and color space conversions (`matplotlib.colors.hsv_to_rgb(...)`) inside per-frame and per-person nested loops incurs noticeable Python interpreter and C-extension overhead across multi-frame video preprocessing. Since limb sequence definitions and edge color spectrums are fixed constants, precomputing them at module scope avoids thousands of redundant allocations per video sequence.

**Action:** Look for static indexing maps, edge sequences, or color transformations inside frame-processing loops and precompute them at module level.
