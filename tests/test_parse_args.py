"""Unit tests for CLI argument parsing and validation in inference.py."""
import contextlib
import io
import math
import unittest
from inference import parse_args


class ParseArgsTests(unittest.TestCase):

    def parse(self, *extra_args):
        """Helper to parse arguments with required flags provided by default."""
        base = ["--ref_video_path", "video.mp4", "--ref_image_path", "image.png"]
        return parse_args(base + list(extra_args))

    def parse_error(self, *extra_args):
        """Helper to verify that parse_args raises SystemExit on invalid input."""
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                self.parse(*extra_args)

    def test_default_args(self):
        args = self.parse()
        self.assertEqual(args.ref_video_path, ["video.mp4"])
        self.assertEqual(args.ref_image_path, ["image.png"])
        self.assertEqual(args.mode, "replace")
        self.assertEqual(args.sample_stride, 1)
        self.assertIsNone(args.fps)
        self.assertEqual(args.num_frames, 72)
        self.assertEqual(args.resolution, 576)
        self.assertEqual(args.frames_overlap, 6)
        self.assertEqual(args.num_inference_steps, 25)
        self.assertEqual(args.noise_aug_strength, 0.0)
        self.assertEqual(args.guidance_scale, 2.0)
        self.assertEqual(args.seed, 42)
        self.assertEqual(args.output_dir, "outputs/")
        self.assertEqual(args.device, "auto")
        self.assertIsNone(args.dtype)

    def test_generate_mode_defaults(self):
        args = self.parse("--mode", "generate")
        self.assertEqual(args.mode, "generate")
        self.assertEqual(args.sample_stride, 2)
        self.assertEqual(args.fps, 15)

    def test_missing_required_args(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args([])
            with self.assertRaises(SystemExit):
                parse_args(["--ref_video_path", "video.mp4"])
            with self.assertRaises(SystemExit):
                parse_args(["--ref_image_path", "image.png"])

    def test_valid_custom_args(self):
        args = self.parse(
            "--num_frames", "64",
            "--resolution", "512",
            "--frames_overlap", "8",
            "--num_inference_steps", "30",
            "--noise_aug_strength", "0.1",
            "--guidance_scale", "2.5",
            "--seed", "123",
            "--dtype", "float16",
            "--mode", "generate",
            "--sample_stride", "4",
            "--fps", "24"
        )
        self.assertEqual(args.num_frames, 64)
        self.assertEqual(args.resolution, 512)
        self.assertEqual(args.frames_overlap, 8)
        self.assertEqual(args.num_inference_steps, 30)
        self.assertAlmostEqual(args.noise_aug_strength, 0.1)
        self.assertAlmostEqual(args.guidance_scale, 2.5)
        self.assertEqual(args.seed, 123)
        self.assertEqual(args.dtype, "float16")
        self.assertEqual(args.sample_stride, 4)
        self.assertEqual(args.fps, 24)

    def test_no_use_float16_flag(self):
        args = self.parse("--no_use_float16")
        self.assertEqual(args.dtype, "float32")

        args_with_dtype = self.parse("--no_use_float16", "--dtype", "float32")
        self.assertEqual(args_with_dtype.dtype, "float32")

        self.parse_error("--no_use_float16", "--dtype", "float16")

    def test_resolution_validation(self):
        self.parse_error("--resolution", "500")  # Not multiple of 64
        self.parse_error("--resolution", "0")    # Must be > 0
        self.parse_error("--resolution", "-64")  # Negative

    def test_num_frames_validation(self):
        self.parse_error("--num_frames", "1")   # Must be >= 2
        self.parse_error("--num_frames", "0")   # Non-positive
        self.parse_error("--num_frames", "-10") # Negative

    def test_frames_overlap_validation(self):
        self.parse_error("--frames_overlap", "-1")   # Negative
        self.parse_error("--frames_overlap", "72")   # >= num_frames (default 72)
        self.parse_error("--frames_overlap", "100")  # > num_frames

    def test_guidance_scale_validation(self):
        self.parse_error("--guidance_scale", "1.0")  # Must be > 1
        self.parse_error("--guidance_scale", "0.5")
        self.parse_error("--guidance_scale", "-2.0")

    def test_noise_aug_strength_validation(self):
        self.parse_error("--noise_aug_strength", "-0.1")

    def test_seed_validation(self):
        self.parse_error("--seed", "-1")
        self.parse_error("--seed", str(2**63))

    def test_ref_image_and_video_path_matching(self):
        # 1 image, multiple videos: valid
        args = parse_args([
            "--ref_video_path", "v1.mp4", "v2.mp4",
            "--ref_image_path", "img.png"
        ])
        self.assertEqual(len(args.ref_video_path), 2)
        self.assertEqual(len(args.ref_image_path), 1)

        # Equal number of images and videos: valid
        args = parse_args([
            "--ref_video_path", "v1.mp4", "v2.mp4",
            "--ref_image_path", "img1.png", "img2.png"
        ])
        self.assertEqual(len(args.ref_video_path), 2)
        self.assertEqual(len(args.ref_image_path), 2)

        # Mismatched counts (2 videos, 3 images or 3 videos, 2 images): invalid
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args([
                    "--ref_video_path", "v1.mp4", "v2.mp4",
                    "--ref_image_path", "i1.png", "i2.png", "i3.png"
                ])
            with self.assertRaises(SystemExit):
                parse_args([
                    "--ref_video_path", "v1.mp4", "v2.mp4", "v3.mp4",
                    "--ref_image_path", "i1.png", "i2.png"
                ])

    def test_output_file_validation(self):
        # Standard output_file must end with .mp4
        args = self.parse("--output_file", "out.mp4")
        self.assertEqual(args.output_file, "out.mp4")

        # Wrong extension for standard output
        self.parse_error("--output_file", "out.mov")

        # Transparent background requires .mov
        args_trans = self.parse("--transparent_background", "--output_file", "out.mov")
        self.assertEqual(args_trans.output_file, "out.mov")

        # Wrong extension for transparent output
        self.parse_error("--transparent_background", "--output_file", "out.mp4")

        # Multiple videos with --output_file is invalid
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args([
                    "--ref_video_path", "v1.mp4", "v2.mp4",
                    "--ref_image_path", "img.png",
                    "--output_file", "out.mp4"
                ])

    def test_replacement_cli_validations(self):
        # In replace mode, sample_stride must be 1 and fps must be omitted
        self.parse_error("--mode", "replace", "--sample_stride", "2")
        self.parse_error("--mode", "replace", "--fps", "15")

        # transparent_background cannot be used with source_masks or occlusion_masks
        self.parse_error("--transparent_background", "--source_masks", "masks_dir")
        self.parse_error("--transparent_background", "--occlusion_masks", "masks_dir")

        # background_candidates < 0 or mask_padding out of range
        self.parse_error("--background_candidates", "-1")
        self.parse_error("--mask_padding", "-1")
        self.parse_error("--mask_padding", "65")

        # edge_feather out of range or not finite
        self.parse_error("--edge_feather", "-1")
        self.parse_error("--edge_feather", "21")

        # person_box invalid (x2 <= x1 or y2 <= y1)
        self.parse_error("--person_box", "10", "10", "5", "20")
        self.parse_error("--person_box", "10", "10", "20", "5")

        # Single-task options with multiple videos
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                parse_args([
                    "--ref_video_path", "v1.mp4", "v2.mp4",
                    "--ref_image_path", "i1.png", "i2.png",
                    "--person_box", "0", "0", "10", "10"
                ])


if __name__ == "__main__":
    unittest.main()
