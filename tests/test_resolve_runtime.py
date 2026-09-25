import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

from inference import resolve_runtime


class ResolveRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.mock_torch = MagicMock()
        # Mocking device creation
        def mock_device(name):
            if name.startswith("invalid"):
                raise RuntimeError("Invalid device name")
            device_mock = MagicMock()
            if ":" in name:
                dev_type, dev_idx = name.split(":", 1)
                device_mock.type = dev_type
                device_mock.index = int(dev_idx)
            else:
                device_mock.type = name
                device_mock.index = None
            device_mock.__str__.return_value = name
            return device_mock

        self.mock_torch.device.side_effect = mock_device
        self.mock_torch.float32 = "torch.float32"
        self.mock_torch.float16 = "torch.float16"

    def test_auto_device_cuda_available(self):
        args = SimpleNamespace(device="auto", dtype=None)
        self.mock_torch.cuda.is_available.return_value = True
        self.mock_torch.cuda.current_device.return_value = 0
        self.mock_torch.cuda.device_count.return_value = 1

        device, dtype = resolve_runtime(args, self.mock_torch)
        self.assertEqual(args.device, "cuda:0")
        self.assertEqual(args.dtype, "float16")
        self.assertEqual(dtype, "torch.float16")

    def test_auto_device_cuda_unavailable(self):
        args = SimpleNamespace(device="auto", dtype=None)
        self.mock_torch.cuda.is_available.return_value = False

        device, dtype = resolve_runtime(args, self.mock_torch)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.dtype, "float32")
        self.assertEqual(dtype, "torch.float32")

    def test_invalid_device_name(self):
        args = SimpleNamespace(device="invalid_dev", dtype=None)
        with self.assertRaises(ValueError) as ctx:
            resolve_runtime(args, self.mock_torch)
        self.assertIn("Invalid --device: invalid_dev", str(ctx.exception))

    def test_unsupported_device_type(self):
        args = SimpleNamespace(device="mps", dtype=None)
        with self.assertRaises(ValueError) as ctx:
            resolve_runtime(args, self.mock_torch)
        self.assertIn("Only CPU and CUDA devices are supported", str(ctx.exception))

    def test_cuda_requested_when_unavailable(self):
        args = SimpleNamespace(device="cuda", dtype=None)
        self.mock_torch.cuda.is_available.return_value = False
        with self.assertRaises(ValueError) as ctx:
            resolve_runtime(args, self.mock_torch)
        self.assertIn("CUDA requested but unavailable", str(ctx.exception))

    def test_cuda_device_index_out_of_range(self):
        args = SimpleNamespace(device="cuda:2", dtype=None)
        self.mock_torch.cuda.is_available.return_value = True
        self.mock_torch.cuda.device_count.return_value = 2  # indices 0, 1 valid

        with self.assertRaises(ValueError) as ctx:
            resolve_runtime(args, self.mock_torch)
        self.assertIn("CUDA device index out of range: 2", str(ctx.exception))

    def test_cuda_default_device_index_out_of_range(self):
        args = SimpleNamespace(device="cuda", dtype=None)
        self.mock_torch.cuda.is_available.return_value = True
        self.mock_torch.cuda.current_device.return_value = 1
        self.mock_torch.cuda.device_count.return_value = 1  # only index 0 valid

        with self.assertRaises(ValueError) as ctx:
            resolve_runtime(args, self.mock_torch)
        self.assertIn("CUDA device index out of range: 1", str(ctx.exception))

    def test_cpu_with_float16_disallowed(self):
        args = SimpleNamespace(device="cpu", dtype="float16")
        with self.assertRaises(ValueError) as ctx:
            resolve_runtime(args, self.mock_torch)
        self.assertIn("Use --dtype float32 with CPU", str(ctx.exception))

    def test_valid_cpu_explicit_float32(self):
        args = SimpleNamespace(device="cpu", dtype="float32")
        device, dtype = resolve_runtime(args, self.mock_torch)
        self.assertEqual(args.device, "cpu")
        self.assertEqual(args.dtype, "float32")
        self.assertEqual(dtype, "torch.float32")

    def test_valid_cuda_explicit_device_and_dtype(self):
        args = SimpleNamespace(device="cuda:1", dtype="float32")
        self.mock_torch.cuda.is_available.return_value = True
        self.mock_torch.cuda.device_count.return_value = 2

        device, dtype = resolve_runtime(args, self.mock_torch)
        self.assertEqual(args.device, "cuda:1")
        self.assertEqual(args.dtype, "float32")
        self.assertEqual(dtype, "torch.float32")


if __name__ == "__main__":
    unittest.main()
