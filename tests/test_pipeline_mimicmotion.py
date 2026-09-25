import unittest
import torch

from mimicmotion.pipelines.pipeline_mimicmotion import _append_dims


class TestAppendDims(unittest.TestCase):
    def test_append_dims_happy_paths(self):
        """Test padding dimensions when target_dims > x.ndim for various shapes and dtypes."""
        test_cases = [
            (torch.tensor([1.0, 2.0, 3.0]), 3, (3, 1, 1)),
            (torch.randn(2, 4), 4, (2, 4, 1, 1)),
            (torch.randn(2, 3, 4), 5, (2, 3, 4, 1, 1)),
            (torch.tensor([[1, 2], [3, 4]], dtype=torch.int64), 5, (2, 2, 1, 1, 1)),
            (torch.tensor([[True, False]]), 3, (1, 2, 1)),
        ]
        for tensor, target_dims, expected_shape in test_cases:
            with self.subTest(shape=tensor.shape, target_dims=target_dims):
                result = _append_dims(tensor, target_dims)
                self.assertEqual(result.shape, expected_shape)
                self.assertEqual(result.ndim, target_dims)
                self.assertEqual(result.dtype, tensor.dtype)
                self.assertEqual(result.device, tensor.device)
                torch.testing.assert_close(result.squeeze(), tensor.squeeze())

    def test_append_dims_same_dimensions(self):
        """Test no-op when target_dims == x.ndim."""
        tensor = torch.randn(2, 3, 4)
        result = _append_dims(tensor, 3)
        self.assertEqual(result.shape, (2, 3, 4))
        self.assertEqual(result.ndim, 3)
        torch.testing.assert_close(result, tensor)

    def test_append_dims_fewer_target_dims_raises_value_error(self):
        """Test that ValueError is raised when target_dims < x.ndim."""
        tensor = torch.randn(2, 3, 4)  # 3D tensor
        expected_msg = "input has 3 dims but target_dims is 2, which is less"
        with self.assertRaises(ValueError) as cm:
            _append_dims(tensor, 2)
        self.assertEqual(str(cm.exception), expected_msg)

    def test_append_dims_0d_scalar(self):
        """Test scalar (0D) tensor padding."""
        scalar = torch.tensor(42.0)
        result = _append_dims(scalar, 3)
        self.assertEqual(result.shape, (1, 1, 1))
        self.assertEqual(result.ndim, 3)
        self.assertEqual(result.item(), 42.0)

    def test_append_dims_empty_tensor(self):
        """Test padding empty tensors."""
        empty_tensor = torch.empty(0, 5)
        result = _append_dims(empty_tensor, 4)
        self.assertEqual(result.shape, (0, 5, 1, 1))
        self.assertEqual(result.ndim, 4)


if __name__ == "__main__":
    unittest.main()
