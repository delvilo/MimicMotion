import unittest
import torch
from mimicmotion.modules.unet import UNetSpatioTemporalConditionModel

class UnetTimestepTests(unittest.TestCase):
    def test_timestep_types(self):
        num_frames = 2
        model = UNetSpatioTemporalConditionModel(
            sample_size=16,
            in_channels=8,
            out_channels=4,
            block_out_channels=(32, 64, 128, 128),
            num_frames=num_frames,
            num_attention_heads=(1, 2, 4, 4),
        )
        model.eval()

        sample = torch.randn(1, num_frames, 8, 16, 16)
        encoder_hidden_states = torch.randn(1, 1, 1024)
        added_time_ids = torch.tensor([[7, 127, 0.02]])

        # 0D Tensor
        t_0d = torch.tensor(10)
        out_0d = model(sample, t_0d, encoder_hidden_states, added_time_ids, return_dict=False)[0]
        self.assertEqual(out_0d.shape, (1, num_frames, 4, 16, 16))

        # 1D Tensor
        t_1d = torch.tensor([10])
        out_1d = model(sample, t_1d, encoder_hidden_states, added_time_ids, return_dict=False)[0]
        self.assertTrue(torch.allclose(out_0d, out_1d))

        # int scalar
        out_int = model(sample, 10, encoder_hidden_states, added_time_ids, return_dict=False)[0]
        self.assertTrue(torch.allclose(out_0d, out_int))

        # float scalar
        out_float = model(sample, 10.0, encoder_hidden_states, added_time_ids, return_dict=False)[0]
        self.assertEqual(out_float.shape, (1, num_frames, 4, 16, 16))


if __name__ == "__main__":
    unittest.main()
