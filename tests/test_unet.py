import unittest
import torch
from mimicmotion.modules.unet import UNetSpatioTemporalConditionModel, UNetSpatioTemporalConditionOutput


class TestUNetTimestepHandling(unittest.TestCase):
    def setUp(self):
        self.model = UNetSpatioTemporalConditionModel(
            sample_size=16,
            in_channels=8,
            out_channels=4,
            block_out_channels=(32, 64),
            down_block_types=("CrossAttnDownBlockSpatioTemporal", "DownBlockSpatioTemporal"),
            up_block_types=("UpBlockSpatioTemporal", "CrossAttnUpBlockSpatioTemporal"),
            num_attention_heads=(2, 4),
            cross_attention_dim=32,
        )
        self.sample = torch.randn(1, 2, 8, 16, 16)
        self.encoder_hidden_states = torch.randn(1, 1, 32)
        self.added_time_ids = torch.tensor([[7, 127, 0.02]])

    def test_timestep_int(self):
        out = self.model(
            sample=self.sample,
            timestep=10,
            encoder_hidden_states=self.encoder_hidden_states,
            added_time_ids=self.added_time_ids,
        )
        self.assertIsInstance(out, UNetSpatioTemporalConditionOutput)
        self.assertEqual(out.sample.shape, (1, 2, 4, 16, 16))

    def test_timestep_float(self):
        out = self.model(
            sample=self.sample,
            timestep=10.0,
            encoder_hidden_states=self.encoder_hidden_states,
            added_time_ids=self.added_time_ids,
        )
        self.assertIsInstance(out, UNetSpatioTemporalConditionOutput)
        self.assertEqual(out.sample.shape, (1, 2, 4, 16, 16))

    def test_timestep_0d_tensor(self):
        out = self.model(
            sample=self.sample,
            timestep=torch.tensor(10),
            encoder_hidden_states=self.encoder_hidden_states,
            added_time_ids=self.added_time_ids,
        )
        self.assertIsInstance(out, UNetSpatioTemporalConditionOutput)
        self.assertEqual(out.sample.shape, (1, 2, 4, 16, 16))

    def test_timestep_1d_tensor(self):
        out = self.model(
            sample=self.sample,
            timestep=torch.tensor([10]),
            encoder_hidden_states=self.encoder_hidden_states,
            added_time_ids=self.added_time_ids,
        )
        self.assertIsInstance(out, UNetSpatioTemporalConditionOutput)
        self.assertEqual(out.sample.shape, (1, 2, 4, 16, 16))


if __name__ == "__main__":
    unittest.main()
