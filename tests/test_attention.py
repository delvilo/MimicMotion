import torch
import pytest
from mimicmotion.modules.attention import TransformerSpatioTemporalModel


def test_transformer_spatio_temporal_model_out_channels_default():
    in_channels = 32
    model = TransformerSpatioTemporalModel(
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=in_channels,
        out_channels=None,
        num_layers=1,
    )
    assert model.out_channels == in_channels
    assert model.proj_out.out_features == in_channels


def test_transformer_spatio_temporal_model_out_channels_specified():
    in_channels = 32
    out_channels = 16
    model = TransformerSpatioTemporalModel(
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=in_channels,
        out_channels=out_channels,
        num_layers=1,
    )
    assert model.out_channels == out_channels
    assert model.proj_out.out_features == out_channels


def test_transformer_spatio_temporal_model_forward():
    in_channels = 32
    cross_attention_dim = 32
    num_frames = 2
    batch_size = 1
    batch_frames = batch_size * num_frames
    height, width = 8, 8

    model = TransformerSpatioTemporalModel(
        num_attention_heads=2,
        attention_head_dim=16,
        in_channels=in_channels,
        out_channels=None,
        num_layers=1,
        cross_attention_dim=cross_attention_dim,
    )

    hidden_states = torch.randn(batch_frames, in_channels, height, width)
    encoder_hidden_states = torch.randn(batch_size * num_frames, 1, cross_attention_dim)
    image_only_indicator = torch.zeros(batch_size, num_frames)

    output = model(
        hidden_states=hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        image_only_indicator=image_only_indicator,
        return_dict=True,
    )

    assert output.sample.shape == (batch_frames, in_channels, height, width)
