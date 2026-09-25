from mimicmotion.modules.attention import TransformerSpatioTemporalModel


def test_transformer_spatio_temporal_model_out_channels_default():
    model = TransformerSpatioTemporalModel(
        num_attention_heads=4,
        attention_head_dim=16,
        in_channels=32,
        out_channels=None,
    )
    assert model.out_channels == 32
    assert model.proj_out.out_features == 32


def test_transformer_spatio_temporal_model_out_channels_custom():
    model = TransformerSpatioTemporalModel(
        num_attention_heads=4,
        attention_head_dim=16,
        in_channels=32,
        out_channels=64,
    )
    assert model.out_channels == 64
    assert model.proj_out.out_features == 64
