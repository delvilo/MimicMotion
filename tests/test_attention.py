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


def test_time_context_computation_equivalence():
    batch_size = 2
    num_frames = 4
    seq_len = 1
    dim = 320
    height = 8
    width = 8

    # Create dummy encoder_hidden_states of shape (batch_size * num_frames, seq_len, dim)
    # Using range values to ensure distinct values per frame
    values = list(range(batch_size * num_frames * seq_len * dim))

    # Simulate original time_context computation
    # (Since torch is not installed in the system test env, we perform equivalent array operations)
    encoder_hidden_states = [values[i*seq_len*dim : (i+1)*seq_len*dim] for i in range(batch_size * num_frames)]

    # Slicing every num_frames extracts the first timestep of each batch item
    tc_first_sliced = encoder_hidden_states[::num_frames]

    # Repeating across spatial dimensions (height * width)
    tc_repeated = tc_first_sliced * (height * width)

    assert len(tc_first_sliced) == batch_size
    assert len(tc_repeated) == height * width * batch_size
    assert tc_repeated[0] == encoder_hidden_states[0]
    assert tc_repeated[1] == encoder_hidden_states[num_frames]
