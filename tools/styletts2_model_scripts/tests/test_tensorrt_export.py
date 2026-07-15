from tools.styletts2_model_scripts.tensorrt_export import profile_shapes


def test_profiles_cover_dynamic_runtime_inputs_and_fixed_weights() -> None:
    assert profile_shapes("A", "tokens", (-1, -1), 64, 512, 5) == ((1, 8), (8, 128), (64, 512))
    assert profile_shapes("BC", "en", (-1, 512, 176), 16, 512, 5) == (
        (1, 512, 176),
        (8, 512, 176),
        (16, 512, 176),
    )
    assert profile_shapes("BC", "source_noise", (-1, 105600, 9), 64, 512, 5) == (
        (1, 105600, 9),
        (8, 105600, 9),
        (64, 105600, 9),
    )
    assert profile_shapes("BC", "decoder.weight", (512, 256, 3), 16, 512, 5) == (
        (512, 256, 3),
        (512, 256, 3),
        (512, 256, 3),
    )
