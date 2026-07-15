from tools.styletts2_model_scripts.tensorrt_export import profile_shapes


def test_profiles_cover_dynamic_runtime_inputs_and_fixed_weights() -> None:
    assert profile_shapes("A", "tokens", (-1, -1), 16, 512, 5) == ((1, 8), (1, 32), (16, 512))
    assert profile_shapes("B", "en", (-1, 512, 128), 16, 512, 5) == (
        (1, 512, 128),
        (1, 512, 128),
        (16, 512, 128),
    )
    assert profile_shapes("C", "har", (-1, 22, 15361), 16, 512, 5) == (
        (1, 22, 15361),
        (1, 22, 15361),
        (16, 22, 15361),
    )
    assert profile_shapes("C", "decoder.weight", (512, 256, 3), 16, 512, 5) == (
        (512, 256, 3),
        (512, 256, 3),
        (512, 256, 3),
    )
