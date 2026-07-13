from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_min_chunk_schedule_is_removed_from_runtime_api_sources():
    runtime_sources = [
        ROOT / "tinfer/tinfer/core/request.py",
        ROOT / "tinfer/tinfer/core/engine.py",
        ROOT / "tinfer/tinfer/config/engine_config.py",
        ROOT / "tinfer/tinfer/utils/text_chunker.py",
    ]

    for source in runtime_sources:
        contents = source.read_text()
        assert "min_chunk_length_schedule" not in contents
        assert "default_min_chunk_schedule" not in contents
