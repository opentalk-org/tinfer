from pathlib import Path
import subprocess
import sys

import pytest

from tools.styletts2_model_scripts.convert_model import find_model_files, load_symbols, resolve_config_paths


def test_find_model_files_requires_one_checkpoint_and_one_yaml(tmp_path: Path) -> None:
    (tmp_path / "model.pth").touch()
    (tmp_path / "config.yml").touch()
    assert find_model_files(tmp_path) == (tmp_path / "model.pth", tmp_path / "config.yml")

    (tmp_path / "second.pth").touch()
    with pytest.raises(ValueError, match="exactly one .pth"):
        find_model_files(tmp_path)


def test_load_symbols_validates_runtime_vocabulary(tmp_path: Path) -> None:
    symbols = tmp_path / "symbols.json"
    symbols.write_text('["$", "a", "ą"]', encoding="utf-8")
    assert load_symbols(symbols, 3) == ("$", "a", "ą")

    with pytest.raises(ValueError, match="symbol count"):
        load_symbols(symbols, 4)

    symbols.write_text('["$", "a", "a"]', encoding="utf-8")
    assert load_symbols(symbols, 3) == ("$", "a", "a")


def test_resolve_config_paths_uses_original_model_folder(tmp_path: Path) -> None:
    utility = tmp_path / "Utils"
    utility.mkdir()
    (utility / "bert").mkdir()
    config = {"PLBERT_dir": "Utils/bert", "batch_size": 4}

    resolved = resolve_config_paths(config, tmp_path)

    assert resolved == {"PLBERT_dir": str((utility / "bert").resolve()), "batch_size": 4}


def test_importing_dispatcher_does_not_import_tensorrt() -> None:
    check = "import sys; import tools.styletts2_model_scripts.convert_model; assert 'tensorrt' not in sys.modules"
    subprocess.run([sys.executable, "-c", check], check=True)
