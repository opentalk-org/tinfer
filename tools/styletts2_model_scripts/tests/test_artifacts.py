from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np
import pytest

from tools.styletts2_model_scripts.artifacts import (
    architecture_id,
    stage_output,
    write_manifest,
    write_tinf,
)


@dataclass(frozen=True)
class ModelTopology:
    hidden: int
    layers: int


def test_write_tinf_uses_native_little_endian_contract(tmp_path: Path) -> None:
    output = tmp_path / "voice.tinf"
    write_tinf(output, {"ref_s": np.array([1.5, -2.0], dtype=np.float32)})

    expected = b"".join(
        [
            b"TINF",
            struct.pack("<i", 1),
            struct.pack("<i", 5),
            b"ref_s",
            struct.pack("<iiq", 1, 1, 2),
            struct.pack("<ff", 1.5, -2.0),
        ]
    )
    assert output.read_bytes() == expected


def test_architecture_id_covers_topology_parameters_and_profile() -> None:
    parameters = (("decoder.weight", (8, 4)),)
    first = architecture_id(ModelTopology(512, 3), parameters, 16, 512, 1200, 5)

    assert first == architecture_id(ModelTopology(512, 3), parameters, 16, 512, 1200, 5)
    assert first != architecture_id(ModelTopology(256, 3), parameters, 16, 512, 1200, 5)
    assert first != architecture_id(ModelTopology(512, 3), (("decoder.weight", (9, 4)),), 16, 512, 1200, 5)
    assert first != architecture_id(ModelTopology(512, 3), parameters, 8, 512, 1200, 5)


def test_staged_target_publishes_only_successful_output(tmp_path: Path) -> None:
    destination = tmp_path / "tensorrt"
    with pytest.raises(RuntimeError, match="failed"):
        with stage_output(destination, force=False) as staging:
            (staging / "A.engine").write_bytes(b"partial")
            raise RuntimeError("failed")
    assert not destination.exists()

    with stage_output(destination, force=False) as staging:
        (staging / "A.engine").write_bytes(b"complete")
    assert (tmp_path / "tensorrt/A.engine").read_bytes() == b"complete"

    with pytest.raises(FileExistsError):
        with stage_output(destination, force=False):
            pass


def test_manifest_rejects_incompatible_existing_output(tmp_path: Path) -> None:
    write_manifest(tmp_path, "styletts2-a", "en", ("en",), ("$", "a"))
    write_manifest(tmp_path, "styletts2-a", "en", ("en",), ("$", "a"))

    with pytest.raises(ValueError, match="incompatible"):
        write_manifest(tmp_path, "styletts2-b", "en", ("en",), ("$", "a"))
