from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import yaml

from tinfer.core.request import AlignmentType


@dataclass
class StreamingTTSConfig:

    default_chunk_schedule: list[int] = field(default_factory=lambda: [80, 160, 250, 290])
    default_alignment_type: AlignmentType = AlignmentType.WORD
    default_timeout_ms: float = 80.0
    # default_dtype: str = "bfloat16" # some bugs with bfloat16
    executor_type: Literal["process"] = "process"
    devices: list[str] | None = None
    

    batch_size_per_device: dict[str, int] = field(default_factory=dict)
    default_batch_size: int = 3
    process_workers_per_gpu: int = 1 # TODO: Check if multi gpu really works

    compile_models: bool = True
    runtime_engine: Literal["torch", "tensorrt"] | None = "tensorrt"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StreamingTTSConfig":
        path = Path(path)
        with open(path) as f:
            data = yaml.safe_load(f)
        if data is None:
            return cls()
        if isinstance(data.get("default_alignment_type"), str):
            data["default_alignment_type"] = AlignmentType(data["default_alignment_type"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        config_data = asdict(self)
        config_data["default_alignment_type"] = self.default_alignment_type.value
        with open(path, 'w') as f:
            yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
