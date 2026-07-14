from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StreamingTTSConfig:
    models: list[dict[str, Any]] = field(default_factory=list)
    chunk_length_schedule: list[int] = field(default_factory=lambda: [80, 160, 250, 290])
    timeout_ms: int = 80
    queue_capacity: int = 64
    grpc_address: str = "127.0.0.1:50051"
    http_address: str = "127.0.0.1:8000"

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StreamingTTSConfig":
        with Path(path).open() as source:
            return cls(**yaml.safe_load(source))

    def to_yaml(self, path: str | Path) -> None:
        with Path(path).open("w") as output:
            yaml.safe_dump(asdict(self), output, sort_keys=False)

    def dumps(self) -> str:
        return yaml.safe_dump(asdict(self), sort_keys=False)
