from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class StreamingTTSConfig:
    engine: dict[str, Any]
    defaults: dict[str, Any]
    grpc: dict[str, Any]
    web: dict[str, Any]
    models: list[dict[str, Any]]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "StreamingTTSConfig":
        with Path(path).open() as source:
            return cls(**yaml.safe_load(source))

    def to_yaml(self, path: str | Path) -> None:
        with Path(path).open("w") as output:
            yaml.safe_dump(asdict(self), output, sort_keys=False)

    def dumps(self) -> str:
        return yaml.safe_dump(asdict(self), sort_keys=False)
