from __future__ import annotations

import tensorrt as trt

from tinfer.support.observability import get_logger

log = get_logger(__name__)


def _severity_name(severity) -> str:
    name = str(severity)
    if "." in name:
        return name.rsplit(".", 1)[-1].lower()
    return name.lower()


class TensorRTJSONLogger(trt.ILogger):
    def __init__(self, min_severity=trt.Logger.WARNING) -> None:
        super().__init__()
        self.min_severity = min_severity

    def log(self, severity, message: str) -> None:
        if severity > self.min_severity:
            return

        severity_name = _severity_name(severity)
        payload = {
            "severity": severity_name,
            "message": message,
        }
        if severity <= trt.Logger.ERROR:
            log.error("tensorrt_log", **payload)
        elif severity == trt.Logger.WARNING:
            log.warning("tensorrt_log", **payload)
        else:
            log.info("tensorrt_log", **payload)
