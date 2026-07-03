"""Runtime environment smoke test for the tinfer image.

Runs every dependency the server needs end-to-end (imports, tmp, ffmpeg,
espeak phonemization, CUDA + a real GPU op) without loading any model.
Exits non-zero if anything fails.

    python -m server.main --smoke-test  # CUDA required
    TINFER_SMOKE_TEST_CPU_OK=1 \
    python -m server.main --smoke-test  # for hosts without a GPU
"""

import importlib
import os
import shutil
import subprocess
import sys
import tempfile

_failures: list[str] = []


def check(name: str, fn):
    try:
        detail = fn()
        print(f"[smoke-test] PASS {name}" + (f": {detail}" if detail else ""))
    except Exception as e:
        print(f"[smoke-test] FAIL {name}: {e!r}")
        _failures.append(name)


def _imports():
    modules = [
        "numpy",
        "yaml",
        "aiohttp",
        "grpc",
        "librosa",
        "transformers",
        "nltk",
        "einops",
        "torch",
        "torchaudio",
        # torch.compile falls back to eager silently if triton can't load
        # (e.g. a missing native lib); fail loudly here instead.
        "triton",
        "espeak_align",
        "tinfer.core.engine",
        "tinfer.server.websocket",
        "tinfer.server.grpc.server",
        "server.main",
    ]
    for m in modules:
        importlib.import_module(m)
    return f"{len(modules)} modules"


def _tmp():
    with tempfile.NamedTemporaryFile() as f:
        f.write(b"ok")
    return tempfile.gettempdir()


def _home():
    home = os.path.expanduser("~")
    probe = os.path.join(home, ".cache", "tinfer-smoke-test")
    os.makedirs(probe, exist_ok=True)
    os.rmdir(probe)
    return home


def _ffmpeg():
    path = shutil.which("ffmpeg")
    if not path:
        raise RuntimeError("ffmpeg not on PATH")
    out = subprocess.run([path, "-version"], capture_output=True, text=True, check=True)
    return out.stdout.splitlines()[0]


def _espeak():
    import espeak_align

    engine = espeak_align.Engine("en-us", tie=True, espeak_workers=1)
    phonemes = engine.text_to_phonemes("hello world")
    if not phonemes.strip():
        raise RuntimeError("empty phonemization")
    return phonemes.strip()


def _cuda():
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"torch.cuda.is_available() is False (torch {torch.__version__}, built for CUDA {torch.version.cuda})"
        )
    props = torch.cuda.get_device_properties(0)
    a = torch.randn(512, 512)
    expected = a @ a
    gpu = (a.cuda() @ a.cuda()).cpu()
    if not torch.allclose(expected, gpu, atol=1e-2):
        raise RuntimeError("GPU matmul result diverges from CPU")
    half = (a.cuda().half() @ a.cuda().half()).float().cpu()
    if not torch.allclose(expected, half, atol=2.0, rtol=0.1):
        raise RuntimeError("fp16 GPU matmul result diverges")
    import tensorrt as trt

    runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
    if runtime is None:
        raise RuntimeError("could not create TensorRT runtime")
    return (
        f"{props.name}, capability {props.major}.{props.minor}, "
        f"{props.total_memory // 2**20} MiB, driver-visible devices {torch.cuda.device_count()}, "
        f"torch {torch.__version__} (CUDA {torch.version.cuda}, cuDNN {torch.backends.cudnn.version()}), "
        f"TensorRT {trt.__version__}"
    )


def main() -> int:
    check("imports", _imports)
    check("tmp", _tmp)
    check("home", _home)
    check("ffmpeg", _ffmpeg)
    check("espeak", _espeak)
    if os.environ.get("TINFER_SMOKE_TEST_CPU_OK"):
        print("[smoke-test] SKIP cuda (TINFER_SMOKE_TEST_CPU_OK set)")
    else:
        check("cuda", _cuda)
    if _failures:
        print(f"[smoke-test] RESULT: FAIL ({', '.join(_failures)})")
        return 1
    print("[smoke-test] RESULT: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
