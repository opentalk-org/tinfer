"""Preload nix-provided native deps of the manylinux wheels.

The devshell puts this module on PYTHONPATH next to a preload.json
holding {"preload_libs": [...], "nvidia_driver_dirs": [...]} (see
devshell.nix); without that file the module is a no-op. Same pattern as
torch's _load_global_deps: libraries are loaded by absolute path, and
later NEEDED entries resolve by SONAME from the link map — so wheels
work without LD_LIBRARY_PATH.
"""

import json
import os
import sys

_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preload.json")

if sys.platform == "linux" and os.path.exists(_CONFIG):
    import ctypes

    with open(_CONFIG) as _f:
        _config = json.load(_f)
    for _path in _config["preload_libs"]:
        ctypes.CDLL(_path, mode=ctypes.RTLD_GLOBAL)
    for _dirname in _config["nvidia_driver_dirs"]:
        if os.path.exists(os.path.join(_dirname, "libcuda.so.1")):
            ctypes.CDLL(os.path.join(_dirname, "libcuda.so.1"), mode=ctypes.RTLD_GLOBAL)
            # dir containing libcuda, for triton's -lcuda link
            os.environ.setdefault("TRITON_LIBCUDA_PATH", _dirname)
            break
