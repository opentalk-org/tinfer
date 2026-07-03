"""Preload nix-provided native deps of the manylinux wheels.

The devshell puts this module on PYTHONPATH next to a preload.json
holding {"preload_libs": [...], "nvidia_driver_dirs": [...]} (see
devshell.nix); without that file the module is a no-op. Same idea as
torch's _load_global_deps — libraries are loaded by absolute path and
later NEEDED entries resolve by SONAME from the link map — but lazy:
nothing is mapped until the first C-extension import, so pure-python
processes stay untouched.
"""

import json
import os
import sys

_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "preload.json")


def _install(preload_libs, nvidia_driver_dirs):
    import ctypes
    from importlib.machinery import ExtensionFileLoader, PathFinder

    def preload():
        for path in preload_libs:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        for dirname in nvidia_driver_dirs:
            if os.path.exists(os.path.join(dirname, "libcuda.so.1")):
                ctypes.CDLL(os.path.join(dirname, "libcuda.so.1"), mode=ctypes.RTLD_GLOBAL)
                # dir containing libcuda, for triton's -lcuda link
                os.environ.setdefault("TRITON_LIBCUDA_PATH", dirname)
                break

    class PreloadOnFirstExtension:
        invalidate_caches = staticmethod(PathFinder.invalidate_caches)
        _triggered = False

        def find_spec(self, fullname, path=None, target=None):
            spec = PathFinder.find_spec(fullname, path, target)
            if spec is not None and isinstance(spec.loader, ExtensionFileLoader) and not self._triggered:
                # dlopen refcounts, so a concurrent duplicate preload is harmless
                self._triggered = True
                preload()
                try:
                    sys.meta_path.remove(self)
                except ValueError:
                    pass
            return spec

    if PathFinder in sys.meta_path:
        sys.meta_path.insert(sys.meta_path.index(PathFinder), PreloadOnFirstExtension())


if sys.platform == "linux" and os.path.exists(_CONFIG):
    with open(_CONFIG) as _f:
        _config = json.load(_f)
    _install(_config["preload_libs"], _config["nvidia_driver_dirs"])
