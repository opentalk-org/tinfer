"""Preload nix-provided native deps of the manylinux wheels.

Installed on PYTHONPATH by the devshell (devshell.nix substitutes the
@-placeholders). Same idea as torch's _load_global_deps — libraries are
loaded by absolute path and later NEEDED entries resolve by SONAME from
the link map — but lazy: nothing is mapped until the first C-extension
import, so pure-python processes stay untouched.
"""

import sys

_PRELOAD_LIBS = @preloadLibs@
_NVIDIA_DRIVER_DIRS = @nvidiaDriverDirs@


def _install():
    import ctypes
    import os
    from importlib.machinery import ExtensionFileLoader, PathFinder

    def preload():
        for path in _PRELOAD_LIBS:
            ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        for dirname in _NVIDIA_DRIVER_DIRS:
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


if sys.platform == "linux":
    _install()
