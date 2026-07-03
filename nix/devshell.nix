# Devshell built from the image's exported runtime contract.
{
  pkgs,
  tinfer-server,
}: let
  lib = pkgs.lib;
  rt = tinfer-server.runtime;

  # Preload the wheels' native deps by absolute path (torch's
  # _load_global_deps pattern): scoped to this project's python processes,
  # no shell-wide LD_LIBRARY_PATH. Lazy: nothing is mapped until the first
  # C-extension import, so pure-python processes stay untouched.
  sitecustomize = pkgs.writeTextDir "sitecustomize.py" ''
    import sys

    if sys.platform == "linux":
        import ctypes, os
        from importlib.machinery import ExtensionFileLoader, PathFinder

        def _preload():
            for _p in [${lib.concatMapStringsSep ", " (p: ''"${p}"'') rt.preloadLibs}]:
                ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
            for _d in [${lib.concatMapStringsSep ", " (d: ''"${d}"'') rt.nvidiaDriverDirs}]:
                if os.path.exists(os.path.join(_d, "libcuda.so.1")):
                    ctypes.CDLL(os.path.join(_d, "libcuda.so.1"), mode=ctypes.RTLD_GLOBAL)
                    # dir containing libcuda, for triton's -lcuda link
                    os.environ.setdefault("TRITON_LIBCUDA_PATH", _d)
                    break

        class _PreloadOnFirstExtension:
            invalidate_caches = staticmethod(PathFinder.invalidate_caches)

            def find_spec(self, fullname, path=None, target=None):
                spec = PathFinder.find_spec(fullname, path, target)
                if spec is not None and isinstance(spec.loader, ExtensionFileLoader):
                    sys.meta_path.remove(self)
                    _preload()
                return spec

        sys.meta_path.insert(sys.meta_path.index(PathFinder), _PreloadOnFirstExtension())
  '';

  devPlatform =
    if pkgs.stdenv.isDarwin
    then "darwin"
    else "linux";
in
  pkgs.mkShell {
    packages = [pkgs.uv rt.python] ++ rt.crateDevTools ++ rt.runtimeExecutableDeps;

    env =
      rt.env
      // {
        # Build .venv against the exact interpreter the image ships.
        UV_PYTHON = "${rt.python}/bin/python${rt.python.pythonVersion}";
        UV_PYTHON_PREFERENCE = "only-system";
        UV_PYTHON_DOWNLOADS = "never";
        PYO3_PYTHON = "${rt.python}/bin/python${rt.python.pythonVersion}";
      };

    shellHook = ''
      # Editable espeak_align: dev/<platform> holds a committed symlink onto
      # cargo's debug output; while it dangles, imports fall through to the
      # nix-built module.
      export PYTHONPATH="$PWD/tinfer/espeak_align/dev/${devPlatform}:${rt.espeakAlignSite}:${sitecustomize}''${PYTHONPATH:+:$PYTHONPATH}"

      if [ -e .venv/bin/activate ]; then
        . .venv/bin/activate
      else
        echo "no .venv yet — run: uv sync --all-packages"
      fi
    '';
  }
