# Devshell built from the image's exported runtime contract.
{
  pkgs,
  tinfer-server,
}: let
  lib = pkgs.lib;
  rt = tinfer-server.runtime;
in
  pkgs.mkShell {
    packages = [pkgs.uv rt.python] ++ rt.memberBuildInputs ++ rt.runtimeExecutableDeps;

    env =
      rt.env
      // {
        # Build .venv against the exact interpreter the image ships.
        UV_PYTHON = "${rt.python}/bin/python${rt.python.pythonVersion}";
        UV_PYTHON_PREFERENCE = "only-system";
        UV_PYTHON_DOWNLOADS = "never";
        # Native deps of the manylinux wheels are preloaded by absolute
        # path at interpreter startup (same pattern as torch's
        # _load_global_deps) — scoped to this project's python processes
        # instead of a shell-wide LD_LIBRARY_PATH.
        PYTHONPATH = pkgs.writeTextDir "sitecustomize.py" ''
          import ctypes, os
          for _p in [
              "${pkgs.stdenv.cc.cc.lib}/lib/libstdc++.so.6",
              "${pkgs.zlib}/lib/libz.so.1",
              "${pkgs.espeak}/lib/libespeak-ng.so.1",
          ]:
              ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
          for _d in [${lib.concatMapStringsSep ", " (d: ''"${d}"'') rt.nvidiaDriverDirs}]:
              _p = os.path.join(_d, "libcuda.so.1")
              if os.path.exists(_p):
                  ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
                  # dir containing libcuda, for triton's -lcuda link
                  os.environ.setdefault("TRITON_LIBCUDA_PATH", _d)
                  break
        '';
      };

    shellHook = ''
      if [ -e .venv/bin/activate ]; then
        . .venv/bin/activate
      else
        echo "no .venv yet — run: uv sync --all-packages"
      fi
    '';
  }
