# Devshell built from the image's exported runtime contract.
{
  pkgs,
  tinfer-server,
}: let
  lib = pkgs.lib;
  rt = tinfer-server.runtime;

  sitecustomize = pkgs.linkFarm "tinfer-sitecustomize" [
    {
      name = "sitecustomize.py";
      path = ./sitecustomize.py;
    }
    {
      name = "preload.json";
      path = pkgs.writeText "preload.json" (builtins.toJSON {
        preload_libs = rt.preloadLibs;
        nvidia_driver_dirs = rt.nvidiaDriverDirs;
      });
    }
  ];

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
