{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    x2container.url = "github:dialohq/x2container.nix/filter-sync";
    x2container.inputs.nixpkgs.follows = "nixpkgs";
    nix2container.follows = "x2container/nix2container";
    vast-cli.url = "github:dialohq/vast-cli.nix";
    vast-cli.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = {
    nixpkgs,
    flake-utils,
    x2container,
    nix2container,
    vast-cli,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };
        lib = pkgs.lib;

        # Shared between the serving image and the devshell so the two can't
        # drift; Python deps come from the same uv.lock in both.
        python = pkgs.python311;
        cudaNvcc = pkgs.cudaPackages.cuda_nvcc;
        gccLib = pkgs.stdenv.cc.cc.lib;

        # Toolchain for building workspace members (espeak_align links
        # libespeak-ng).
        memberBuildInputs = [
          pkgs.espeak
          pkgs.rustc
          pkgs.cargo
        ];

        # Runtime LD_LIBRARY_PATH. zlib: libtriton.so links libz; without it
        # torch.compile silently falls back to eager.
        runtimeLibs = [
          pkgs.espeak
          pkgs.ffmpeg
          pkgs.gcc
          gccLib
          pkgs.glibc
          pkgs.zlib
        ];

        # Executables needed at runtime (torch.compile, triton, audio).
        runtimeExecutableDeps = [
          pkgs.ffmpeg
          pkgs.patchelf
          pkgs.gcc
          pkgs.openssl
          cudaNvcc
        ];

        # Driver locations: nvidia container toolkit mounts, then the stock
        # distro path bare hosts have.
        nvidiaDriverDirs = [
          "/usr/local/nvidia/lib"
          "/usr/local/nvidia/lib64"
          "/usr/lib/x86_64-linux-gnu"
        ];
        nvidiaDriverPath = lib.concatStringsSep ":" nvidiaDriverDirs;

        commonEnv = {
          CC = "${pkgs.gcc}/bin/gcc";
          SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
          PYTHONUNBUFFERED = "1";
          TRITON_PTXAS_PATH = "${cudaNvcc}/bin/ptxas";
          TRITON_PTXAS_BLACKWELL_PATH = "${cudaNvcc}/bin/ptxas";
        };

        tinfer-server =
          (x2container.lib.${system}.uv2container.buildImage {
            name = "tinfer";
            src = ./.;
            inherit python runtimeLibs runtimeExecutableDeps;
            extraBuildInputs =
              [
                (pkgs.runCommand "cargo-build-env" {} ''
                  mkdir -p "$out/nix-support"
                  cat > "$out/nix-support/setup-hook" <<'EOF'
                  export CARGO_HOME="$NIX_BUILD_TOP/cargo-home"
                  export CARGO_TARGET_DIR="$NIX_BUILD_TOP/cargo-target"
                  mkdir -p "$CARGO_HOME" "$CARGO_TARGET_DIR"
                  EOF
                '')
              ]
              ++ memberBuildInputs;
            imageCheck = ["python" "-m" "server.main" "--smoke-test"];
            imageCheckEnv.TINFER_SMOKE_TEST_CPU_OK = "1";
            # Serving only deserializes engines (built by the trtc pipeline);
            # the tensorrt wheel's engine-builder payload — including Windows
            # binaries — is 5.6GB of dead weight.
            prunePackageFiles."tensorrt-cu12-libs" = [
              "libnvinfer_builder_resource*"
              "*_win_*"
            ];

            extraLdLibraryPath = ":" + nvidiaDriverPath;
            extraLibraryPath = ":" + nvidiaDriverPath;
            members = ["server" "tinfer" "tinfer/espeak_align"];
            config = {
              Env = lib.mapAttrsToList (k: v: "${k}=${v}") (commonEnv
                // {
                  USER = "root";
                  HOME = "/root";
                  TORCHINDUCTOR_CACHE_DIR = "/tmp/torchinductor";
                  # A directory: triton asserts $TRITON_LIBCUDA_PATH/libcuda.so.1
                  # exists (the previous file-path value could never pass that).
                  TRITON_LIBCUDA_PATH = "/usr/local/nvidia/lib";
                });
              Cmd = ["python" "-m" "server.main"];
            };
          })
          # The image exports its runtime contract; the devshell consumes
          # only these attributes, so it cannot drift from the image.
          .overrideAttrs (old: {
            passthru =
              (old.passthru or {})
              // {
                runtime = {
                  inherit python memberBuildInputs runtimeLibs runtimeExecutableDeps nvidiaDriverDirs nvidiaDriverPath;
                  env = commonEnv;
                };
              };
          });
      in {
        packages = rec {
          inherit tinfer-server;

          vast-smoke-test = pkgs.writeShellApplication {
            name = "vast-smoke-test";
            runtimeInputs = [
              vast-cli.packages.${system}.default
              pkgs.jq
              pkgs.coreutils
              pkgs.gnugrep
            ];
            text = builtins.readFile ./tools/vast-smoke-test.sh;
          };

          copy-platform-image = pkgs.writeShellScriptBin "copy-platform-image" ''
            if [ $# -ne 3 ]; then
              echo "Usage: $0 <platform> <image> <tag>"
            fi

            platform="$1"
            image="$2"
            tag="$3"
            nix build ".#packages.$platform.images.$image"
            echo "Copying to docker daemon: $image:$tag"
            ${nix2container.packages.${system}.skopeo-nix2container}/bin/skopeo copy nix:result "docker-daemon:$image:$tag"
          '';

          push-platform-image = pkgs.writeShellScriptBin "copy-platform-image" ''
            set -euo pipefail

            if [ "$#" -ne 3 ]; then
              echo "Usage: $0 <platform> <image> <tag>"
              exit 1
            fi

            platform="$1"
            image="$2"
            tag="$3"

            nix build ".#packages.$platform.$image"

            target="docker://docker.io/plan9better/$image:$tag"
            echo "Pushing image to $target"
            policy_file="$(mktemp)"
            cat > "$policy_file" <<'EOF'
            {
              "default": [
                { "type": "insecureAcceptAnything" }
              ]
            }
            EOF
            ${nix2container.packages.${system}.skopeo-nix2container}/bin/skopeo copy \
              --policy "$policy_file" \
              --dest-tls-verify=false \
              nix:result "$target"
            rm -f "$policy_file"
          '';
          default = tinfer-server;
        };

        devShells.default = let
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
                # Native deps of the manylinux wheels are preloaded by
                # absolute path at interpreter startup (same pattern as
                # torch's _load_global_deps) — scoped to this project's
                # python processes instead of a shell-wide LD_LIBRARY_PATH.
                PYTHONPATH = pkgs.writeTextDir "sitecustomize.py" ''
                  import ctypes, os
                  for _p in [
                      "${gccLib}/lib/libstdc++.so.6",
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
          };
      }
    );
}
