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

        # -----------------------------------------------------------------
        # Shared source of truth for the serving image and the devshell.
        #
        # Everything the workspace needs at build or run time is declared
        # once in this block; the tinfer-server image and `nix develop`
        # consume the exact same lists. Python dependencies come from the
        # same uv.lock in both (the image installs it with uv against the
        # same nix interpreter the devshell exports as UV_PYTHON), so the
        # devshell cannot drift from what ships.
        # -----------------------------------------------------------------
        python = pkgs.python311;
        cudaNvcc = pkgs.cudaPackages.cuda_nvcc;
        gccLib = pkgs.stdenv.cc.cc.lib;

        # Native toolchain for building workspace members (espeak_align is a
        # maturin/Rust extension that links libespeak-ng).
        memberBuildInputs = [
          pkgs.espeak
          pkgs.rustc
          pkgs.cargo
        ];

        # Shared libraries the Python runtime needs on LD_LIBRARY_PATH.
        # zlib: triton's libtriton.so links libz; without it torch.compile
        # can't even import triton and silently falls back to eager.
        runtimeLibs = [
          pkgs.espeak
          pkgs.ffmpeg
          pkgs.gcc
          gccLib
          pkgs.glibc
          pkgs.zlib
        ];

        # Executables the server (torch.compile, triton, audio handling)
        # shells out to at runtime.
        runtimeExecutableDeps = [
          pkgs.ffmpeg
          pkgs.patchelf
          pkgs.gcc
          pkgs.openssl
          cudaNvcc
        ];

        # Directories where NVIDIA driver libraries appear: the /usr/local
        # ones are where the modern nvidia container toolkit injects them,
        # /usr/lib/x86_64-linux-gnu is the stock distro location (what bare
        # hosts and this devshell rely on).
        nvidiaDriverDirs = [
          "/usr/local/nvidia/lib"
          "/usr/local/nvidia/lib64"
          "/usr/lib/x86_64-linux-gnu"
        ];
        nvidiaDriverPath = lib.concatStringsSep ":" nvidiaDriverDirs;

        # Environment shared between the image and the devshell.
        commonEnv = {
          CC = "${pkgs.gcc}/bin/gcc";
          SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
          PYTHONUNBUFFERED = "1";
          TRITON_PTXAS_PATH = "${cudaNvcc}/bin/ptxas";
          TRITON_PTXAS_BLACKWELL_PATH = "${cudaNvcc}/bin/ptxas";
        };

        # Image-only environment: container filesystem layout and the fixed
        # path the nvidia container toolkit mounts the driver at.
        imageEnv =
          commonEnv
          // {
            USER = "root";
            HOME = "/root";
            TORCHINDUCTOR_CACHE_DIR = "/tmp/torchinductor";
            TRITON_LIBCUDA_PATH = "/usr/local/nvidia/lib/libcuda.so";
          };

        cargoBuildEnv = pkgs.runCommand "cargo-build-env" {} ''
          mkdir -p "$out/nix-support"
          cat > "$out/nix-support/setup-hook" <<'EOF'
          export CARGO_HOME="$NIX_BUILD_TOP/cargo-home"
          export CARGO_TARGET_DIR="$NIX_BUILD_TOP/cargo-target"
          mkdir -p "$CARGO_HOME" "$CARGO_TARGET_DIR"
          EOF
        '';
      in {
        packages = rec {
          tinfer-server = x2container.lib.${system}.uv2container.buildImage {
            name = "tinfer";
            src = ./.;
            inherit python runtimeLibs runtimeExecutableDeps;
            extraBuildInputs = [cargoBuildEnv] ++ memberBuildInputs;
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
              Env = lib.mapAttrsToList (k: v: "${k}=${v}") imageEnv;
              Cmd = ["python" "-m" "server.main"];
            };
          };

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

        devShells.default = pkgs.mkShell {
          # Same toolchain the image build uses, plus uv for day-to-day work.
          packages = [pkgs.uv python] ++ memberBuildInputs ++ runtimeExecutableDeps;

          env =
            commonEnv
            // {
              # Pin uv to the interpreter the image ships so .venv is built
              # against the exact same python.
              UV_PYTHON = "${python}/bin/python${python.pythonVersion}";
              UV_PYTHON_PREFERENCE = "only-system";
              UV_PYTHON_DOWNLOADS = "never";
            };

          shellHook = ''
            # Same library path the image gets, minus nix glibc: outside the
            # container the host dynamic loader must keep resolving its own
            # libc, and the driver libs come from the host instead of the
            # nvidia container toolkit mount.
            export LD_LIBRARY_PATH=${lib.makeLibraryPath (lib.remove pkgs.glibc runtimeLibs)}:${nvidiaDriverPath}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
            export LIBRARY_PATH=${lib.makeLibraryPath [gccLib]}:${nvidiaDriverPath}''${LIBRARY_PATH:+:$LIBRARY_PATH}

            # triton dlopens the driver's libcuda.so; point it at wherever
            # this host actually has it (the image pins the container path).
            if [ -z "''${TRITON_LIBCUDA_PATH:-}" ]; then
              for d in ${lib.escapeShellArgs nvidiaDriverDirs}; do
                if [ -e "$d/libcuda.so" ]; then
                  export TRITON_LIBCUDA_PATH="$d/libcuda.so"
                  break
                fi
              done
            fi

            if [ -e .venv/bin/activate ]; then
              . .venv/bin/activate
            else
              echo "tinfer devshell: no .venv yet — run 'uv sync --all-packages' to install the workspace (then re-enter or 'source .venv/bin/activate')."
            fi
          '';
        };
      }
    );
}
