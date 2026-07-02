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
        pkgs = import nixpkgs {inherit system;};
        gccLib = pkgs.stdenv.cc.cc.lib;
        # The one GPU-container contract for every GPU image this flake
        # produces. libcuda/libnvidia-ml always come from the HOST driver,
        # injected by the container runtime; the image only carries CUDA
        # runtime libs. Nix binaries use the nix loader, which never reads
        # the container's /etc/ld.so.cache, so the injection locations must
        # be on LD_LIBRARY_PATH explicitly (nix libs first, so they win):
        # /usr/local/nvidia is the legacy mount, /usr/lib/x86_64-linux-gnu is
        # where nvidia-container-toolkit actually places driver libs. The
        # NVIDIA_* markers tell the runtime to inject when the host default
        # doesn't already.
        gpuContainer = {
          driverLibraryPath = ":/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu";
          env = [
            "NVIDIA_VISIBLE_DEVICES=all"
            "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
          ];
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
            python = pkgs.python311;
            extraBuildInputs = [
              cargoBuildEnv
              pkgs.espeak
              pkgs.rustc
              pkgs.cargo
            ];
            imageCheck = ["python" "-m" "server.main" "--smoke-test"];
            imageCheckEnv.TINFER_SMOKE_TEST_CPU_OK = "1";
            # Serving only deserializes engines (built by the trtc pipeline);
            # the tensorrt wheel's engine-builder payload — including Windows
            # binaries — is 5.6GB of dead weight. The trtc-builder image below
            # deliberately does NOT prune this: it is the thing that builds.
            prunePackageFiles."tensorrt-cu12-libs" = [
              "libnvinfer_builder_resource*"
              "*_win_*"
            ];

            runtimeLibs = [
              pkgs.espeak
              pkgs.ffmpeg
              pkgs.gcc
              gccLib
              pkgs.glibc
            ];
            extraLdLibraryPath = gpuContainer.driverLibraryPath;
            extraLibraryPath = gpuContainer.driverLibraryPath;
            runtimeExecutableDeps = [pkgs.ffmpeg pkgs.patchelf pkgs.gcc pkgs.openssl];
            members = ["server" "tinfer" "tinfer/espeak_align" "trtc"];
            config = {
              Env =
                gpuContainer.env
                ++ [
                  "CC=${pkgs.gcc}/bin/gcc"
                  "USER=root"
                  "HOME=/root"
                  "SSL_CERT_FILE=${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt"
                  "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor"
                  "PYTHONUNBUFFERED=1"
                  "TRITON_LIBCUDA_PATH=/usr/local/nvidia/lib/libcuda.so"
                ];
              Cmd = ["python" "-m" "server.main"];
            };
          };

          # The remote builder: a fixed, correct environment — trtc plus the
          # TensorRT the workspace lock pins (via the trtc-builder member's
          # trtc[builder] dependency). Like a nix derivation, it is pinned to
          # one TensorRT version; a plan pinning a different version fails the
          # job. Run: docker run --gpus all -p 8080:8080 ...
          trtc-builder = x2container.lib.${system}.uv2container.buildImage {
            name = "trtc-builder";
            src = ./.;
            python = pkgs.python311;
            members = ["trtc" "trtc-builder"];
            # Only libstdc++ for the manylinux TRT libs; deliberately NOT nix
            # glibc — host-injected FHS binaries (nvidia-smi) must not resolve
            # a foreign libc ahead of their own.
            runtimeLibs = [gccLib];
            extraLdLibraryPath = gpuContainer.driverLibraryPath;
            extraLibraryPath = gpuContainer.driverLibraryPath;
            config = {
              Env =
                gpuContainer.env
                ++ [
                  "USER=root"
                  "PYTHONUNBUFFERED=1"
                  "TRTC_DATA_DIR=/data"
                ];
              Cmd = ["python" "-m" "trtc.cli" "serve" "--host" "0.0.0.0" "--port" "8080"];
            };
          };

          # Prebuilt env for `trtc launch` — trtc resolved once at build time,
          # vastai from the vast-cli flake input (no runtime installs). The
          # launch logic is pure Python in trtc/vast.py.
          launch-env = pkgs.stdenv.mkDerivation {
            name = "trtc-launch-env";
            src = ./trtc;
            __noChroot = true;
            dontFixup = true;
            NIX_SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
            SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
            nativeBuildInputs = [pkgs.python311 pkgs.uv];
            buildPhase = ''
              runHook preBuild
              export HOME="$TMPDIR" UV_CACHE_DIR="$TMPDIR/uv"
              export UV_PYTHON_PREFERENCE=only-system UV_PYTHON=${pkgs.python311}/bin/python3.11
              cp -r "$src" ./trtc-src && chmod -R u+w ./trtc-src
              uv venv "$out"
              uv pip install --python "$out/bin/python" ./trtc-src
              runHook postBuild
            '';
            dontInstall = true;
          };

          # Rents a vast.ai GPU and starts the builder image matching this
          # repo's locked TensorRT; prints `export TRTC_BUILDER=...`.
          #   nix run .#launch-builder -- [--trt-version 10.13] [--token TOK]
          #     [--idle-timeout SECS] [--gpu RTX_4090] [--image REF] ...
          # Needs a vast.ai key (VAST_API_KEY or a configured vastai CLI).
          launch-builder = pkgs.writeShellApplication {
            name = "launch-builder";
            runtimeInputs = [vast-cli.packages.${system}.default];
            text = ''exec ${launch-env}/bin/trtc launch "$@"'';
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
          packages = [pkgs.espeak pkgs.uv];
          shellHook = ''
            export LD_LIBRARY_PATH=${pkgs.espeak}/lib:$LD_LIBRARY_PATH
          '';
        };
      }
    );
}
