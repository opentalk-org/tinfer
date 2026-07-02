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
        python = pkgs.python311;

        # The espeak_align extension module as a plain nix-built cdylib — no
        # wheel, no maturin, no rust toolchain inside the uv/image machinery.
        # CPython imports a properly named .so straight off PYTHONPATH, and
        # the nix cc wrapper records an rpath to nix's espeak-ng, so linkage
        # needs no environment variables anywhere.
        espeak-align = pkgs.rustPlatform.buildRustPackage {
          pname = "espeak-align";
          version = "0.1.0";
          src = ./tinfer/espeak_align;
          cargoLock.lockFile = ./tinfer/espeak_align/Cargo.lock;
          buildInputs = [pkgs.espeak-ng];
          nativeBuildInputs = [python];
          env.PYO3_PYTHON = "${python}/bin/python${python.pythonVersion}";
          buildAndTestSubdir = "espeak_align";
          # buildRustPackage installs binaries; we want the cdylib as an
          # importable extension module in site-packages layout.
          postInstall = ''
            site="$out/lib/python${python.pythonVersion}/site-packages"
            mkdir -p "$site"
            cp target/*/release/libespeak_align.so "$site/espeak_align.so" 2>/dev/null \
              || cp target/release/libespeak_align.so "$site/espeak_align.so"
          '';
          doCheck = false;
        };
      in {
        packages = rec {
          inherit espeak-align;

          tinfer-server = x2container.lib.${system}.uv2container.buildImage {
            name = "tinfer";
            src = ./.;
            inherit python;
            imageCheck = ["python" "-m" "server.main" "--smoke-test"];
            imageCheckEnv.TINFER_SMOKE_TEST_CPU_OK = "1";
            # Serving only deserializes engines (built by the trtc pipeline);
            # the tensorrt wheel's engine-builder payload — including Windows
            # binaries — is 5.6GB of dead weight.
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
            # /usr/lib/x86_64-linux-gnu is where the modern nvidia container
            # toolkit injects the driver libraries.
            extraLdLibraryPath = ":/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu";
            extraLibraryPath = ":/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu";
            runtimeExecutableDeps = [pkgs.ffmpeg pkgs.patchelf pkgs.gcc pkgs.openssl];
            members = ["server" "tinfer"];
            pythonLibs = [espeak-align];
            config = {
              Env = [
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
            # espeak_align is nix-built (packages.espeak-align), not a wheel:
            # expose it to the uv venv via PYTHONPATH. After editing the rust
            # code, `direnv reload` (or re-entering the shell) rebuilds it.
            export PYTHONPATH=${espeak-align}/lib/python${python.pythonVersion}/site-packages''${PYTHONPATH:+:$PYTHONPATH}
          '';
        };
      }
    );
}
