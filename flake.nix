{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    x2container.url = "github:dialohq/x2container.nix/filter-sync";
    x2container.inputs.nixpkgs.follows = "nixpkgs";
    nix2container.follows = "x2container/nix2container";
    crane.url = "github:ipetkov/crane";
  };

  outputs = {
    nixpkgs,
    flake-utils,
    x2container,
    nix2container,
    crane,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {inherit system;};
        gccLib = pkgs.stdenv.cc.cc.lib;

        # Single interpreter shared by the image and the crane-built wheel so
        # the wheel's ABI tag (cpXX) can never drift from the venv it is
        # installed into. Bump this in one place.
        python = pkgs.python311;

        # --- espeak_align (maturin/Rust) wheel, built with crane so the
        # compiled crate dependencies live in their own derivation keyed only
        # on Cargo.{toml,lock}. A change to the Rust *sources* rebuilds just
        # the two workspace crates and reuses the cached dependency layer;
        # today's uv+maturin-from-source path recompiles all crates on every
        # source change.
        craneLib = crane.mkLib pkgs;
        espeakAlignRoot = ./tinfer/espeak_align;

        # maturin is in both the deps and wheel derivations so PATH (and thus
        # pyo3's build-script inputs) match and the pyo3 chain stays cached.
        espeakAlignArgs = {
          strictDeps = true;
          cargoExtraArgs = "-p espeak_align";
          nativeBuildInputs = [python pkgs.maturin];
          buildInputs = [pkgs.espeak];
          RUSTFLAGS = "-L native=${pkgs.espeak}/lib";
          PYO3_PYTHON = "${python}/bin/python3";
          doCheck = false;
        };

        espeakAlignMaturinBuild = ''
          maturin build --release --offline \
            --manifest-path espeak_align/Cargo.toml \
            --target-dir target \
            --auditwheel skip \
            --out dist
        '';

        # Dependency layer: keyed on manifests only, so Rust source edits do
        # not invalidate it.
        espeakAlignDeps = craneLib.buildDepsOnly (espeakAlignArgs
          // {
            src = pkgs.lib.fileset.toSource {
              root = espeakAlignRoot;
              fileset = pkgs.lib.fileset.unions [
                (espeakAlignRoot + "/Cargo.toml")
                (espeakAlignRoot + "/Cargo.lock")
                (espeakAlignRoot + "/pyproject.toml")
                (espeakAlignRoot + "/espeak_align/Cargo.toml")
                (espeakAlignRoot + "/espeak_align/build.rs")
                (espeakAlignRoot + "/espeak_align_core/Cargo.toml")
                (espeakAlignRoot + "/espeak_align_core/build.rs")
              ];
            };
            pname = "espeak_align-deps";
            version = "0.1.0";
            buildPhaseCargoCommand = espeakAlignMaturinBuild;
          });

        espeakAlignWheel = craneLib.mkCargoDerivation (espeakAlignArgs
          // {
            src = espeakAlignRoot;
            cargoArtifacts = espeakAlignDeps;
            pname = "espeak_align-wheel";
            version = "0.1.0";
            doInstallCargoArtifacts = false;
            buildPhaseCargoCommand = espeakAlignMaturinBuild;
            installPhaseCommand = ''
              mkdir -p $out
              cp dist/*.whl $out/
            '';
          });
      in {
        packages = rec {
          tinfer-server = x2container.lib.${system}.uv2container.buildImage {
            name = "tinfer";
            src = ./.;
            inherit python;
            # espeak_align is no longer compiled here — it is installed from a
            # crane-built wheel (see memberWheels below), so the Rust toolchain
            # is not needed in the dependency/member layers.
            extraBuildInputs = [pkgs.espeak];
            dependencyLayers = "autosplit";

            runtimeLibs = [
              pkgs.espeak
              pkgs.ffmpeg
              pkgs.gcc
              gccLib
              pkgs.glibc
            ];
            extraLdLibraryPath = ":/usr/local/nvidia/lib:/usr/local/nvidia/lib64";
            extraLibraryPath = ":/usr/local/nvidia/lib:/usr/local/nvidia/lib64";

            # Dynamic linker problems
            baseImage = {
              imageName = "docker.io/library/ubuntu";
              imageDigest = "sha256:fed6ddb82c61194e1814e93b59cfcb6759e5aa33c4e41bb3782313c2386ed6df";
              arch = "amd64";
              sha256 = "sha256-idRF8oA0N5fuUNN2ch3iA+moDtx0KyP4EDDWHmb2PeY=";
            };
            runtimeExecutableDeps = [pkgs.ffmpeg pkgs.patchelf pkgs.gcc pkgs.openssl];
            members = ["server" "tinfer" "tinfer/espeak_align"];
            # Install espeak_align from its prebuilt (crane) wheel instead of
            # compiling it from source during the image build.
            memberWheels = {"tinfer/espeak_align" = espeakAlignWheel;};
            config = {
              Env = [
                "CC=${pkgs.gcc}/bin/gcc"
                "USER=root"
                "TORCHINDUCTOR_CACHE_DIR=/tmp/torchinductor"
                "PHONEMIZER_ESPEAK_LIBRARY=${pkgs.espeak}/lib/libespeak-ng.so.1"
                "PHONEMIZER_ESPEAK_PATH=${pkgs.espeak}/bin/espeak"
                "PYTHONUNBUFFERED=1"
                "TRITON_LIBCUDA_PATH=/usr/local/nvidia/lib/libcuda.so"
              ];
              Cmd = ["python" "-m" "server.main"];
            };
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
          # Exposed for inspection / cache warming:
          #   espeak-align-deps  — cached Rust dependency layer (manifests-keyed)
          #   espeak-align-wheel — the built extension wheel (source-keyed)
          espeak-align-deps = espeakAlignDeps;
          espeak-align-wheel = espeakAlignWheel;
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
