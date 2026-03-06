{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
    x2container.url = "git+file:///workspace/x2container.nix";
    nix2container.follows = "x2container/nix2container";
  };

  outputs = {
    nixpkgs,
    flake-utils,
    x2container,
    nix2container,
    ...
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        pkgs = import nixpkgs {inherit system;};
      in {
        packages = rec {
          tinfer-server = x2container.lib.${system}.uv2container.buildImage {
            name = "tinfer";
            src = ./.;
            python = pkgs.python311;
            extraBuildInputs = [
              pkgs.espeak
            ];
            runtimeLibs = [
              pkgs.espeak
            ];
            uvOverride = pkgs.uv;
            members = ["server" "tinfer"];
            localDeps = ["server" "tinfer"];
            config = {
              Env = [
                "LD_LIBRARY_PATH=${pkgs.espeak}/lib:$LD_LIBRARY_PATH"
              ];
              Cmd = ["python" "-m" "server.main"];
            };
          };
          copy-platform-image = pkgs.writeShellScriptBin "copy-platform-image" ''
            set -euo pipefail

            if [ "$#" -ne 3 ]; then
              echo "Usage: $0 <platform> <image> <tag>"
              exit 1
            fi

            platform="$1"
            image="$2"
            tag="$3"

            nix build ".#packages.$platform.$image"

            if [ -n "''${DOCKERHUB_TOKEN:-}" ]; then
              echo "$DOCKERHUB_TOKEN" | ${pkgs.docker}/bin/docker login -u plan9better --password-stdin
            else
              ${pkgs.docker}/bin/docker login -u plan9better
            fi

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
              nix:result "$target"
            rm -f "$policy_file"
          '';
          default = tinfer-server;
        };
        devShells.default = pkgs.mkShell {
          packages = [pkgs.espeak pkgs.uv];
          shellHook = ''
            echo "Found: ${pkgs.espeak}/lib"
            export LD_LIBRARY_PATH=${pkgs.espeak}/lib:$LD_LIBRARY_PATH
          '';
        };
      }
    );
}
