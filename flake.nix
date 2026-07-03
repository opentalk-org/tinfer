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

        tinfer-server = import ./nix/image.nix {
          inherit pkgs;
          uv2container = x2container.lib.${system}.uv2container;
        };
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

        devShells.default = import ./nix/devshell.nix {
          inherit pkgs tinfer-server;
        };
      }
    );
}
