{
  inputs.nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
  inputs.flake-utils.url = "github:numtide/flake-utils";

  outputs = {nixpkgs, flake-utils, ...}:
    flake-utils.lib.eachDefaultSystem (system: let
      pkgs = import nixpkgs {inherit system;};
      server = import ./nix/server.nix {inherit pkgs;};
    in {
      packages = {
        tinfer-rust = server;
        tinfer-server = import ./nix/image.nix {inherit pkgs server;};
        default = server;
      };
      devShells.default = import ./nix/devshell.nix {inherit pkgs;};
    });
}
