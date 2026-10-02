{pkgs}: pkgs.rustPlatform.buildRustPackage {
  pname = "tinfer-rust";
  version = "0.1.0";
  src = pkgs.lib.cleanSourceWith {
    src = ../tinfer_rust;
    filter = path: type:
      builtins.baseNameOf path != "target" && pkgs.lib.cleanSourceFilter path type;
  };
  cargoLock.lockFile = ../tinfer_rust/Cargo.lock;
  buildFeatures = ["onnx"];
  nativeBuildInputs = [pkgs.pkg-config];
  buildInputs = [pkgs.onnxruntime pkgs.espeak-ng];
  ORT_INCLUDE_DIR = "${pkgs.onnxruntime.dev}/include";
  ORT_LIB_DIR = "${pkgs.onnxruntime}/lib";
}
