{pkgs}: pkgs.mkShell {
  packages = [pkgs.cargo pkgs.rustc pkgs.gcc pkgs.pkg-config pkgs.uv pkgs.ffmpeg-headless];
  buildInputs = [pkgs.onnxruntime pkgs.espeak-ng];
  ORT_INCLUDE_DIR = "${pkgs.onnxruntime.dev}/include";
  ORT_LIB_DIR = "${pkgs.onnxruntime}/lib";
  LIBRARY_PATH = "${pkgs.espeak-ng}/lib";
}
