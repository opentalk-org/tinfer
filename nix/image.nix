{pkgs, server}: pkgs.dockerTools.buildLayeredImage {
  name = "tinfer";
  tag = "latest";
  contents = [server pkgs.ffmpeg-headless pkgs.cacert];
  extraCommands = ''
    mkdir -p etc/tinfer
    cp ${../tinfer_rust/config.yaml} etc/tinfer/config.yaml
  '';
  config = {
    Cmd = ["${server}/bin/tinfer_rust" "/etc/tinfer/config.yaml"];
    ExposedPorts = {"8000/tcp" = {}; "50051/tcp" = {};};
  };
}
