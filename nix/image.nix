# The serving image. Exports its runtime contract via passthru.runtime;
# the devshell consumes only that, so the two can't drift.
{
  pkgs,
  uv2container,
  naersk,
}: let
  lib = pkgs.lib;

  python = pkgs.python311;
  cudaNvcc = pkgs.cudaPackages.cuda_nvcc;

  # naersk-built cdylib; python imports the .so off PYTHONPATH, no wheel.
  # espeak-ng resolves via rpath.
  espeak-align = (pkgs.callPackage naersk {}).buildPackage {
    src = ../tinfer/espeak_align;
    copyLibs = true;
    copyBins = false;
    buildInputs = [pkgs.espeak];
    nativeBuildInputs = [python];
    PYO3_PYTHON = "${python}/bin/python${python.pythonVersion}";
    postInstall = ''
      site="$out/lib/python${python.pythonVersion}/site-packages"
      mkdir -p "$site"
      mv "$out/lib/libespeak_align.so" "$site/espeak_align.so"
    '';
  };
  espeakAlignSite = "${espeak-align}/lib/python${python.pythonVersion}/site-packages";

  # Sonames the manylinux wheels resolve: preloaded by the devshell, on
  # LD_LIBRARY_PATH in the image. Without libz triton can't load and
  # torch.compile silently falls back to eager.
  wheelLibs = [
    {
      pkg = pkgs.stdenv.cc.cc.lib;
      sonames = ["libstdc++.so.6"];
    }
    {
      pkg = pkgs.zlib;
      sonames = ["libz.so.1"];
    }
  ];

  # glibc: prebuilt wheel binaries (the devshell uses the host loader).
  runtimeLibs = map (l: l.pkg) wheelLibs ++ [pkgs.ffmpeg-headless pkgs.gcc pkgs.glibc];

  # ffmpeg-headless covers every codec the server encodes (lame, opus,
  # mulaw, pcm); the default variant adds ~750MB of display/capture closure.
  runtimeExecutableDeps = [pkgs.ffmpeg-headless pkgs.patchelf pkgs.gcc pkgs.openssl cudaNvcc];

  # Container-toolkit mount points, then the stock path of bare hosts.
  nvidiaDriverDirs = ["/usr/local/nvidia/lib" "/usr/local/nvidia/lib64" "/usr/lib/x86_64-linux-gnu"];
  nvidiaDriverPath = lib.concatStringsSep ":" nvidiaDriverDirs;

  commonEnv = {
    CC = "${pkgs.gcc}/bin/gcc";
    SSL_CERT_FILE = "${pkgs.cacert}/etc/ssl/certs/ca-bundle.crt";
    PYTHONUNBUFFERED = "1";
    TRITON_PTXAS_PATH = "${cudaNvcc}/bin/ptxas";
    TRITON_PTXAS_BLACKWELL_PATH = "${cudaNvcc}/bin/ptxas";
  };
in
  (uv2container.buildImage {
    name = "tinfer";
    src = ../.;
    inherit python runtimeLibs runtimeExecutableDeps;
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
    members = ["server" "tinfer"];
    extraPythonPath = ":" + espeakAlignSite;
    config = {
      Env = lib.mapAttrsToList (k: v: "${k}=${v}") (commonEnv
        // {
          USER = "root";
          HOME = "/root";
          TORCHINDUCTOR_CACHE_DIR = "/tmp/torchinductor";
          # A directory: triton asserts $TRITON_LIBCUDA_PATH/libcuda.so.1
          # exists (the previous file-path value could never pass that).
          TRITON_LIBCUDA_PATH = "/usr/local/nvidia/lib";
        });
      Cmd = ["python" "-m" "server.main"];
    };
  })
  .overrideAttrs (old: {
    passthru =
      (old.passthru or {})
      // {
        runtime = {
          inherit python runtimeLibs runtimeExecutableDeps nvidiaDriverDirs nvidiaDriverPath espeakAlignSite;
          espeakAlign = espeak-align;
          env = commonEnv;
          # For hacking the espeak_align crate directly (cargo test, the
          # editable dev symlink).
          crateDevTools = [pkgs.espeak pkgs.rustc pkgs.cargo];
          preloadLibs = lib.concatMap (l: map (s: "${l.pkg}/lib/${s}") l.sonames) wheelLibs;
        };
      };
  })
