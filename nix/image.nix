# The serving image. Exports its runtime contract via passthru.runtime;
# the devshell consumes only that, so the two can't drift. Python deps
# come from the same uv.lock in both.
{
  pkgs,
  uv2container,
  naersk,
}: let
  lib = pkgs.lib;

  python = pkgs.python311;
  cudaNvcc = pkgs.cudaPackages.cuda_nvcc;

  # The extension module as a nix-built cdylib; python imports the .so
  # off PYTHONPATH, no wheel. espeak-ng resolves via rpath. naersk
  # caches the compiled dependency graph keyed by Cargo.lock, so source
  # edits only rebuild the crate itself.
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

  # Native libs the manylinux wheels resolve at runtime, with the sonames
  # they ask for. The devshell preloads exactly these files; the image
  # exposes the packages on LD_LIBRARY_PATH. zlib: libtriton.so links
  # libz; without it torch.compile silently falls back to eager.
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

  # Image-only: glibc for prebuilt binaries shipped in wheels (in the
  # devshell the host loader owns libc), ffmpeg/gcc libs for torchaudio
  # and torch.compile.
  runtimeLibs =
    map (l: l.pkg) wheelLibs
    ++ [
      pkgs.ffmpeg-headless
      pkgs.gcc
      pkgs.glibc
    ];

  # Executables needed at runtime (torch.compile, triton, audio).
  # ffmpeg-headless has every codec the server encodes with (lame, opus,
  # mulaw, pcm); the default variant drags in ~750MB of display/capture
  # dependencies.
  runtimeExecutableDeps = [
    pkgs.ffmpeg-headless
    pkgs.patchelf
    pkgs.gcc
    pkgs.openssl
    cudaNvcc
  ];

  # Driver locations: nvidia container toolkit mounts, then the stock
  # distro path bare hosts have.
  nvidiaDriverDirs = [
    "/usr/local/nvidia/lib"
    "/usr/local/nvidia/lib64"
    "/usr/lib/x86_64-linux-gnu"
  ];
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
          # Toolchain for hacking the espeak_align crate directly
          # (cargo test / the editable dev symlink).
          crateDevTools = [
            pkgs.espeak
            pkgs.rustc
            pkgs.cargo
          ];
          # The wheelLibs sonames as absolute paths (the devshell preloads
          # these).
          preloadLibs = lib.concatMap (l: map (s: "${l.pkg}/lib/${s}") l.sonames) wheelLibs;
        };
      };
  })
