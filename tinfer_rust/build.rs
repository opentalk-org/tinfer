use std::env;
use std::path::PathBuf;

fn main() {
    let protoc = protoc_bin_vendored::protoc_bin_path().expect("vendored protoc binary");
    unsafe { env::set_var("PROTOC", protoc) };
    let output = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo output directory"));
    tonic_build::configure()
        .file_descriptor_set_path(output.join("styletts_descriptor.bin"))
        .compile_protos(&["proto/styletts.proto"], &["proto"])
        .expect("compile styletts gRPC contract");
    let manifest = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").expect("manifest directory"));
    let workspace = manifest.parent().expect("tinfer_rust parent directory");
    let mut native = cxx_build::bridge("src/models/base/native.rs");
    native
        .file("src/models/base/cpp/model.cpp")
        .file("src/models/base/cpp/tensor.cpp")
        .file("src/models/base/cpp/engine.cpp")
        .file("src/models/base/cpp/onnx.cpp")
        .file("src/models/base/cpp/tensorrt.cpp")
        .file("src/models/stub/cpp/model.cpp")
        .file("src/models/styletts2/cpp/model.cpp")
        .file("src/models/styletts2/cpp/device.cpp")
        .file("src/models/styletts2/cpp/pipeline.cpp")
        .file("src/models/styletts2/cpp/prosody.cpp")
        .file("src/models/styletts2/cpp/cuda/state.cpp")
        .file("src/models/styletts2/cpp/window.cpp")
        .file("src/models/styletts2/cpp/cpu/glue.cpp")
        .include(workspace)
        .flag_if_supported("-std=c++20");
    let onnx = env::var_os("CARGO_FEATURE_ONNX").is_some();
    let tensorrt = env::var_os("CARGO_FEATURE_TENSORRT").is_some();
    let cuda = env::var_os("CARGO_FEATURE_NATIVE_CUDA").is_some();
    if onnx {
        let include = PathBuf::from(env::var_os("ORT_INCLUDE_DIR").expect("ORT_INCLUDE_DIR is required by onnx"));
        let library = PathBuf::from(env::var_os("ORT_LIB_DIR").expect("ORT_LIB_DIR is required by onnx"));
        native.define("TINFER_ONNX", None).include(include);
        println!("cargo:rustc-link-search=native={}", library.display());
        let unversioned = library.join("libonnxruntime.so");
        if unversioned.exists() {
            println!("cargo:rustc-link-lib=dylib=onnxruntime");
        } else {
            let versioned = std::fs::read_dir(&library)
                .expect("read ORT_LIB_DIR")
                .map(|entry| entry.expect("read ONNX Runtime library entry").file_name())
                .find(|name| name.to_string_lossy().starts_with("libonnxruntime.so."))
                .expect("ORT_LIB_DIR must contain libonnxruntime.so");
            println!("cargo:rustc-link-lib=dylib:+verbatim={}", versioned.to_string_lossy());
        }
    }
    if cuda {
        let cuda_home = PathBuf::from(env::var_os("CUDA_HOME").expect("CUDA_HOME is required by native-cuda"));
        native.define("TINFER_CUDA", None).include(cuda_home.join("include"));
        cc::Build::new()
            .cuda(true)
            .file("src/models/styletts2/cpp/cuda/kernels.cu")
            .include(workspace)
            .include(cuda_home.join("include"))
            .flag("-std=c++20")
            .flag("-Xcompiler")
            .flag("-U_GNU_SOURCE")
            .compile("tinfer_styletts2_cuda");
        println!("cargo:rustc-link-search=native={}", cuda_home.join("lib64").display());
        println!("cargo:rustc-link-lib=dylib=cudart");
    }
    if tensorrt {
        let include = PathBuf::from(env::var_os("TENSORRT_INCLUDE_DIR").expect("TENSORRT_INCLUDE_DIR is required by tensorrt"));
        let library = PathBuf::from(env::var_os("TENSORRT_LIB_DIR").expect("TENSORRT_LIB_DIR is required by tensorrt"));
        native.define("TINFER_TENSORRT", None).include(include);
        println!("cargo:rustc-link-search=native={}", library.display());
        let unversioned = library.join("libnvinfer.so");
        if unversioned.exists() {
            println!("cargo:rustc-link-lib=dylib=nvinfer");
        } else {
            let versioned = std::fs::read_dir(&library)
                .expect("read TENSORRT_LIB_DIR")
                .map(|entry| entry.expect("read TensorRT library entry").file_name())
                .find(|name| name.to_string_lossy().starts_with("libnvinfer.so."))
                .expect("TENSORRT_LIB_DIR must contain libnvinfer.so");
            println!("cargo:rustc-link-lib=dylib:+verbatim={}", versioned.to_string_lossy());
        }
    }
    native.compile("tinfer_models_native");
    println!("cargo:rerun-if-changed=proto/styletts.proto");
    println!("cargo:rerun-if-changed=src/models/base/native.rs");
    println!("cargo:rerun-if-changed=src/models/base/cpp/model.hpp");
    println!("cargo:rerun-if-changed=src/models/base/cpp/model.cpp");
    println!("cargo:rerun-if-changed=src/models/base/cpp/tensor.hpp");
    println!("cargo:rerun-if-changed=src/models/base/cpp/tensor.cpp");
    println!("cargo:rerun-if-changed=src/models/base/cpp/engine.hpp");
    println!("cargo:rerun-if-changed=src/models/base/cpp/engine.cpp");
    println!("cargo:rerun-if-changed=src/models/base/cpp/onnx.hpp");
    println!("cargo:rerun-if-changed=src/models/base/cpp/onnx.cpp");
    println!("cargo:rerun-if-changed=src/models/base/cpp/tensorrt.hpp");
    println!("cargo:rerun-if-changed=src/models/base/cpp/tensorrt.cpp");
    println!("cargo:rerun-if-changed=src/models/stub/cpp/native.hpp");
    println!("cargo:rerun-if-changed=src/models/stub/cpp/model.cpp");
    println!("cargo:rerun-if-changed=src/models/styletts2/cpp/cpu/glue.hpp");
    println!("cargo:rerun-if-changed=src/models/styletts2/cpp/cpu/glue.cpp");
    println!("cargo:rerun-if-changed=src/models/styletts2/cpp/model.hpp");
    println!("cargo:rerun-if-changed=src/models/styletts2/cpp/device.cpp");
    println!("cargo:rerun-if-changed=src/models/styletts2/cpp/pipeline.cpp");
    println!("cargo:rerun-if-changed=src/models/styletts2/cpp/prosody.cpp");
    println!("cargo:rerun-if-changed=src/models/styletts2/cpp/session.hpp");
    println!("cargo:rerun-if-changed=src/models/styletts2/cpp/window.hpp");
    println!("cargo:rerun-if-changed=src/models/styletts2/cpp/window.cpp");
    println!("cargo:rerun-if-changed=src/models/styletts2/cpp/model.cpp");
    println!("cargo:rerun-if-changed=src/models/styletts2/cpp/cuda/glue.hpp");
    println!("cargo:rerun-if-changed=src/models/styletts2/cpp/cuda/kernels.cu");
    println!("cargo:rerun-if-changed=src/models/styletts2/cpp/cuda/state.cpp");
    println!("cargo:rerun-if-env-changed=ORT_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=ORT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=TENSORRT_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=TENSORRT_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
}
