fn main() {
    let target = std::env::var("TARGET").unwrap_or_default();
    let host = std::env::var("HOST").unwrap_or_default();
    let cargo_cfg_target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    let is_linux = cargo_cfg_target_os == "linux" || target.contains("linux");
    let is_macos = cargo_cfg_target_os == "macos" || target.contains("apple-darwin");
    let is_windows = cargo_cfg_target_os == "windows" || target.contains("windows");

    if is_linux {
        println!("cargo:rustc-link-search=native=/lib/x86_64-linux-gnu");
        println!("cargo:rustc-link-search=native=/usr/lib/x86_64-linux-gnu");
        println!("cargo:rustc-link-lib=dylib=espeak-ng");
        return;
    }

    if is_macos {
        println!("cargo:rustc-link-lib=dylib=espeak-ng");
        return;
    }

    if is_windows {
        println!("cargo:rustc-link-lib=dylib=espeak-ng");
        return;
    }

    if !host.is_empty() || !target.is_empty() {
        println!("cargo:warning=Unknown target/host for espeak-ng linking (TARGET={target}, HOST={host}, CARGO_CFG_TARGET_OS={cargo_cfg_target_os}).");
    }
}

