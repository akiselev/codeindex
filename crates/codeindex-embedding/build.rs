use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=../../Cargo.lock");
    let lock = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Cargo.lock");
    let text = std::fs::read_to_string(lock).unwrap_or_default();
    for (package, var) in [
        ("fastembed", "DECOMBINE_FASTEMBED_VERSION"),
        ("ort", "DECOMBINE_ORT_VERSION"),
    ] {
        let version = locked_version(&text, package).unwrap_or_else(|| "unknown".to_string());
        println!("cargo:rustc-env={var}={version}");
    }
}

fn locked_version(lock: &str, package: &str) -> Option<String> {
    let mut in_package = false;
    for line in lock.lines() {
        if line == "[[package]]" {
            in_package = false;
        } else if line == format!("name = \"{package}\"") {
            in_package = true;
        } else if in_package && let Some(version) = line.strip_prefix("version = \"") {
            return Some(version.trim_end_matches('"').to_string());
        }
    }
    None
}
