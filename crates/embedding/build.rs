use std::path::Path;

fn main() {
    println!("cargo:rerun-if-changed=../../Cargo.lock");
    let lock = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../Cargo.lock");
    let text = std::fs::read_to_string(lock).unwrap_or_default();
    for (package, var) in [
        ("fastembed", "CODEINDEX_FASTEMBED_VERSION"),
        ("ort", "CODEINDEX_ORT_VERSION"),
        ("candle-core", "CODEINDEX_CANDLE_VERSION"),
    ] {
        let versions = locked_versions(&text, package);
        let version = if versions.is_empty() {
            "unknown".to_string()
        } else {
            // With more than one resolved version a textual scan cannot tell
            // which one this crate links; report all of them so the persisted
            // identity is never silently wrong.
            versions.join("+")
        };
        println!("cargo:rustc-env={var}={version}");
    }
}

fn locked_versions(lock: &str, package: &str) -> Vec<String> {
    let mut versions = Vec::new();
    let mut in_package = false;
    for line in lock.lines() {
        if line == "[[package]]" {
            in_package = false;
        } else if line == format!("name = \"{package}\"") {
            in_package = true;
        } else if in_package && let Some(version) = line.strip_prefix("version = \"") {
            let version = version.trim_end_matches('"').to_string();
            if !versions.contains(&version) {
                versions.push(version);
            }
            in_package = false;
        }
    }
    versions
}
