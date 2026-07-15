//! End-to-end daemon lifecycle through the real binary: add a project (the
//! daemon autostarts and indexes it), search it from a subdirectory (upward
//! `.codeindex.toml` discovery), folder overrides exclude their subtree,
//! then remove and stop. No embedding spaces are configured, so the whole
//! test runs model-free on lexical retrieval.

use std::path::Path;
use std::process::{Command, Output};
use std::time::{Duration, Instant};

struct DaemonFixture {
    binary: &'static str,
    data_dir: tempfile::TempDir,
    isolation: String,
}

impl DaemonFixture {
    fn new() -> Self {
        Self {
            binary: env!("CARGO_BIN_EXE_codeindex"),
            data_dir: tempfile::tempdir().expect("tempdir"),
            isolation: format!("citest-{}", std::process::id()),
        }
    }

    fn run(&self, cwd: &Path, args: &[&str]) -> Output {
        Command::new(self.binary)
            .args(args)
            .current_dir(cwd)
            .env("CODEINDEX_DATA_DIR", self.data_dir.path())
            .env("CODEINDEX_ISOLATION", &self.isolation)
            .output()
            .expect("running codeindex")
    }

    fn run_ok(&self, cwd: &Path, args: &[&str]) -> String {
        let output = self.run(cwd, args);
        assert!(
            output.status.success(),
            "codeindex {args:?} failed\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr),
        );
        String::from_utf8_lossy(&output.stdout).into_owned()
    }

    fn json(&self, cwd: &Path, args: &[&str]) -> serde_json::Value {
        let stdout = self.run_ok(cwd, args);
        let line = stdout
            .lines()
            .last()
            .unwrap_or_else(|| panic!("no output from {args:?}"));
        serde_json::from_str::<serde_json::Value>(line)
            .unwrap_or_else(|error| panic!("bad JSON from {args:?}: {error}\n{stdout}"))
    }
}

impl Drop for DaemonFixture {
    fn drop(&mut self) {
        let _ = Command::new(self.binary)
            .args(["daemon", "stop"])
            .env("CODEINDEX_DATA_DIR", self.data_dir.path())
            .env("CODEINDEX_ISOLATION", &self.isolation)
            .output();
    }
}

fn write(path: &Path, text: &str) {
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(path, text).unwrap();
}

#[test]
fn daemon_lifecycle_add_search_remove() {
    let fixture = DaemonFixture::new();
    let tree = tempfile::tempdir().unwrap();
    let root = tree.path().join("demo");
    write(
        &root.join("src/geometry.rs"),
        "/// Computes the frobnication quotient of a mesh.\n\
         pub fn frobnication_quotient(vertices: usize, edges: usize) -> f64 {\n\
             (vertices * 3 + edges) as f64 / 7.0\n\
         }\n",
    );
    // Same searchable tokens as the real function: if the vendor override
    // leaks into the index, this WILL rank for the test query.
    write(
        &root.join("vendor/decoy.rs"),
        "pub fn frobnication_quotient_decoy(vertices: usize, edges: usize) -> f64 {\n    \
         (vertices + edges) as f64\n}\n",
    );
    write(&root.join("vendor/.codeindex.toml"), "index = false\n");

    // `add` writes the starter root config, autostarts the daemon, and
    // spawns the index job.
    let added = fixture.json(&root, &["add", ".", "--json"]);
    assert!(
        root.join(".codeindex.toml").is_file(),
        "starter config written"
    );
    let label = added["data"]["label"].as_str().expect("label").to_owned();
    assert_eq!(added["data"]["job"], "indexing");

    // Wait for the background job to finish.
    let deadline = Instant::now() + Duration::from_secs(120);
    let units = loop {
        let status = fixture.json(&root, &["status", "--json"]);
        let projects = status["data"]["projects"].as_array().expect("projects");
        let project = projects
            .iter()
            .find(|project| project["label"] == label.as_str())
            .expect("registered project in status");
        let job = &project["job"];
        if job["finished"] == true {
            assert!(job["error"].is_null(), "index job failed: {}", job["error"]);
            break project["units"].as_i64().expect("unit count");
        }
        assert!(
            Instant::now() < deadline,
            "index job did not finish: {status}"
        );
        std::thread::sleep(Duration::from_millis(200));
    };
    assert!(
        units >= 1,
        "expected at least the real function, got {units}"
    );

    // Search from a nested directory: discovery walks up to the root and
    // the daemon serves the warm index. The vendor decoy must not appear.
    let nested = root.join("src");
    let results = fixture.json(
        &nested,
        &["search", "frobnication_quotient vertices edges", "--json"],
    );
    let hits = results["data"]["hits"].as_array().expect("hits");
    assert!(
        hits.iter()
            .any(|hit| hit["name"] == "frobnication_quotient"),
        "expected frobnication_quotient in {results}"
    );
    assert!(
        hits.iter()
            .all(|hit| hit["name"] != "frobnication_quotient_decoy"),
        "vendor override leaked into the index: {results}"
    );
    assert!(
        hits.iter().all(|hit| {
            hit["sources"].as_array().is_some_and(|sources| {
                sources
                    .iter()
                    .all(|source| source.as_str().unwrap_or("").starts_with("lexical#"))
            })
        }),
        "model-free project should be lexical-only: {results}"
    );

    // The `query` alias answers identically.
    let aliased = fixture.json(&nested, &["query", "frobnication_quotient", "--json"]);
    assert!(
        aliased["data"]["hits"]
            .as_array()
            .is_some_and(|hits| !hits.is_empty()),
        "query alias returned nothing: {aliased}"
    );

    // Remove with purge deletes the managed database directory.
    let removed = fixture.json(&root, &["remove", &label, "--purge", "--json"]);
    assert_eq!(removed["data"]["label"], label.as_str());
    let purged = removed["data"]["purged"].as_str().expect("purged path");
    assert!(!Path::new(purged).exists(), "purged dir still present");

    let status = fixture.json(&root, &["status", "--json"]);
    assert_eq!(
        status["data"]["projects"].as_array().map(Vec::len),
        Some(0),
        "project still registered after remove: {status}"
    );

    // Explicit stop; the Drop guard is only a safety net.
    fixture.run_ok(&root, &["daemon", "stop"]);
}
