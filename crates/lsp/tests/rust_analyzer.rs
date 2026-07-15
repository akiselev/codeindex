//! Integration test driving a real rust-analyzer over a tiny cargo fixture:
//! index → LSP enrichment → typed_signature representations + exact `calls`
//! relations in the snapshot. Slow (rust-analyzer loads a workspace), so it
//! runs explicitly:
//!
//! ```sh
//! cargo test -p codeindex-lsp -- --ignored
//! ```

use codeindex_core::RepresentationKind;
use codeindex_indexer::{IndexSettings, RetentionMode, SourceProject};
use codeindex_lsp::{LspServer, TYPED_SIGNATURE_CHANNEL, enrich_project};

#[test]
#[ignore = "spawns rust-analyzer and loads a cargo workspace"]
fn rust_analyzer_enriches_signatures_and_call_relations() {
    let fixture = tempfile::tempdir().unwrap();
    std::fs::create_dir_all(fixture.path().join("src")).unwrap();
    std::fs::write(
        fixture.path().join("Cargo.toml"),
        "[package]\nname = \"lsp-fixture\"\nversion = \"0.1.0\"\nedition = \"2021\"\n",
    )
    .unwrap();
    std::fs::write(
        fixture.path().join("src/lib.rs"),
        "/// Adds one to the beta value.\n\
         pub fn alpha() -> u32 {\n    let base = beta();\n    base + 1\n}\n\n\
         /// The base value.\n\
         pub fn beta() -> u32 {\n    let value = 41;\n    value\n}\n",
    )
    .unwrap();

    let db = codeindex_sqlite::open_in_memory().unwrap();
    let source = codeindex_indexer::FileSystemSource::new(fixture.path().join("src"));
    let projects = [SourceProject {
        label: "fixture".into(),
        provider: &source,
    }];
    let settings = IndexSettings {
        enabled_languages: vec!["rust".into()],
        body_node_count_threshold: 1,
        max_body_chars: 10_000,
        retention: RetentionMode::Full,
    };
    codeindex_indexer::index_sources(&db, &settings, &projects, None).unwrap();

    let report = enrich_project(
        &db,
        "fixture",
        &fixture.path().join("src"),
        &LspServer {
            language_id: "rust".into(),
            command: "rust-analyzer".into(),
            args: Vec::new(),
        },
    )
    .unwrap();
    assert!(report.units_visited >= 2, "report: {report:?}");
    assert!(report.typed_signatures >= 1, "report: {report:?}");
    assert!(report.relations >= 1, "report: {report:?}");

    let snapshot = db.snapshot(&[]).unwrap();
    let alpha = snapshot
        .units
        .iter()
        .find(|unit| unit.name == "alpha")
        .expect("alpha indexed");
    let beta = snapshot
        .units
        .iter()
        .find(|unit| unit.name == "beta")
        .expect("beta indexed");
    let signature = alpha
        .content(&RepresentationKind::Custom(TYPED_SIGNATURE_CHANNEL.into()))
        .expect("alpha has a typed signature");
    assert!(signature.contains("alpha"), "signature: {signature}");

    let call = snapshot
        .relations
        .iter()
        .find(|relation| relation.from_entity_id == alpha.entity_id && relation.kind == "calls")
        .expect("alpha has a calls relation");
    assert_eq!(call.resolution, "exact");
    assert_eq!(call.to_entity_id.as_ref(), Some(&beta.entity_id));
}
