//! Second-server proof of LSP-agnosticism: the same enrichment pass against
//! clangd over a C fixture — a language whose tree-sitter `references.scm` is
//! an empty placeholder, so call relations here can only come from the
//! server's own call hierarchy. Run explicitly:
//!
//! ```sh
//! cargo test -p codeindex-lsp --test clangd -- --ignored
//! ```

use codeindex_core::RepresentationKind;
use codeindex_indexer::{IndexSettings, RetentionMode, SourceProject};
use codeindex_lsp::{LspServer, TYPED_SIGNATURE_CHANNEL, enrich_project};

#[test]
#[ignore = "spawns clangd"]
fn clangd_enriches_c_signatures_and_call_relations() {
    let fixture = tempfile::tempdir().unwrap();
    std::fs::write(
        fixture.path().join("adder.c"),
        "/* The base value. */\n\
         int base_value(void) {\n    int value = 41;\n    return value;\n}\n\n\
         /* Adds one to the base value. */\n\
         int add_one(void) {\n    int base = base_value();\n    return base + 1;\n}\n",
    )
    .unwrap();

    let db = codeindex_sqlite::open_in_memory().unwrap();
    let source = codeindex_indexer::FileSystemSource::new(fixture.path());
    let projects = [SourceProject {
        label: "fixture".into(),
        provider: &source,
    }];
    let settings = IndexSettings {
        enabled_languages: vec!["c".into()],
        body_node_count_threshold: 1,
        max_body_chars: 10_000,
        retention: RetentionMode::Full,
    };
    codeindex_indexer::index_sources(&db, &settings, &projects, None).unwrap();

    let report = enrich_project(
        &db,
        "fixture",
        fixture.path(),
        &LspServer {
            language_id: "c".into(),
            command: "clangd".into(),
            args: Vec::new(),
        },
    )
    .unwrap();
    assert!(report.units_visited >= 2, "report: {report:?}");
    assert!(report.typed_signatures >= 1, "report: {report:?}");
    assert!(
        report.relations >= 1,
        "call hierarchy should produce relations without any references.scm: {report:?}"
    );

    let snapshot = db.snapshot(&[]).unwrap();
    let caller = snapshot
        .units
        .iter()
        .find(|unit| unit.name == "add_one")
        .expect("add_one indexed");
    let callee = snapshot
        .units
        .iter()
        .find(|unit| unit.name == "base_value")
        .expect("base_value indexed");
    let signature = caller
        .content(&RepresentationKind::Custom(TYPED_SIGNATURE_CHANNEL.into()))
        .expect("add_one has a typed signature");
    assert!(signature.contains("add_one"), "signature: {signature}");

    let call = snapshot
        .relations
        .iter()
        .find(|relation| relation.from_entity_id == caller.entity_id && relation.kind == "calls")
        .expect("add_one has a calls relation");
    assert_eq!(call.resolution, "exact");
    assert_eq!(call.to_entity_id.as_ref(), Some(&callee.entity_id));
}
