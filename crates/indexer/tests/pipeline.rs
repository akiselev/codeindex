//! End-to-end M4 acceptance tests driving the real pipeline: index Rust source
//! on disk, resolve the Usage channel, embed every channel, and export a
//! snapshot. Covers entity identity across a rename, Usage call-site capture,
//! and multi-channel extraction + embedding.

use codeindex_core::RepresentationKind;
use codeindex_embedding::config::{EmbeddingConfig, EmbeddingRunConfig, SourceRecoveryConfig};
use codeindex_embedding::embed::hash::HashEmbedder;
use codeindex_indexer::{IndexOptions, ProjectSpec, RetentionMode, embed_pending, index};
use codeindex_sqlite::{Db, open_in_memory};

fn options(dir: &std::path::Path) -> IndexOptions {
    IndexOptions {
        projects: vec![ProjectSpec {
            label: "main".into(),
            source_dir: dir.to_path_buf(),
            exclude: Vec::new(),
        }],
        enabled_languages: vec!["rust".into()],
        body_node_count_threshold: 1,
        max_body_chars: 10_000,
        retention: RetentionMode::Full,
    }
}

fn run_config() -> EmbeddingRunConfig {
    EmbeddingRunConfig {
        embedding: EmbeddingConfig::default(),
        source_recovery: SourceRecoveryConfig {
            body_node_count_threshold: 1,
        },
    }
}

fn entity_of(db: &Db, name: &str) -> (String, String) {
    let project_id = db.get_project("main").unwrap().unwrap().id;
    let unit = db
        .list_units_for_project(project_id)
        .unwrap()
        .into_iter()
        .find(|u| u.name == name)
        .unwrap_or_else(|| panic!("unit {name} not found"));
    (unit.entity_id, unit.entity_version_id)
}

fn representation(db: &Db, name: &str, kind: &RepresentationKind) -> Option<String> {
    db.conn()
        .query_row(
            "SELECT r.content FROM representations r
             JOIN code_units u ON u.id = r.unit_id
             WHERE u.name = ?1 AND r.kind = ?2",
            rusqlite::params![name, kind.as_str()],
            |row| row.get::<_, Option<String>>(0),
        )
        .optional()
        .unwrap()
        .flatten()
}

use rusqlite::OptionalExtension;

#[test]
fn entity_id_survives_rename_but_version_changes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("lib.rs");
    std::fs::write(
        &path,
        "fn alpha(a: i32, b: i32) -> i32 { let sum = a + b; let doubled = sum * 2; doubled }",
    )
    .unwrap();
    let db = open_in_memory().unwrap();
    index(&db, &options(dir.path()), None).unwrap();
    let (id_before, ver_before) = entity_of(&db, "alpha");

    // Rename the function, keep the body identical -> rename detection by body
    // hash must carry the same entity_id forward.
    std::fs::write(
        &path,
        "fn beta(a: i32, b: i32) -> i32 { let sum = a + b; let doubled = sum * 2; doubled }",
    )
    .unwrap();
    // Force a re-read (mtime/size may match on fast writes): change size via a
    // trailing space would change the body; instead re-index picks up the
    // content-hash change because the source differs.
    index(&db, &options(dir.path()), None).unwrap();
    let (id_after, ver_after) = entity_of(&db, "beta");

    assert_eq!(id_before, id_after, "rename preserves the logical entity id");
    assert_ne!(
        ver_before, ver_after,
        "a new source version gets a new entity_version_id"
    );
}

#[test]
fn usage_channel_records_call_sites() {
    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("lib.rs"),
        r#"
fn helper() -> i32 { let x = 1; let y = 2; x + y }
fn caller() -> i32 { let a = helper(); let b = helper(); a + b }
"#,
    )
    .unwrap();
    let db = open_in_memory().unwrap();
    index(&db, &options(dir.path()), None).unwrap();

    let usage = representation(&db, "helper", &RepresentationKind::Usage)
        .expect("helper should have a Usage representation");
    assert!(
        usage.contains("caller"),
        "Usage should name the calling function: {usage:?}"
    );
    // The callee that is never called has no Usage channel.
    assert!(representation(&db, "caller", &RepresentationKind::Usage).is_none());
}

#[test]
fn all_channels_extracted_and_embedded_and_searchable() {
    use codeindex_embedding::Embedder;
    use codeindex_query::WhereFilter;
    use codeindex_search::SearchIndex;

    let dir = tempfile::tempdir().unwrap();
    std::fs::write(
        dir.path().join("lib.rs"),
        r#"
/// Parses command line flags into typed options.
fn parse_flags(input: &str) -> i32 { let n = input.len(); let doubled = n * 2; doubled as i32 }

fn use_it() -> i32 { let value = parse_flags("hello world"); value }
"#,
    )
    .unwrap();
    let db = open_in_memory().unwrap();
    index(&db, &options(dir.path()), None).unwrap();

    // Definition-site channels are present.
    for kind in [
        RepresentationKind::FullSource,
        RepresentationKind::Implementation,
        RepresentationKind::Signature,
        RepresentationKind::Documentation,
        RepresentationKind::Symbol,
    ] {
        assert!(
            representation(&db, "parse_flags", &kind).is_some(),
            "channel {kind} missing"
        );
    }
    let signature = representation(&db, "parse_flags", &RepresentationKind::Signature).unwrap();
    assert!(signature.contains("fn parse_flags") && !signature.contains('{'));
    let doc = representation(&db, "parse_flags", &RepresentationKind::Documentation).unwrap();
    assert!(doc.contains("Parses command line flags"));

    // Embed every channel, then search each independently.
    let mut embedder = HashEmbedder::new(48);
    embed_pending(&db, &mut embedder, &run_config()).unwrap();

    let index = SearchIndex::from_snapshot(db.snapshot(&[]).unwrap());
    let embedded: Vec<String> = index.embedded_channels().map(|c| c.to_string()).collect();
    for expected in ["implementation", "signature", "documentation", "symbol"] {
        assert!(
            embedded.iter().any(|c| c == expected),
            "channel {expected} not embedded; have {embedded:?}"
        );
    }

    // A query targets a specific channel.
    let results = index
        .search_text(
            &mut embedder,
            "parse command line flags",
            &RepresentationKind::Documentation,
            &WhereFilter::default(),
            10,
        )
        .unwrap();
    assert!(results.matched >= 1);
    assert_eq!(index.units[results.hits[0].index].name, "parse_flags");
}
