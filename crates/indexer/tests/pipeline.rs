//! End-to-end acceptance tests for source providers, representations, entity
//! identity, Usage, and per-space embedding/search.

use codeindex_core::{EmbeddingSpaceId, RepresentationKind};
use codeindex_embedding::config::{EmbeddingConfig, EmbeddingRunConfig, SourceRecoveryConfig};
use codeindex_embedding::embed::hash::HashEmbedder;
use codeindex_indexer::{IndexOptions, ProjectSpec, RetentionMode, embed_pending, index};
use codeindex_sqlite::{Db, open_in_memory};
use rusqlite::OptionalExtension;

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
        .find(|unit| unit.name == name)
        .unwrap_or_else(|| panic!("unit {name} not found"));
    (
        unit.entity_id.to_string(),
        unit.entity_version_id.to_string(),
    )
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
    let (id_before, version_before) = entity_of(&db, "alpha");

    std::fs::write(
        &path,
        "fn beta(a: i32, b: i32) -> i32 { let sum = a + b; let doubled = sum * 2; doubled }",
    )
    .unwrap();
    index(&db, &options(dir.path()), None).unwrap();
    let (id_after, version_after) = entity_of(&db, "beta");

    assert_eq!(id_before, id_after);
    assert_ne!(version_before, version_after);
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
    assert!(usage.contains("caller"));
    assert!(representation(&db, "caller", &RepresentationKind::Usage).is_none());
}

#[test]
fn all_channels_are_embedded_into_independent_default_spaces() {
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

    for kind in [
        RepresentationKind::FullSource,
        RepresentationKind::Implementation,
        RepresentationKind::Body,
        RepresentationKind::BodyWithoutDeclaredName,
        RepresentationKind::Signature,
        RepresentationKind::Documentation,
        RepresentationKind::Symbol,
    ] {
        assert!(
            representation(&db, "parse_flags", &kind).is_some(),
            "channel {kind} missing"
        );
    }

    let mut embedder = HashEmbedder::new(48);
    let stats = embed_pending(&db, &mut embedder, &run_config()).unwrap();
    assert!(stats.spaces >= 6);

    let index = SearchIndex::from_snapshot(db.snapshot(&[]).unwrap()).unwrap();
    let embedded: Vec<String> = index
        .embedded_spaces()
        .map(|space| space.id.to_string())
        .collect();
    for expected in [
        "default/implementation",
        "default/body",
        "default/body_without_declared_name",
        "default/signature",
        "default/documentation",
        "default/symbol",
    ] {
        assert!(embedded.iter().any(|space| space == expected));
    }

    let results = index
        .search_text(
            &mut embedder,
            "parse command line flags",
            None,
            &EmbeddingSpaceId::new("default/documentation"),
            &WhereFilter::default(),
            10,
        )
        .unwrap();
    assert!(results.matched >= 1);
    assert_eq!(index.units[results.hits[0].index].name, "parse_flags");
}
