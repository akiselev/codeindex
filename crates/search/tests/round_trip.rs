//! End-to-end round trip across the crate seams: store units + representations
//! in SQLite, embed one channel with the offline hash backend, export a
//! storage-neutral snapshot, load a `SearchIndex` from it, and exercise the
//! channel-aware search operations. Walks sqlite → storage → embedding →
//! query → search together.

use codeindex_core::RepresentationKind;
use codeindex_embedding::embed::hash::HashEmbedder;
use codeindex_embedding::{Embedder, normalize_in_place};
use codeindex_query::WhereFilter;
use codeindex_search::{SearchIndex, resolve_selector};
use codeindex_sqlite::{NewCodeUnit, NewFile, NewRepresentation, open_in_memory};

const DIMS: usize = 64;

/// (name, body) for three units with distinct vocabulary.
const BODIES: &[(&str, &str)] = &[
    ("parse_flags", "parse command line flags into typed options"),
    ("render_table", "render html table rows from records"),
    ("checksum", "compute a rolling checksum over file bytes"),
];

fn unit(name: &str, body: &str, start: usize) -> NewCodeUnit {
    let body_hash = format!("body-{name}");
    NewCodeUnit {
        entity_id: format!("ent-{name}"),
        entity_version_id: format!("ver-{name}"),
        generation: 1,
        language_id: "rust".into(),
        kind: "function".into(),
        name: name.into(),
        scope: None,
        start_byte: start,
        end_byte: start + body.len(),
        start_line: start,
        end_line: start + 1,
        body_node_count: 10,
        source_hash: format!("src-{name}"),
        normalized_body_hash: body_hash.clone(),
        representations: vec![
            NewRepresentation {
                kind: RepresentationKind::FullSource,
                content_hash: format!("src-{name}"),
                content: Some(body.to_owned()),
            },
            NewRepresentation {
                kind: RepresentationKind::Implementation,
                content_hash: body_hash,
                content: Some(body.to_owned()),
            },
        ],
    }
}

/// Build an in-memory index: one project, one file, the three bodies embedded
/// on the Implementation channel with the hash backend.
fn build_index() -> (SearchIndex, HashEmbedder) {
    let db = open_in_memory().unwrap();
    let project_id = db.upsert_project("main", "/src").unwrap();
    let file_id = db
        .upsert_file(&NewFile {
            project_id,
            relative_path: "lib.rs".into(),
            language_id: "rust".into(),
            mtime_ns: 0,
            size: 0,
            source_hash: "file".into(),
        })
        .unwrap();
    let units: Vec<NewCodeUnit> = BODIES
        .iter()
        .enumerate()
        .map(|(i, (name, body))| unit(name, body, i + 1))
        .collect();
    db.insert_units(file_id, &units).unwrap();

    let mut embedder = HashEmbedder::new(DIMS);
    let model_id = db.find_or_create_model(embedder.identity()).unwrap();
    let bodies: Vec<String> = BODIES.iter().map(|(_, body)| (*body).to_owned()).collect();
    let mut vectors = embedder.embed(&bodies).unwrap();
    for vector in vectors.iter_mut() {
        normalize_in_place(vector);
    }
    for ((name, _), vector) in BODIES.iter().zip(vectors.iter()) {
        db.insert_embedding(
            model_id,
            &RepresentationKind::Implementation,
            &format!("body-{name}"),
            vector,
        )
        .unwrap();
    }

    let snapshot = db.snapshot(&[]).unwrap();
    (SearchIndex::from_snapshot(snapshot), embedder)
}

fn impl_channel() -> RepresentationKind {
    RepresentationKind::Implementation
}

#[test]
fn load_populates_units_and_vectors() {
    let (index, _) = build_index();
    assert_eq!(index.units.len(), 3);
    let store = index.channels.get(&impl_channel()).unwrap();
    assert_eq!(store.len(), 3, "every body was embedded");
    assert_eq!(index.identity.dimensions, DIMS);
}

#[test]
fn search_text_ranks_by_shared_vocabulary() {
    let (index, mut embedder) = build_index();
    let filter = WhereFilter::default();
    let results = index
        .search_text(
            &mut embedder,
            "parse command line flags",
            &impl_channel(),
            &filter,
            10,
        )
        .unwrap();
    assert_eq!(results.matched, 3, "all embedded units are candidates");
    let top = &index.units[results.hits[0].index];
    assert_eq!(top.name, "parse_flags", "shared tokens rank highest");
    assert!(results.hits[0].score >= results.hits[1].score);
}

#[test]
fn search_text_rejects_mismatched_model_identity() {
    let (index, _) = build_index();
    let mut other = HashEmbedder::new(DIMS * 2);
    let err = index
        .search_text(&mut other, "anything", &impl_channel(), &WhereFilter::default(), 10)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("same model identity") && err.contains("dimensions"),
        "identity mismatch should be reported with the differing field: {err}"
    );
}

#[test]
fn search_missing_channel_errors() {
    let (index, _) = build_index();
    let err = index
        .search_vector(&[0.0; DIMS], &RepresentationKind::Signature, &WhereFilter::default(), 10)
        .unwrap_err()
        .to_string();
    assert!(err.contains("no embeddings"), "unhelpful error: {err}");
}

#[test]
fn similar_to_unit_excludes_query_and_honors_threshold() {
    let (index, _) = build_index();
    let query_index = index
        .units
        .iter()
        .position(|u| u.name == "parse_flags")
        .unwrap();
    let all = index
        .similar_to_unit(query_index, &impl_channel(), &WhereFilter::default(), 10, -1.0)
        .unwrap();
    assert_eq!(all.matched, 2, "the query unit itself is excluded");
    assert!(all.hits.iter().all(|h| h.index != query_index));

    let none = index
        .similar_to_unit(query_index, &impl_channel(), &WhereFilter::default(), 10, 1.5)
        .unwrap();
    assert_eq!(none.matched, 0);
}

#[test]
fn resolve_selector_round_trips() {
    let (index, _) = build_index();
    let id = codeindex_query::unit_id(&index.units[0]);
    assert_eq!(resolve_selector(&index.units, &id).unwrap(), 0);
    assert!(resolve_selector(&index.units, "unit:deadbeef").is_err());
    assert!(resolve_selector(&index.units, "not-a-selector").is_err());
}
