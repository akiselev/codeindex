//! End-to-end storage-neutral search tests with multiple embedding spaces.

use codeindex_core::{
    EmbeddingSpaceId, EmbeddingSpaceIdentity, EntityId, EntityVersionId, RepresentationKind,
    RepresentationOrigin,
};
use codeindex_embedding::embed::hash::HashEmbedder;
use codeindex_embedding::{EmbedRequest, EmbeddingBackend, normalize_in_place};
use codeindex_query::WhereFilter;
use codeindex_search::{SearchIndex, SpaceVectorQuery, resolve_selector};
use codeindex_sqlite::{NewCodeUnit, NewFile, NewRepresentation, open_in_memory};

const DIMS: usize = 64;
const BODIES: &[(&str, &str, &str)] = &[
    (
        "parse_flags",
        "parse command line flags into typed options",
        "Parses command line flags.",
    ),
    (
        "render_table",
        "render html table rows from records",
        "Renders a tabular HTML view.",
    ),
    (
        "checksum",
        "compute a rolling checksum over file bytes",
        "Computes a file checksum.",
    ),
];

fn representation(kind: RepresentationKind, hash: String, text: &str) -> NewRepresentation {
    NewRepresentation {
        kind,
        content_hash: hash,
        content: Some(text.to_owned()),
        origin: RepresentationOrigin::default(),
    }
}

fn unit(name: &str, body: &str, docs: &str, start: usize) -> NewCodeUnit {
    NewCodeUnit {
        entity_id: EntityId::new(format!("ent-{name}")),
        entity_version_id: EntityVersionId::new(format!("ver-{name}")),
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
        normalized_body_hash: format!("body-{name}"),
        representations: vec![
            representation(RepresentationKind::FullSource, format!("src-{name}"), body),
            representation(
                RepresentationKind::Implementation,
                format!("impl-{name}"),
                body,
            ),
            representation(
                RepresentationKind::Documentation,
                format!("docs-{name}"),
                docs,
            ),
        ],
    }
}

fn build_index() -> (SearchIndex, HashEmbedder, HashEmbedder) {
    let db = open_in_memory().unwrap();
    let project_id = db.upsert_project("main", "memory://fixture").unwrap();
    let file_id = db
        .upsert_file(&NewFile {
            project_id,
            source_document_id: "lib.rs".into(),
            source_revision: "r1".into(),
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
        .map(|(index, (name, body, docs))| unit(name, body, docs, index + 1))
        .collect();
    db.insert_units(file_id, &units).unwrap();

    let mut code_embedder = HashEmbedder::new(DIMS);
    let mut docs_embedder = HashEmbedder::new(DIMS * 2);
    let code_space = EmbeddingSpaceIdentity::new(
        "code",
        RepresentationKind::Implementation,
        code_embedder.contract().clone(),
    );
    let docs_space = EmbeddingSpaceIdentity::new(
        "docs",
        RepresentationKind::Documentation,
        docs_embedder.contract().clone(),
    );
    db.find_or_create_space(&code_space).unwrap();
    db.find_or_create_space(&docs_space).unwrap();

    let bodies: Vec<&str> = BODIES.iter().map(|(_, body, _)| *body).collect();
    let docs: Vec<&str> = BODIES.iter().map(|(_, _, docs)| *docs).collect();
    let mut code_vectors = code_embedder
        .embed(&EmbedRequest::documents(&bodies, None))
        .unwrap();
    let mut docs_vectors = docs_embedder
        .embed(&EmbedRequest::documents(&docs, None))
        .unwrap();
    for vector in &mut code_vectors {
        normalize_in_place(vector);
    }
    for vector in &mut docs_vectors {
        normalize_in_place(vector);
    }
    for (((name, _, _), code), docs) in BODIES
        .iter()
        .zip(code_vectors.iter())
        .zip(docs_vectors.iter())
    {
        db.insert_embedding(
            &EmbeddingSpaceId::new("code"),
            &format!("impl-{name}"),
            code,
        )
        .unwrap();
        db.insert_embedding(
            &EmbeddingSpaceId::new("docs"),
            &format!("docs-{name}"),
            docs,
        )
        .unwrap();
    }

    let snapshot = db.snapshot(&[]).unwrap();
    (
        SearchIndex::from_snapshot(snapshot).unwrap(),
        code_embedder,
        docs_embedder,
    )
}

fn call(from: &str, to: &str) -> codeindex_search::storage::RelationRecord {
    codeindex_search::storage::RelationRecord {
        from_entity_id: EntityId::new(format!("ent-{from}")),
        to_entity_id: Some(EntityId::new(format!("ent-{to}"))),
        to_symbol: to.into(),
        kind: "calls".into(),
        resolution: "exact".into(),
        provenance: "lsp:test".into(),
    }
}

#[test]
fn relations_expand_seeds_one_hop_in_both_directions() {
    let (mut index, _, _) = build_index();
    // parse_flags -> render_table, checksum -> parse_flags. Unit order in the
    // fixture: 0=parse_flags, 1=render_table, 2=checksum.
    index.relations = vec![
        call("parse_flags", "render_table"),
        call("checksum", "parse_flags"),
    ];

    // Seeding with parse_flags reaches its callee AND its caller.
    assert_eq!(index.expand_by_relations(&[0], 10), vec![1, 2]);
    // Seeds themselves are never re-emitted, and the cap is honored.
    assert_eq!(index.expand_by_relations(&[0, 1], 10), vec![2]);
    assert_eq!(index.expand_by_relations(&[0], 1), vec![1]);
    // Seeding the callee side walks the edge backwards to the caller.
    assert_eq!(index.expand_by_relations(&[1], 10), vec![0]);
    assert!(index.expand_by_relations(&[], 10).is_empty());

    // Unresolved targets (to_entity_id = None) are ignored gracefully.
    index
        .relations
        .push(codeindex_search::storage::RelationRecord {
            from_entity_id: EntityId::new("ent-render_table"),
            to_entity_id: None,
            to_symbol: "std::fmt::Display".into(),
            kind: "calls".into(),
            resolution: "heuristic".into(),
            provenance: "lsp:test".into(),
        });
    assert_eq!(index.expand_by_relations(&[1], 10), vec![0]);

    // Corroboration outranks discovery order: two call sites reaching
    // `checksum` beat the single caller edge that was discovered first.
    index.relations.push(call("render_table", "checksum"));
    index.relations.push(call("render_table", "checksum"));
    assert_eq!(index.expand_by_relations(&[1], 10), vec![2, 0]);
}

#[test]
fn load_populates_independent_spaces() {
    let (index, _, _) = build_index();
    assert_eq!(index.units.len(), 3);
    assert_eq!(index.spaces.len(), 2);
    assert_eq!(
        index
            .space(&EmbeddingSpaceId::new("code"))
            .unwrap()
            .vectors
            .len(),
        3
    );
    assert_eq!(
        index
            .space(&EmbeddingSpaceId::new("docs"))
            .unwrap()
            .identity
            .model
            .native_dimensions,
        DIMS * 2
    );
}

#[test]
fn search_text_uses_the_selected_space_model() {
    let (index, mut code_embedder, mut docs_embedder) = build_index();
    let filter = WhereFilter::default();
    let code = index
        .search_text(
            &mut code_embedder,
            "parse command line flags",
            None,
            &EmbeddingSpaceId::new("code"),
            &filter,
            10,
        )
        .unwrap();
    assert_eq!(index.units[code.hits[0].index].name, "parse_flags");

    let docs = index
        .search_text(
            &mut docs_embedder,
            "file checksum",
            None,
            &EmbeddingSpaceId::new("docs"),
            &filter,
            10,
        )
        .unwrap();
    assert_eq!(index.units[docs.hits[0].index].name, "checksum");

    let error = index
        .search_text(
            &mut code_embedder,
            "anything",
            None,
            &EmbeddingSpaceId::new("docs"),
            &filter,
            10,
        )
        .unwrap_err()
        .to_string();
    assert!(error.contains("model contract"));
}

#[test]
fn weighted_rrf_fuses_incompatible_vector_spaces() {
    let (index, mut code_embedder, mut docs_embedder) = build_index();
    let mut code_query = code_embedder
        .embed(&EmbedRequest::queries(&["rolling checksum bytes"], None))
        .unwrap()
        .pop()
        .unwrap();
    let mut docs_query = docs_embedder
        .embed(&EmbedRequest::queries(&["file checksum"], None))
        .unwrap()
        .pop()
        .unwrap();
    normalize_in_place(&mut code_query);
    normalize_in_place(&mut docs_query);
    let code_id = EmbeddingSpaceId::new("code");
    let docs_id = EmbeddingSpaceId::new("docs");
    let fused = index
        .search_vectors_fused(
            &[
                SpaceVectorQuery {
                    space_id: &code_id,
                    vector: &code_query,
                    weight: 1.0,
                },
                SpaceVectorQuery {
                    space_id: &docs_id,
                    vector: &docs_query,
                    weight: 1.0,
                },
            ],
            &WhereFilter::default(),
            10,
            60,
        )
        .unwrap();
    assert_eq!(index.units[fused.hits[0].index].name, "checksum");
    assert_eq!(fused.hits[0].contributions.len(), 2);
}

#[test]
fn selector_round_trips() {
    let (index, _, _) = build_index();
    let id = codeindex_query::unit_id(&index.units[0]);
    assert_eq!(resolve_selector(&index.units, &id).unwrap(), 0);
}

/// A hash embedder whose contract is instruction-aware (Qwen3-shaped):
/// queries render through `Instruct:/Query:`, documents embed raw.
struct InstructionEmbedder {
    inner: HashEmbedder,
    contract: codeindex_core::ModelContract,
}

impl InstructionEmbedder {
    fn new(dimensions: usize) -> Self {
        let inner = HashEmbedder::new(dimensions);
        let mut contract = inner.contract().clone();
        contract.prompts = codeindex_core::PromptContract::QueryInstruction {
            query_template: "Instruct: {instruction}\nQuery:{query}".into(),
            default_instruction: None,
        };
        Self { inner, contract }
    }
}

impl EmbeddingBackend for InstructionEmbedder {
    fn contract(&self) -> &codeindex_core::ModelContract {
        &self.contract
    }
    fn execution(&self) -> &codeindex_core::ExecutionInfo {
        self.inner.execution()
    }
    fn embed(&mut self, request: &EmbedRequest<'_>) -> anyhow::Result<Vec<Vec<f32>>> {
        let rendered = codeindex_embedding::render_inputs(&self.contract, request)?;
        let rendered_refs: Vec<&str> = rendered.iter().map(|text| text.as_ref()).collect();
        // Re-issue as an already-rendered symmetric request against the inner
        // hash model so the instruction actually reaches the vector.
        self.inner
            .embed(&EmbedRequest::documents(&rendered_refs, None))
    }
}

#[test]
fn instruction_tasked_queries_hit_an_uninstructed_document_index() {
    use codeindex_core::EmbeddingTask;

    let db = open_in_memory().unwrap();
    let project_id = db.upsert_project("main", "memory://fixture").unwrap();
    let file_id = db
        .upsert_file(&NewFile {
            project_id,
            source_document_id: "lib.rs".into(),
            source_revision: "r1".into(),
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
        .map(|(index, (name, body, docs))| unit(name, body, docs, index + 1))
        .collect();
    db.insert_units(file_id, &units).unwrap();

    let mut embedder = InstructionEmbedder::new(DIMS);
    let space = EmbeddingSpaceIdentity::new(
        "code",
        RepresentationKind::Implementation,
        embedder.contract().clone(),
    );
    db.find_or_create_space(&space).unwrap();
    let bodies: Vec<&str> = BODIES.iter().map(|(_, body, _)| *body).collect();
    let mut vectors = embedder
        .embed(&EmbedRequest::documents(&bodies, None))
        .unwrap();
    for vector in &mut vectors {
        normalize_in_place(vector);
    }
    for ((name, _, _), vector) in BODIES.iter().zip(vectors.iter()) {
        db.insert_embedding(
            &EmbeddingSpaceId::new("code"),
            &format!("impl-{name}"),
            vector,
        )
        .unwrap();
    }
    let index = SearchIndex::from_snapshot(db.snapshot(&[]).unwrap()).unwrap();

    // Without a task the model has no default instruction: must error loudly.
    let error = index
        .search_text(
            &mut embedder,
            "parse flags",
            None,
            &EmbeddingSpaceId::new("code"),
            &WhereFilter::default(),
            10,
        )
        .unwrap_err()
        .to_string();
    assert!(error.contains("task instruction"), "got: {error}");

    // With a task instruction the query renders differently from documents
    // yet still searches the unchanged document index.
    let task = EmbeddingTask::new("code-search", "retrieve relevant code");
    let results = index
        .search_text(
            &mut embedder,
            "parse command line flags",
            Some(&task),
            &EmbeddingSpaceId::new("code"),
            &WhereFilter::default(),
            10,
        )
        .unwrap();
    assert_eq!(index.units[results.hits[0].index].name, "parse_flags");
}
