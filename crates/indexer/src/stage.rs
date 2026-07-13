//! Pure-ish preparation of one document for durable checkpointing.

use anyhow::{Context, Result};
use codeindex_sqlite::{Db, NewFile, StagedDocumentPayload, StagedReference};
use codeindex_tree_sitter::{ExtractOptions, LanguageRegistry, extract_references, extract_units};

use crate::{
    IndexSettings, RepresentationEnricher, SourceDocument, apply_enrichers, apply_retention,
    assign_identity, complete_representations,
};

pub(crate) struct PrepareDocument<'a> {
    pub db: &'a Db,
    pub settings: &'a IndexSettings,
    pub project_label: &'a str,
    pub document: &'a SourceDocument,
    pub source: &'a str,
    pub source_hash: &'a str,
    pub generation: i64,
    pub input_fingerprint: &'a str,
    pub enrichers: &'a [&'a dyn RepresentationEnricher],
}

pub(crate) fn prepare_document(input: PrepareDocument<'_>) -> Result<StagedDocumentPayload> {
    let definition = LanguageRegistry::global()
        .get(&input.document.language_id)
        .with_context(|| format!("unknown language {}", input.document.language_id))?;
    let extraction = ExtractOptions {
        body_node_count_threshold: input.settings.body_node_count_threshold,
        max_body_chars: input.settings.max_body_chars,
    };
    let mut entities = extract_units(definition, input.source, &extraction).with_context(|| {
        format!(
            "failed to extract units from {}",
            input.document.relative_path
        )
    })?;
    for entity in &mut entities {
        complete_representations(input.source, entity);
        apply_enrichers(input.document, input.source, entity, input.enrichers)?;
    }
    let raw_references = extract_references(definition, input.source).with_context(|| {
        format!(
            "failed to extract references from {}",
            input.document.relative_path
        )
    })?;

    let existing = match input.db.get_project(input.project_label)? {
        Some(project) => input
            .db
            .get_file_by_source_id(project.id, &input.document.id)?,
        None => None,
    };
    let prior = match existing {
        Some(ref file) => input.db.list_units_for_file(file.id)?,
        None => Vec::new(),
    };
    let mut units = assign_identity(
        input.project_label,
        &input.document.id,
        &prior,
        entities,
        input.generation,
    );
    apply_retention(&mut units, input.settings.retention);

    let mut references = Vec::new();
    for reference in raw_references {
        let mut best: Option<usize> = None;
        for (ordinal, unit) in units.iter().enumerate() {
            if unit.start_byte <= reference.start_byte && reference.start_byte < unit.end_byte {
                match best {
                    Some(previous) if units[previous].start_byte >= unit.start_byte => {}
                    _ => best = Some(ordinal),
                }
            }
        }
        if let Some(caller_unit_ordinal) = best {
            references.push(StagedReference {
                caller_unit_ordinal,
                callee_symbol: reference.callee_symbol,
                call_snippet: reference.call_snippet,
                start_line: reference.start_line as i64,
            });
        }
    }

    Ok(StagedDocumentPayload {
        payload_schema_version: codeindex_sqlite::index_runs::STAGED_PAYLOAD_SCHEMA_VERSION,
        file: NewFile {
            // A project does not become live until publication assigns its id.
            project_id: 0,
            source_document_id: input.document.id.clone(),
            source_revision: input.document.revision.opaque.clone(),
            relative_path: input.document.relative_path.clone(),
            language_id: input.document.language_id.clone(),
            mtime_ns: input.document.revision.modified_ns.unwrap_or_default(),
            size: input.document.revision.size.unwrap_or_default() as i64,
            source_hash: input.source_hash.to_string(),
        },
        units,
        references,
        input_fingerprint: input.input_fingerprint.to_string(),
        source_hash: input.source_hash.to_string(),
    })
}
