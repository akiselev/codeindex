//! Grammar-free extraction of prose documents into index units.
//!
//! `codeindex-tree-sitter` turns source code into [`ExtractedEntity`] values via
//! compiled grammars. This crate does the same job for documents that have no
//! grammar — Markdown notes, OCR'd scans, plain text — so the rest of the
//! pipeline (identity, representations, embedding, search) is unchanged.
//!
//! Design and the survey of chunking strategies behind it:
//! `docs/prose-chunking-plan.md`.
//!
//! Dependency boundary: this crate sees only `codeindex-core`. No grammars, no
//! storage — the same boundary `codeindex-storage` keeps.

mod chunk;
mod markdown;

pub use chunk::{ChunkOptions, chunk_ranges};

use anyhow::Result;
use codeindex_core::{
    EntityKind, ExtractedEntity, LanguageId, Representation, RepresentationKind, SourceSpan,
};
use sha2::{Digest, Sha256};

/// A document format handled without a grammar.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProseFormat {
    /// Heading-structured text. Sections become units.
    Markdown,
    /// Unstructured text: OCR output, transcripts, plain notes.
    Text,
}

impl ProseFormat {
    /// Stable language id persisted with the document.
    pub fn id(self) -> &'static str {
        match self {
            Self::Markdown => "markdown",
            Self::Text => "text",
        }
    }

    pub fn from_id(id: &str) -> Option<Self> {
        match id {
            "markdown" => Some(Self::Markdown),
            "text" => Some(Self::Text),
            _ => None,
        }
    }

    /// Resolve a lowercase file extension to a format.
    pub fn from_extension(extension: &str) -> Option<Self> {
        match extension {
            "md" | "markdown" | "mdown" | "mkd" => Some(Self::Markdown),
            "txt" | "text" => Some(Self::Text),
            _ => None,
        }
    }

    pub fn all() -> [ProseFormat; 2] {
        [Self::Markdown, Self::Text]
    }
}

/// Extraction knobs for prose.
#[derive(Debug, Clone, Copy)]
pub struct ProseOptions {
    pub chunking: ChunkOptions,
    /// A Markdown section longer than this is fragmented into child units,
    /// mirroring Track 2 of `oversized-units-plan.md` (fragment, never
    /// truncate). The section itself is still emitted so the parent view works.
    pub section_fragment_threshold: usize,
}

impl Default for ProseOptions {
    fn default() -> Self {
        Self::with_target(
            ChunkOptions::default().target_chars,
            ChunkOptions::default().overlap_chars,
        )
    }
}

impl ProseOptions {
    /// Build from the two knobs that actually change retrieval behaviour.
    ///
    /// `section_fragment_threshold` is derived as **twice the target** rather
    /// than exposed separately: fragmenting a section shorter than that yields
    /// a single chunk identical to the section, so the extra unit is pure cost.
    pub fn with_target(target_chars: usize, overlap_chars: usize) -> Self {
        let target = target_chars.max(64);
        Self {
            chunking: ChunkOptions {
                target_chars: target,
                overlap_chars: overlap_chars.min(target / 2),
                ..ChunkOptions::default()
            },
            section_fragment_threshold: target.saturating_mul(2),
        }
    }
}

fn sha256_hex(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    hex::encode(hasher.finalize())
}

/// Whitespace-collapsed hash, so trivial reflow does not churn identity.
fn normalized_hash(text: &str) -> String {
    sha256_hex(&text.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// 1-based line number. `offsets` holds the start byte of lines 2..n, so the
/// line is one more than the count of offsets at or below `byte`.
fn line_of(offsets: &[usize], byte: usize) -> usize {
    match offsets.binary_search(&byte) {
        Ok(index) => index + 2,
        Err(index) => index + 1,
    }
}

/// Byte offset of the first character of each line after the first.
fn line_offsets(source: &str) -> Vec<usize> {
    source
        .char_indices()
        .filter(|(_, c)| *c == '\n')
        .map(|(i, _)| i + 1)
        .collect()
}

fn make_entity(
    format: ProseFormat,
    kind: &str,
    name: String,
    scope: Option<String>,
    source: &str,
    range: std::ops::Range<usize>,
    offsets: &[usize],
) -> ExtractedEntity {
    let text = &source[range.clone()];
    let span = SourceSpan::new(
        range.start,
        range.end,
        line_of(offsets, range.start),
        line_of(offsets, range.end.saturating_sub(1)),
    );
    let full_hash = sha256_hex(text);
    ExtractedEntity {
        language: LanguageId::new(format.id()),
        kind: EntityKind::Other(kind.to_string()),
        name,
        scope,
        span,
        body_span: Some(span),
        // Prose has no AST; the node count that gates code heuristics is
        // meaningless here and is reported as zero.
        body_node_count: 0,
        source_hash: full_hash.clone(),
        normalized_body_hash: normalized_hash(text),
        representations: vec![
            // FullSource keeps the untruncated unit so full-text lexical search
            // works (Track 1 of oversized-units-plan.md).
            Representation::new(RepresentationKind::FullSource, text, full_hash.clone()),
            // Body is what gets embedded.
            Representation::new(RepresentationKind::Body, text, full_hash),
        ],
    }
}

/// Turn one prose document into index units.
///
/// Markdown yields one unit per content-bearing heading section, plus `chunk`
/// fragments for sections over `section_fragment_threshold`. Plain text yields
/// recursive chunks.
///
/// Coverage differs by format, deliberately. **Text** is fully tiled: every
/// byte lands in exactly one chunk. **Markdown** covers every byte *except* the
/// heading lines of pure container sections, which are skipped as retrieval
/// units — their text survives as the `scope` of the sections beneath them.
pub fn extract_prose(
    format: ProseFormat,
    source: &str,
    options: &ProseOptions,
) -> Result<Vec<ExtractedEntity>> {
    let offsets = line_offsets(source);
    let mut entities = Vec::new();

    match format {
        ProseFormat::Markdown => {
            let sections = markdown::sections(source);
            for section in sections {
                let title = section.title.clone();
                let scope = section.scope.clone();
                // A heading whose content lives entirely in sub-headings is a
                // structural container, not a passage. Embedding "## Findings"
                // on its own produces a vector that matches everything and
                // means nothing; on this corpus 5% of sections were these.
                // The heading is not lost — it survives as its children's
                // `scope`.
                if markdown::is_container(source, &section) {
                    continue;
                }
                entities.push(make_entity(
                    format,
                    "section",
                    title.clone(),
                    scope.clone(),
                    source,
                    section.range.clone(),
                    &offsets,
                ));
                let len = section.range.end - section.range.start;
                if len > options.section_fragment_threshold {
                    let body = &source[section.range.clone()];
                    // Sections are semantically closed; do not overlap inside one.
                    let inner = ChunkOptions {
                        overlap_chars: 0,
                        ..options.chunking
                    };
                    let ranges = chunk_ranges(body, &inner);
                    if ranges.len() > 1 {
                        for (i, range) in ranges.into_iter().enumerate() {
                            let absolute =
                                section.range.start + range.start..section.range.start + range.end;
                            entities.push(make_entity(
                                format,
                                "chunk",
                                format!("{title} [{}]", i + 1),
                                Some(title.clone()),
                                source,
                                absolute,
                                &offsets,
                            ));
                        }
                    }
                }
            }
        }
        ProseFormat::Text => {
            for (i, range) in chunk_ranges(source, &options.chunking)
                .into_iter()
                .enumerate()
            {
                entities.push(make_entity(
                    format,
                    "chunk",
                    format!("chunk {}", i + 1),
                    None,
                    source,
                    range,
                    &offsets,
                ));
            }
        }
    }

    Ok(entities)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_target_derives_threshold_and_clamps_overlap() {
        let o = ProseOptions::with_target(1_000, 900);
        assert_eq!(o.chunking.target_chars, 1_000);
        assert_eq!(o.section_fragment_threshold, 2_000);
        // Overlap larger than half the target would make chunks barely advance.
        assert_eq!(o.chunking.overlap_chars, 500);
    }

    #[test]
    fn with_target_floors_absurdly_small_targets() {
        let o = ProseOptions::with_target(1, 0);
        assert!(o.chunking.target_chars >= 64);
    }

    #[test]
    fn default_matches_with_target_defaults() {
        let d = ProseOptions::default();
        assert_eq!(d.chunking.target_chars, 1_200);
        assert_eq!(d.chunking.overlap_chars, 150);
        assert_eq!(d.section_fragment_threshold, 2_400);
    }

    #[test]
    fn extensions_resolve() {
        assert_eq!(
            ProseFormat::from_extension("md"),
            Some(ProseFormat::Markdown)
        );
        assert_eq!(ProseFormat::from_extension("txt"), Some(ProseFormat::Text));
        assert_eq!(ProseFormat::from_extension("rs"), None);
    }

    #[test]
    fn markdown_sections_become_units() {
        let source = "---\ntitle: x\n---\n\n# Note\n\nintro line\n\n## Findings\n\nfact one\n\n## Assessment\n\nconclusion\n";
        let units = extract_prose(ProseFormat::Markdown, source, &ProseOptions::default()).unwrap();
        let names: Vec<&str> = units.iter().map(|u| u.name.as_str()).collect();
        assert!(names.contains(&"Findings"), "got {names:?}");
        assert!(names.contains(&"Assessment"), "got {names:?}");
    }

    #[test]
    fn section_scope_is_the_parent_heading() {
        let source = "# Top\n\nlead\n\n## Child\n\nbody text here\n";
        let units = extract_prose(ProseFormat::Markdown, source, &ProseOptions::default()).unwrap();
        let child = units.iter().find(|u| u.name == "Child").unwrap();
        assert_eq!(child.scope.as_deref(), Some("Top"));
    }

    #[test]
    fn plain_text_covers_every_byte() {
        let source = "alpha beta\n\ngamma delta\n\nepsilon zeta\n";
        let options = ProseOptions {
            chunking: ChunkOptions {
                target_chars: 12,
                min_chars: 4,
                overlap_chars: 0,
            },
            ..ProseOptions::default()
        };
        let units = extract_prose(ProseFormat::Text, source, &options).unwrap();
        let mut cursor = 0;
        for unit in &units {
            assert_eq!(unit.span.start_byte, cursor);
            cursor = unit.span.end_byte;
        }
        assert_eq!(cursor, source.len());
    }

    #[test]
    fn oversized_section_emits_parent_and_fragments() {
        let body = "sentence number one here. ".repeat(200); // ~5 KB
        let source = format!("## Big\n\n{body}");
        let units =
            extract_prose(ProseFormat::Markdown, &source, &ProseOptions::default()).unwrap();
        assert!(
            units
                .iter()
                .any(|u| u.name == "Big"
                    && matches!(&u.kind, EntityKind::Other(k) if k == "section"))
        );
        let fragments: Vec<_> = units
            .iter()
            .filter(|u| matches!(&u.kind, EntityKind::Other(k) if k == "chunk"))
            .collect();
        assert!(
            fragments.len() > 1,
            "expected fragments, got {}",
            fragments.len()
        );
        assert!(fragments.iter().all(|f| f.scope.as_deref() == Some("Big")));
    }

    #[test]
    fn representations_are_full_source_and_body_only() {
        let units = extract_prose(
            ProseFormat::Text,
            "some prose text",
            &ProseOptions::default(),
        )
        .unwrap();
        let kinds: Vec<&RepresentationKind> =
            units[0].representations.iter().map(|r| &r.kind).collect();
        assert!(kinds.contains(&&RepresentationKind::FullSource));
        assert!(kinds.contains(&&RepresentationKind::Body));
        assert_eq!(kinds.len(), 2);
    }

    #[test]
    fn line_numbers_are_one_based() {
        let source = "first\n\nsecond paragraph\n";
        let units = extract_prose(ProseFormat::Text, source, &ProseOptions::default()).unwrap();
        assert_eq!(units[0].span.start_line, 1);
    }

    #[test]
    fn empty_document_yields_no_units() {
        let units = extract_prose(ProseFormat::Text, "", &ProseOptions::default()).unwrap();
        assert!(units.is_empty());
    }
}
