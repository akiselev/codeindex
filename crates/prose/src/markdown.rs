//! Heading-structured sectioning for Markdown.
//!
//! Deliberately not a full Markdown parser. The only structure this needs is
//! ATX headings (`#`..`######`), and the only things that must not be mistaken
//! for one are fenced code blocks and YAML frontmatter — both of which can
//! contain lines starting with `#`.

use std::ops::Range;

#[derive(Debug, Clone)]
pub(crate) struct Section {
    pub title: String,
    /// Nearest enclosing heading of a shallower level, if any.
    pub scope: Option<String>,
    /// Heading line through the byte before the next heading of any level.
    pub range: Range<usize>,
}

struct Heading {
    level: usize,
    title: String,
    start: usize,
}

fn parse_heading(line: &str) -> Option<(usize, String)> {
    let trimmed = line.trim_start();
    if !trimmed.starts_with('#') {
        return None;
    }
    let hashes = trimmed.chars().take_while(|c| *c == '#').count();
    if hashes == 0 || hashes > 6 {
        return None;
    }
    let rest = &trimmed[hashes..];
    // ATX requires a space after the hashes; `#hashtag` is not a heading.
    if !rest.starts_with(' ') && !rest.is_empty() {
        return None;
    }
    let title = rest
        .trim()
        .trim_end_matches('#')
        .trim()
        .trim_matches('*')
        .trim()
        .to_string();
    if title.is_empty() {
        return None;
    }
    Some((hashes, title))
}

/// True when a section carries no prose of its own — its body after the
/// heading line is blank, because every child heading follows immediately.
pub(crate) fn is_container(source: &str, section: &Section) -> bool {
    let text = &source[section.range.clone()];
    match text.split_once('\n') {
        Some((_, body)) => body.trim().is_empty(),
        // Last line of the file with no trailing newline: heading only.
        None => true,
    }
}

/// Split a Markdown document into heading sections.
///
/// Content before the first heading (frontmatter, a lead paragraph) becomes a
/// leading section named `preamble`, so no byte is unindexed.
pub(crate) fn sections(source: &str) -> Vec<Section> {
    let mut headings: Vec<Heading> = Vec::new();
    let mut in_fence = false;
    let mut fence_marker = "";
    let mut in_frontmatter = false;
    let mut offset = 0usize;

    for (index, line) in source.split_inclusive('\n').enumerate() {
        let bare = line.trim_end_matches(['\n', '\r']);
        let trimmed = bare.trim_start();

        // YAML frontmatter: only when `---` is the very first line.
        if index == 0 && bare.trim() == "---" {
            in_frontmatter = true;
            offset += line.len();
            continue;
        }
        if in_frontmatter {
            if bare.trim() == "---" || bare.trim() == "..." {
                in_frontmatter = false;
            }
            offset += line.len();
            continue;
        }

        // Fenced code: ``` or ~~~, closed by the same marker.
        if in_fence {
            if trimmed.starts_with(fence_marker) {
                in_fence = false;
            }
            offset += line.len();
            continue;
        }
        if trimmed.starts_with("```") {
            in_fence = true;
            fence_marker = "```";
            offset += line.len();
            continue;
        }
        if trimmed.starts_with("~~~") {
            in_fence = true;
            fence_marker = "~~~";
            offset += line.len();
            continue;
        }

        if let Some((level, title)) = parse_heading(bare) {
            headings.push(Heading {
                level,
                title,
                start: offset,
            });
        }
        offset += line.len();
    }

    if headings.is_empty() {
        if source.trim().is_empty() {
            return Vec::new();
        }
        return vec![Section {
            title: "preamble".to_string(),
            scope: None,
            range: 0..source.len(),
        }];
    }

    let mut out = Vec::new();
    if headings[0].start > 0 && !source[..headings[0].start].trim().is_empty() {
        out.push(Section {
            title: "preamble".to_string(),
            scope: None,
            range: 0..headings[0].start,
        });
    }

    for (i, heading) in headings.iter().enumerate() {
        let end = headings
            .get(i + 1)
            .map(|next| next.start)
            .unwrap_or(source.len());
        let scope = headings[..i]
            .iter()
            .rev()
            .find(|candidate| candidate.level < heading.level)
            .map(|candidate| candidate.title.clone());
        out.push(Section {
            title: heading.title.clone(),
            scope,
            range: heading.start..end,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frontmatter_hash_is_not_a_heading() {
        let source = "---\ntitle: a # b\n---\n\n# Real\n\nbody\n";
        let found = sections(source);
        let titles: Vec<&str> = found.iter().map(|s| s.title.as_str()).collect();
        assert_eq!(titles, vec!["preamble", "Real"]);
    }

    #[test]
    fn fenced_code_comments_are_not_headings() {
        let source = "# Top\n\n```sh\n# not a heading\necho hi\n```\n\n## After\n\nx\n";
        let titles: Vec<String> = sections(source).into_iter().map(|s| s.title).collect();
        assert_eq!(titles, vec!["Top", "After"]);
    }

    #[test]
    fn hashtag_without_space_is_not_a_heading() {
        let source = "# Top\n\n#nothashtag text\n";
        assert_eq!(sections(source).len(), 1);
    }

    #[test]
    fn scope_is_nearest_shallower_heading() {
        let source = "# A\n\nx\n\n## B\n\ny\n\n### C\n\nz\n\n## D\n\nw\n";
        let found = sections(source);
        let by = |name: &str| {
            found
                .iter()
                .find(|s| s.title == name)
                .unwrap()
                .scope
                .clone()
        };
        assert_eq!(by("A"), None);
        assert_eq!(by("B"), Some("A".into()));
        assert_eq!(by("C"), Some("B".into()));
        assert_eq!(by("D"), Some("A".into()));
    }

    #[test]
    fn sections_tile_the_document() {
        let source = "---\nk: v\n---\n\n# A\n\nbody\n\n## B\n\nmore\n";
        let found = sections(source);
        let mut cursor = 0;
        for section in &found {
            assert_eq!(section.range.start, cursor);
            cursor = section.range.end;
        }
        assert_eq!(cursor, source.len());
    }

    #[test]
    fn no_headings_yields_single_preamble() {
        let found = sections("just a paragraph\n\nand another\n");
        assert_eq!(found.len(), 1);
        assert_eq!(found[0].title, "preamble");
    }

    #[test]
    fn container_sections_are_detected() {
        let source = "# Top\n\n## Findings\n\n### Detail\n\nreal prose here\n";
        let found = sections(source);
        let by = |n: &str| found.iter().find(|s| s.title == n).unwrap().clone();
        assert!(
            is_container(source, &by("Findings")),
            "Findings has no own prose"
        );
        assert!(!is_container(source, &by("Detail")), "Detail has prose");
        assert!(is_container(source, &by("Top")));
    }

    #[test]
    fn empty_source_yields_nothing() {
        assert!(sections("   \n\n").is_empty());
    }
}
