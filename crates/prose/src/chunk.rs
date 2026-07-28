//! Recursive, separator-hierarchy chunking for prose.
//!
//! The strategy is method (4) from `docs/prose-chunking-plan.md`: try the
//! strongest available boundary first and only descend to a weaker one when a
//! piece is still oversized. On well-formed prose this behaves like paragraph
//! splitting; on OCR garbage with no paragraph structure it degrades to a
//! character window without ever failing.

use std::ops::Range;

/// Chunking knobs. Sizes are in **bytes**, matching `SourceSpan`.
#[derive(Debug, Clone, Copy)]
pub struct ChunkOptions {
    /// Preferred upper bound for one chunk.
    pub target_chars: usize,
    /// A chunk is never emitted below this unless it is the whole input or the
    /// trailing remainder; prevents a flood of one-line units on ragged text.
    pub min_chars: usize,
    /// Trailing bytes of the previous chunk repeated at the head of the next.
    /// Applies to unstructured text only; see the plan for why sections do not
    /// overlap.
    pub overlap_chars: usize,
}

impl Default for ChunkOptions {
    fn default() -> Self {
        Self {
            target_chars: 1_200,
            min_chars: 200,
            overlap_chars: 150,
        }
    }
}

/// Separator strength, strongest first.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Level {
    Paragraph,
    Line,
    Sentence,
    Word,
}

impl Level {
    fn next(self) -> Option<Self> {
        match self {
            Self::Paragraph => Some(Self::Line),
            Self::Line => Some(Self::Sentence),
            Self::Sentence => Some(Self::Word),
            Self::Word => None,
        }
    }
}

/// Abbreviations after which a period does **not** end a sentence. Kept small
/// and deliberately legal/civic-flavoured: this corpus is full of `v.`, `No.`,
/// `Cal. App. 4th`, `F. Supp. 2d`. This is a heuristic, not a parser — when it
/// is wrong the cost is a slightly odd boundary, never a dropped byte.
const ABBREVIATIONS: &[&str] = &[
    "v", "vs", "no", "nos", "cal", "app", "supp", "stat", "stats", "ch", "art", "sec", "secs",
    "para", "pp", "p", "fig", "ex", "inc", "corp", "co", "ltd", "dept", "govt", "st", "ave", "dr",
    "mr", "mrs", "ms", "jr", "sr", "hon", "id", "cf", "eg", "ie", "etc", "et", "al", "u", "s", "f",
    "d", "n", "r",
];

fn is_abbreviation(text: &str, period: usize) -> bool {
    let head = &text[..period];
    // Step past the whole delimiter char: `i + 1` lands inside a multibyte
    // char, which OCR text ("Section J¡.(f)") supplies in quantity.
    let start = head
        .char_indices()
        .rev()
        .find(|(_, c)| !c.is_alphanumeric())
        .map(|(i, c)| i + c.len_utf8())
        .unwrap_or(0);
    let word = &head[start..];
    if word.is_empty() {
        return false;
    }
    // A single capital letter ("J." in a name) or a known abbreviation.
    if word.chars().count() == 1 && word.chars().all(|c| c.is_ascii_uppercase()) {
        return true;
    }
    let lower = word.to_ascii_lowercase();
    ABBREVIATIONS.contains(&lower.as_str())
}

/// Byte offsets just past each boundary of the given strength.
fn boundaries(text: &str, level: Level) -> Vec<usize> {
    let bytes = text.as_bytes();
    let mut out = Vec::new();
    match level {
        Level::Paragraph => {
            let mut i = 0;
            while i + 1 < bytes.len() {
                if bytes[i] == b'\n' {
                    // Consume a run of whitespace containing a second newline.
                    let mut j = i + 1;
                    let mut newlines = 1;
                    while j < bytes.len() && (bytes[j] as char).is_whitespace() {
                        if bytes[j] == b'\n' {
                            newlines += 1;
                        }
                        j += 1;
                    }
                    if newlines >= 2 {
                        out.push(j);
                        i = j;
                        continue;
                    }
                }
                i += 1;
            }
        }
        Level::Line => {
            for (i, b) in bytes.iter().enumerate() {
                if *b == b'\n' && i + 1 < bytes.len() {
                    out.push(i + 1);
                }
            }
        }
        Level::Sentence => {
            for (i, b) in bytes.iter().enumerate() {
                if !matches!(*b, b'.' | b'?' | b'!') {
                    continue;
                }
                if *b == b'.' && is_abbreviation(text, i) {
                    continue;
                }
                // Require following whitespace so "3.5" and "No.19BB" survive.
                let mut j = i + 1;
                if j < bytes.len() && (bytes[j] == b'"' || bytes[j] == b')') {
                    j += 1;
                }
                if j < bytes.len() && (bytes[j] as char).is_whitespace() {
                    while j < bytes.len() && (bytes[j] as char).is_whitespace() {
                        j += 1;
                    }
                    if j < bytes.len() {
                        out.push(j);
                    }
                }
            }
        }
        Level::Word => {
            for (i, b) in bytes.iter().enumerate() {
                if (*b as char).is_whitespace() && i + 1 < bytes.len() {
                    let mut j = i + 1;
                    while j < bytes.len() && (bytes[j] as char).is_whitespace() {
                        j += 1;
                    }
                    if j < bytes.len() {
                        out.push(j);
                    }
                }
            }
        }
    }
    out.dedup();
    out
}

/// Greedily pack `text` into ranges no larger than `target`, splitting at the
/// strongest separator that yields progress and descending when it does not.
fn pack(
    text: &str,
    base: usize,
    level: Level,
    options: &ChunkOptions,
    out: &mut Vec<Range<usize>>,
) {
    if text.is_empty() {
        return;
    }
    if text.len() <= options.target_chars {
        out.push(base..base + text.len());
        return;
    }

    let points = boundaries(text, level);
    if points.is_empty() {
        match level.next() {
            Some(next) => pack(text, base, next, options, out),
            // No separator of any strength: hard-cut on a char boundary.
            None => {
                let mut cursor = 0;
                while cursor < text.len() {
                    let mut end = (cursor + options.target_chars).min(text.len());
                    while end < text.len() && !text.is_char_boundary(end) {
                        end += 1;
                    }
                    out.push(base + cursor..base + end);
                    cursor = end;
                }
            }
        }
        return;
    }

    let mut start = 0usize;
    let mut last_fit = None;
    for &point in &points {
        if point <= start {
            continue;
        }
        if point - start <= options.target_chars {
            last_fit = Some(point);
            continue;
        }
        match last_fit.take() {
            Some(cut) => {
                out.push(base + start..base + cut);
                start = cut;
                // Reconsider this point against the new start.
                if point - start <= options.target_chars {
                    last_fit = Some(point);
                }
            }
            None => {
                // A single span between separators exceeds target: descend.
                let span = &text[start..point];
                match level.next() {
                    Some(next) => pack(span, base + start, next, options, out),
                    None => out.push(base + start..base + point),
                }
                start = point;
            }
        }
    }
    if start < text.len() {
        let tail = &text[start..];
        if tail.len() <= options.target_chars {
            out.push(base + start..base + text.len());
        } else {
            match level.next() {
                Some(next) => pack(tail, base + start, next, options, out),
                None => out.push(base + start..base + text.len()),
            }
        }
    }
}

/// Split `text` into byte ranges, merging runts forward and applying overlap.
pub fn chunk_ranges(text: &str, options: &ChunkOptions) -> Vec<Range<usize>> {
    let mut raw = Vec::new();
    pack(text, 0, Level::Paragraph, options, &mut raw);

    // Merge anything below `min_chars` into its predecessor when that keeps the
    // result within a modest tolerance of target. Prevents one-word units.
    let mut merged: Vec<Range<usize>> = Vec::with_capacity(raw.len());
    for range in raw {
        if range.start >= range.end {
            continue;
        }
        let len = range.end - range.start;
        match merged.last_mut() {
            Some(prev)
                if len < options.min_chars
                    && (range.end - prev.start) <= options.target_chars * 2 =>
            {
                prev.end = range.end;
            }
            _ => merged.push(range),
        }
    }

    // The loop above can only merge *backwards*, so a runt in first position
    // survives it. Fold it forward into its successor instead.
    if merged.len() > 1 && (merged[0].end - merged[0].start) < options.min_chars {
        let head = merged.remove(0);
        merged[0].start = head.start;
    }

    if options.overlap_chars == 0 || merged.len() < 2 {
        return merged;
    }
    // Extend each chunk backwards into its predecessor.
    let mut with_overlap = Vec::with_capacity(merged.len());
    for (i, range) in merged.iter().enumerate() {
        if i == 0 {
            with_overlap.push(range.clone());
            continue;
        }
        let mut start = range.start.saturating_sub(options.overlap_chars);
        start = start.max(merged[i - 1].start);
        while start < range.start && !text.is_char_boundary(start) {
            start += 1;
        }
        with_overlap.push(start..range.end);
    }
    with_overlap
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts(target: usize, overlap: usize) -> ChunkOptions {
        ChunkOptions {
            target_chars: target,
            min_chars: 10,
            overlap_chars: overlap,
        }
    }

    #[test]
    fn short_text_is_one_chunk() {
        let ranges = chunk_ranges("a short note", &opts(1_000, 0));
        assert_eq!(ranges.len(), 1);
    }

    #[test]
    fn splits_on_paragraphs_before_lines() {
        let text = "alpha one two\nthree four\n\nbeta one two three\n\ngamma final line";
        let ranges = chunk_ranges(text, &opts(30, 0));
        assert!(
            ranges.len() >= 3,
            "expected paragraph splits, got {ranges:?}"
        );
        assert!(text[ranges[0].clone()].starts_with("alpha"));
    }

    #[test]
    fn covers_every_byte_without_overlap() {
        let text = "one two three four five six seven eight nine ten eleven twelve";
        let ranges = chunk_ranges(text, &opts(20, 0));
        let mut cursor = 0;
        for range in &ranges {
            assert_eq!(range.start, cursor, "gap or overlap at {range:?}");
            cursor = range.end;
        }
        assert_eq!(cursor, text.len());
    }

    #[test]
    fn legal_abbreviations_do_not_end_sentences() {
        // If "v." or "No." split, these become separate chunks.
        let text = "City of South Pasadena v. Slater is No. 98-6996 and it matters. Next.";
        let points = boundaries(text, Level::Sentence);
        for &p in &points {
            let head = &text[..p];
            assert!(
                !head.trim_end().ends_with("v.") && !head.trim_end().ends_with("No."),
                "split after abbreviation at {p}: {head:?}"
            );
        }
    }

    #[test]
    fn ocr_noise_never_panics_and_never_drops_bytes() {
        let text = "Case No.19BBCV00118 Section J\u{a1}.(f) 54956. 9( d)( 2) \
                    City of Gardena v. Los A ngeles Regional Water Qu ality";
        let ranges = chunk_ranges(text, &opts(25, 0));
        let joined: String = ranges.iter().map(|r| &text[r.clone()]).collect();
        assert_eq!(joined, text);
    }

    #[test]
    fn unbroken_run_is_hard_cut_on_char_boundaries() {
        let text = "é".repeat(500); // 1000 bytes, no separators at all
        let ranges = chunk_ranges(&text, &opts(100, 0));
        assert!(ranges.len() > 1);
        for range in &ranges {
            assert!(text.is_char_boundary(range.start));
            assert!(text.is_char_boundary(range.end));
        }
    }

    #[test]
    fn runt_first_chunk_is_folded_forward() {
        // A tiny leading paragraph must not survive as its own chunk; the
        // backwards-merge loop cannot fix position zero.
        let text = "hi\n\nnow a substantially longer paragraph that stands alone fine\n";
        let ranges = chunk_ranges(text, &opts(60, 0));
        assert!(
            ranges.iter().all(|r| r.end - r.start >= 10),
            "runt survived: {:?}",
            ranges.iter().map(|r| r.end - r.start).collect::<Vec<_>>()
        );
        assert_eq!(ranges[0].start, 0);
    }

    #[test]
    fn overlap_extends_backwards_only() {
        let text = "alpha one two\n\nbeta three four\n\ngamma five six\n\ndelta seven";
        let ranges = chunk_ranges(text, &opts(20, 8));
        for pair in ranges.windows(2) {
            assert!(pair[1].start < pair[0].end, "expected overlap");
            assert!(pair[1].end > pair[0].end, "chunks must advance");
        }
    }
}
