#![forbid(unsafe_code)]

use anyhow::{Context, Result, bail};
use codeindex_core::ModelContract;
use globset::{GlobBuilder, GlobMatcher};
use sha2::{Digest, Sha256};

pub trait UnitView {
    fn project_label(&self) -> &str;
    fn relative_path(&self) -> &str;
    fn language_id(&self) -> &str;
    fn kind(&self) -> &str;
    fn name(&self) -> &str;
    fn scope(&self) -> Option<&str>;
    fn start_byte(&self) -> usize;
    fn end_byte(&self) -> usize;
    fn start_line(&self) -> usize;
    fn end_line(&self) -> usize;
    fn body_node_count(&self) -> usize;
    fn normalized_body_hash(&self) -> &str;
}

pub fn unit_id(unit: &impl UnitView) -> String {
    let ingredients = [
        unit.project_label(),
        unit.relative_path(),
        &unit.start_byte().to_string(),
        &unit.end_byte().to_string(),
        unit.normalized_body_hash(),
        unit.name(),
        unit.scope().unwrap_or(""),
        unit.language_id(),
    ]
    .join("\0");
    let digest = Sha256::digest(ingredients.as_bytes());
    format!("unit:{}", &hex::encode(digest)[..16])
}

pub fn unit_line(unit: &impl UnitView) -> String {
    let scope = unit
        .scope()
        .map(|scope| format!(" ({scope})"))
        .unwrap_or_default();
    format!(
        "{} {}:{}:{}-{} {} {}{}",
        unit_id(unit),
        unit.project_label(),
        unit.relative_path(),
        unit.start_line(),
        unit.end_line(),
        unit.kind(),
        unit.name(),
        scope,
    )
}

/// Whether test entities participate in a query. Trivial test chunks
/// dominate degraded rankings (see evals/2026-07-14-qwen3-scenarios.md), so
/// consumers typically exclude them unless the intent is about tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TestsPolicy {
    Include,
    Exclude,
    Only,
}

/// Heuristic test detection over the extracted metadata: the frontend's
/// `test` kind, the `tests` scope assigned to `cfg(test)` items,
/// conventional test-file paths, and test-prefixed/suffixed names.
pub fn looks_like_test(unit: &impl UnitView) -> bool {
    if unit.kind() == "test" || unit.scope() == Some("tests") {
        return true;
    }
    let path = unit.relative_path();
    if path.starts_with("tests/")
        || path.contains("/tests/")
        || path.contains("/test/")
        || path.ends_with("_test.go")
        || path.ends_with("_test.py")
        || path.ends_with(".test.ts")
        || path.ends_with(".test.js")
        || path.ends_with("_spec.rb")
        || path
            .rsplit('/')
            .next()
            .is_some_and(|file| file.starts_with("test_"))
    {
        return true;
    }
    let name = unit.name();
    name.starts_with("test_") || name.ends_with("_test")
}

#[derive(Default)]
pub struct WhereFilter {
    clauses: Vec<String>,
    project: Vec<String>,
    language: Vec<String>,
    kind: Vec<String>,
    name: Vec<GlobMatcher>,
    scope: Vec<GlobMatcher>,
    path: Vec<GlobMatcher>,
    min_nodes: Option<usize>,
    tests: Option<TestsPolicy>,
}

impl WhereFilter {
    pub fn parse(expression: Option<&str>) -> Result<Self> {
        let mut filter = Self::default();
        let Some(expression) = expression else {
            return Ok(filter);
        };
        for clause in expression.split_whitespace() {
            let Some((key, value)) = clause.split_once('=') else {
                bail!("malformed --where clause {clause:?}: expected key=value");
            };
            match key {
                "project" => filter.project.push(value.to_owned()),
                "language" => filter.language.push(value.to_owned()),
                "kind" => filter.kind.push(value.to_owned()),
                "name" => filter.name.push(glob(value, false)?),
                "scope" => filter.scope.push(glob(value, false)?),
                "path" => filter.path.push(glob(value, true)?),
                "min_nodes" => {
                    filter.min_nodes =
                        Some(value.parse().with_context(|| {
                            format!("min_nodes wants an integer, got {value:?}")
                        })?);
                }
                "tests" => {
                    filter.tests = Some(match value {
                        "include" => TestsPolicy::Include,
                        "exclude" => TestsPolicy::Exclude,
                        "only" => TestsPolicy::Only,
                        other => bail!("tests wants include|exclude|only, got {other:?}"),
                    });
                }
                _ => bail!(
                    "unknown --where key {key:?} (supported: project, language, kind, name, scope, path, min_nodes, tests)"
                ),
            }
            filter.clauses.push(clause.to_owned());
        }
        Ok(filter)
    }

    pub fn matches(&self, unit: &impl UnitView) -> bool {
        let any_eq = |values: &[String], actual: &str| {
            values.is_empty() || values.iter().any(|value| value == actual)
        };
        let any_glob = |values: &[GlobMatcher], actual: &str| {
            values.is_empty() || values.iter().any(|value| value.is_match(actual))
        };
        any_eq(&self.project, unit.project_label())
            && any_eq(&self.language, unit.language_id())
            && any_eq(&self.kind, unit.kind())
            && any_glob(&self.name, unit.name())
            && (self.scope.is_empty()
                || unit
                    .scope()
                    .is_some_and(|scope| self.scope.iter().any(|matcher| matcher.is_match(scope))))
            && any_glob(&self.path, unit.relative_path())
            && self
                .min_nodes
                .is_none_or(|minimum| unit.body_node_count() >= minimum)
            && match self.tests.unwrap_or(TestsPolicy::Include) {
                TestsPolicy::Include => true,
                TestsPolicy::Exclude => !looks_like_test(unit),
                TestsPolicy::Only => looks_like_test(unit),
            }
    }

    /// The explicit `tests=` clause, when one was given. Consumers apply
    /// their own default (the CLI excludes tests) only when this is unset.
    pub fn tests_policy(&self) -> Option<TestsPolicy> {
        self.tests
    }

    /// Set the tests policy without an explicit `tests=` clause.
    pub fn set_tests_policy(&mut self, policy: TestsPolicy) {
        self.tests = Some(policy);
    }

    pub fn clauses(&self) -> &[String] {
        &self.clauses
    }
}

fn glob(pattern: &str, literal_separator: bool) -> Result<GlobMatcher> {
    Ok(GlobBuilder::new(pattern)
        .literal_separator(literal_separator)
        .build()
        .with_context(|| format!("invalid glob {pattern:?}"))?
        .compile_matcher())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScoredIndex {
    pub index: usize,
    pub score: f32,
}

pub fn dot(left: &[f32], right: &[f32]) -> f32 {
    left.iter().zip(right).map(|(a, b)| a * b).sum()
}

pub fn rank_candidates<'a>(
    query: &[f32],
    candidates: impl IntoIterator<Item = (usize, &'a [f32])>,
    threshold: f32,
) -> Vec<ScoredIndex> {
    let mut scored: Vec<ScoredIndex> = candidates
        .into_iter()
        .filter_map(|(index, vector)| {
            let score = dot(query, vector);
            (score >= threshold).then_some(ScoredIndex { index, score })
        })
        .collect();
    scored.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then(left.index.cmp(&right.index))
    });
    scored
}

/// Built-in retrieval intents (FEEDBACK.md §1): stable ids agents can pass
/// as a task preset instead of hand-writing instructions. Project configs
/// may extend or override these via `[tasks.<id>]`.
pub const TASK_PRESETS: &[(&str, &str)] = &[
    (
        "code-search",
        "Given a code search query, retrieve relevant code implementations",
    ),
    (
        "locate-edit-targets",
        "Given a software change request, retrieve code regions likely to require editing",
    ),
    (
        "explain-behavior",
        "Given a question about repository behavior, retrieve code that provides evidence for \
         the answer",
    ),
    (
        "find-analogues",
        "Given a code fragment, retrieve functionally equivalent implementations",
    ),
    (
        "diagnose-failure",
        "Given a failure report, retrieve implementation, tests, configuration, and \
         error-handling paths relevant to diagnosing it",
    ),
];

/// Function words dropped by [`compress_query`]. Deliberately small and
/// English-only: the goal is shedding narrative filler, not linguistics.
const QUERY_STOPWORDS: &[&str] = &[
    "a", "about", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because",
    "been", "being", "but", "by", "can", "could", "did", "do", "does", "doing", "for", "from",
    "had", "has", "have", "having", "how", "i", "if", "in", "into", "is", "it", "its", "just",
    "me", "my", "no", "not", "of", "on", "or", "our", "out", "over", "should", "so", "some",
    "such", "than", "that", "the", "their", "them", "then", "there", "these", "they", "this",
    "those", "to", "under", "up", "very", "want", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "why", "will", "with", "would", "you", "your",
];

/// Compress a narrative query to its distinctive terms: strip function
/// words, deduplicate while preserving order, cap the length. Long
/// paragraph-style questions drown their salient terms in phrasing and
/// drift dense retrieval toward generic chunks; the compressed form is
/// fused *alongside* the original, never instead of it. Returns `None`
/// when the query has fewer than `min_words` words or too few distinctive
/// terms — compressing an already-compact query only discards signal.
pub fn compress_query(query: &str, max_terms: usize, min_words: usize) -> Option<String> {
    let words = query.split_whitespace().count();
    if words < min_words {
        return None;
    }
    let mut seen = std::collections::HashSet::new();
    let terms: Vec<&str> = query
        .split(|c: char| !(c.is_alphanumeric() || c == '_'))
        .filter(|term| term.len() >= 3)
        .filter(|term| !QUERY_STOPWORDS.contains(&term.to_lowercase().as_str()))
        .filter(|term| seen.insert(term.to_lowercase()))
        .take(max_terms)
        .collect();
    if terms.len() < 4 {
        return None;
    }
    let compressed = terms.join(" ");
    if compressed == query {
        return None;
    }
    Some(compressed)
}

/// One retrieval source's ranked unit indices for fusion.
pub struct RankedList {
    /// Source label recorded in contributions (e.g. `dense`, `lexical`).
    pub source: String,
    pub weight: f32,
    /// Unit indices, best first.
    pub indices: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FusedIndex {
    pub index: usize,
    pub score: f32,
    /// `(source, 1-based rank)` for every list that contributed.
    pub contributions: Vec<(String, usize)>,
}

/// Weighted reciprocal-rank fusion across heterogeneous retrieval sources
/// (dense spaces, lexical BM25, …). Ranks are combined — never raw scores,
/// which are incomparable across sources. Deterministic: ties break by unit
/// index.
pub fn reciprocal_rank_fusion(lists: &[RankedList], rrf_k: usize) -> Vec<FusedIndex> {
    let mut fused: std::collections::HashMap<usize, (f32, Vec<(String, usize)>)> =
        std::collections::HashMap::new();
    for list in lists {
        for (zero_rank, &index) in list.indices.iter().enumerate() {
            let rank = zero_rank + 1;
            let entry = fused.entry(index).or_default();
            entry.0 += list.weight / (rrf_k + rank) as f32;
            entry.1.push((list.source.clone(), rank));
        }
    }
    let mut hits: Vec<FusedIndex> = fused
        .into_iter()
        .map(|(index, (score, contributions))| FusedIndex {
            index,
            score,
            contributions,
        })
        .collect();
    hits.sort_by(|left, right| {
        right
            .score
            .total_cmp(&left.score)
            .then(left.index.cmp(&right.index))
    });
    hits
}

/// Field-by-field differences between two semantic model contracts, derived
/// mechanically from their serialized form so newly added contract fields can
/// never be silently omitted from mismatch diagnostics.
pub fn contract_diff(stored: &ModelContract, current: &ModelContract) -> Vec<String> {
    let (Ok(serde_json::Value::Object(stored)), Ok(serde_json::Value::Object(current))) =
        (serde_json::to_value(stored), serde_json::to_value(current))
    else {
        return vec!["model contracts could not be serialized for comparison".into()];
    };
    stored
        .iter()
        .filter(|(key, left)| current.get(*key) != Some(left))
        .map(|(key, left)| {
            let right = current.get(key).unwrap_or(&serde_json::Value::Null);
            format!("{key} ({left} -> {right})")
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone)]
    struct Unit {
        project: String,
        path: String,
        name: String,
        scope: Option<String>,
        nodes: usize,
    }

    impl UnitView for Unit {
        fn project_label(&self) -> &str {
            &self.project
        }
        fn relative_path(&self) -> &str {
            &self.path
        }
        fn language_id(&self) -> &str {
            "rust"
        }
        fn kind(&self) -> &str {
            "function"
        }
        fn name(&self) -> &str {
            &self.name
        }
        fn scope(&self) -> Option<&str> {
            self.scope.as_deref()
        }
        fn start_byte(&self) -> usize {
            10
        }
        fn end_byte(&self) -> usize {
            90
        }
        fn start_line(&self) -> usize {
            2
        }
        fn end_line(&self) -> usize {
            9
        }
        fn body_node_count(&self) -> usize {
            self.nodes
        }
        fn normalized_body_hash(&self) -> &str {
            "body"
        }
    }

    fn unit(path: &str) -> Unit {
        Unit {
            project: "main".into(),
            path: path.into(),
            name: "parse".into(),
            scope: Some("Parser".into()),
            nodes: 20,
        }
    }

    #[test]
    fn selectors_and_filters_are_deterministic() {
        let value = unit("src/parser.rs");
        assert_eq!(unit_id(&value), unit_id(&value));
        assert!(
            WhereFilter::parse(Some("path=src/** scope=Parser min_nodes=10"))
                .unwrap()
                .matches(&value)
        );
        assert!(
            !WhereFilter::parse(Some("path=tests/**"))
                .unwrap()
                .matches(&value)
        );
    }

    #[test]
    fn test_likeness_and_policy_filtering() {
        let mut test_unit = unit("src/walk.rs");
        test_unit.scope = Some("tests".into());
        let plain = unit("src/walk.rs");
        assert!(looks_like_test(&test_unit));
        assert!(!looks_like_test(&plain));
        let by_path = unit("tests/integration.rs");
        assert!(looks_like_test(&by_path));

        let mut filter = WhereFilter::parse(None).unwrap();
        assert!(filter.matches(&test_unit), "default includes tests");
        filter.set_tests_policy(TestsPolicy::Exclude);
        assert!(!filter.matches(&test_unit));
        assert!(filter.matches(&plain));

        let only = WhereFilter::parse(Some("tests=only")).unwrap();
        assert_eq!(only.tests_policy(), Some(TestsPolicy::Only));
        assert!(only.matches(&test_unit));
        assert!(!only.matches(&plain));
    }

    #[test]
    fn narrative_queries_compress_to_distinctive_terms() {
        let paragraph = "I am building a larger application and want to avoid a global \
                         application object. The pattern I read about creates the application \
                         inside a function so that configuration can be passed in, multiple \
                         instances can exist for testing, and extensions get initialized \
                         against whichever instance is current. Where is that implemented?";
        let compressed = compress_query(paragraph, 16, 25).unwrap();
        assert!(compressed.contains("application"));
        assert!(!compressed.contains("factory"), "no invented terms");
        assert!(!compressed.contains("the "), "stopwords stripped");
        assert!(compressed.split_whitespace().count() <= 16);
        // Compact queries stay untouched.
        assert_eq!(compress_query("parse command line flags", 16, 25), None);
        assert_eq!(
            compress_query(
                "walk the directory tree while respecting gitignore rules",
                16,
                25
            ),
            None
        );
        // `min_words: 0` (--compress always) still refuses no-op compression.
        assert_eq!(compress_query("parse command line flags", 16, 0), None);
    }

    #[test]
    fn rrf_fuses_ranked_lists_deterministically() {
        let fused = reciprocal_rank_fusion(
            &[
                RankedList {
                    source: "dense".into(),
                    weight: 1.0,
                    indices: vec![7, 3, 9],
                },
                RankedList {
                    source: "lexical".into(),
                    weight: 1.0,
                    indices: vec![3, 11],
                },
            ],
            60,
        );
        // 3 appears in both lists (ranks 2 and 1) and must win.
        assert_eq!(fused[0].index, 3);
        assert_eq!(fused[0].contributions.len(), 2);
        // 7 (dense#1) beats 11 (lexical#2) and 9 (dense#3).
        assert_eq!(fused[1].index, 7);
        let expected = 1.0 / 62.0 + 1.0 / 61.0;
        assert!((fused[0].score - expected).abs() < 1e-6);
    }

    #[test]
    fn candidate_ranking_is_stable() {
        let first = [1.0, 0.0];
        let second = [0.5, 0.5];
        let ranked = rank_candidates(
            &first,
            [(1, second.as_slice()), (0, first.as_slice())],
            -1.0,
        );
        assert_eq!(ranked[0].index, 0);
        assert_eq!(ranked[1].index, 1);
    }
}
