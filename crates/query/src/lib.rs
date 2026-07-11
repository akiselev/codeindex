#![forbid(unsafe_code)]

use anyhow::{Context, Result, bail};
use codeindex_core::ModelIdentity;
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
                _ => bail!(
                    "unknown --where key {key:?} (supported: project, language, kind, name, scope, path, min_nodes)"
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

pub fn identity_diff(stored: &ModelIdentity, current: &ModelIdentity) -> Vec<String> {
    let optional = |value: &Option<String>| value.clone().unwrap_or_else(|| "none".into());
    let mut differences = Vec::new();
    let mut field = |name: &str, left: String, right: String| {
        if left != right {
            differences.push(format!("{name} ({left:?} -> {right:?})"));
        }
    };
    field("backend", stored.backend.clone(), current.backend.clone());
    field(
        "backend_version",
        stored.backend_version.clone(),
        current.backend_version.clone(),
    );
    field(
        "runtime_version",
        optional(&stored.runtime_version),
        optional(&current.runtime_version),
    );
    field("model", stored.model.clone(), current.model.clone());
    field(
        "revision",
        optional(&stored.revision),
        optional(&current.revision),
    );
    field(
        "dimensions",
        stored.dimensions.to_string(),
        current.dimensions.to_string(),
    );
    field(
        "tokenizer_hash",
        optional(&stored.tokenizer_hash),
        optional(&current.tokenizer_hash),
    );
    field(
        "model_hash",
        optional(&stored.model_hash),
        optional(&current.model_hash),
    );
    field(
        "normalize",
        stored.normalize.to_string(),
        current.normalize.to_string(),
    );
    field(
        "execution_provider",
        stored.execution_provider.clone(),
        current.execution_provider.clone(),
    );
    field(
        "quantization",
        optional(&stored.quantization),
        optional(&current.quantization),
    );
    field(
        "cache_path",
        optional(&stored.cache_path),
        optional(&current.cache_path),
    );
    differences
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
