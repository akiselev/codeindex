use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::OnceLock;

use anyhow::{Context, Result};
use serde::Deserialize;
use tree_sitter::{Language, Node, Query};

/// Parsed `assets/languages/<id>.toml`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LanguageSpec {
    pub id: String,
    pub extensions: Vec<String>,
    /// Node kinds stripped from unit text before hashing/embedding.
    #[serde(default)]
    pub comment_nodes: Vec<String>,
    /// Node kinds for attributes/annotations/decorators that may sit between
    /// documentation comments and the declaration; skipped (not collected)
    /// when harvesting the Documentation channel.
    #[serde(default)]
    pub attribute_nodes: Vec<String>,
    /// Wrapper node kinds whose child carries the declaration (Python's
    /// `decorated_definition`); the documentation walk continues from the
    /// wrapper's preceding siblings when the declaration has none of its own.
    #[serde(default)]
    pub doc_container_nodes: Vec<String>,
    /// Optional adapter hook name (see `adapter_by_name`).
    #[serde(default)]
    pub adapter: Option<String>,
    /// Ancestor node kinds that contribute display scope, e.g. classes.
    #[serde(default)]
    pub scopes: Vec<ScopeRule>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ScopeRule {
    pub kind: String,
    pub field: String,
}

/// A unit in flight between query matching and final entity construction.
/// Adapters may refine any of these fields.
pub struct PendingUnit<'t> {
    pub node: Node<'t>,
    pub body: Option<Node<'t>>,
    pub kind: String,
    pub name: Option<String>,
    pub scope: Option<String>,
    /// Absolute byte ranges (into the file source) removed before
    /// normalization, in addition to comment nodes.
    pub strip: Vec<Range<usize>>,
    /// Absolute byte range of an adapter-identified documentation literal
    /// (e.g. a Python docstring), preferred over leading comment siblings for
    /// the Documentation channel.
    pub doc: Option<Range<usize>>,
}

/// Hook layer for cases queries cannot express cleanly: anonymous-function
/// naming, receiver scopes, docstrings, decorators, macros, preprocessors.
pub trait LanguageAdapter: Send + Sync {
    fn refine(&self, source: &str, unit: &mut PendingUnit<'_>);
}

/// Python: strip a leading docstring from function bodies.
struct PythonAdapter;

impl LanguageAdapter for PythonAdapter {
    fn refine(&self, source: &str, unit: &mut PendingUnit<'_>) {
        let Some(body) = unit.body else { return };
        let Some(first) = body.named_child(0) else {
            return;
        };
        if first.kind() == "expression_statement"
            && let Some(expr) = first.named_child(0)
            && expr.kind() == "string"
            && is_plain_string_literal(source, expr)
        {
            unit.strip.push(first.byte_range());
            unit.doc = Some(expr.byte_range());
        }
    }
}

/// Only a plain (optionally raw/unicode) string literal is a docstring;
/// f-strings and bytes literals are runtime expressions, not documentation.
fn is_plain_string_literal(source: &str, string_node: Node<'_>) -> bool {
    source[string_node.byte_range()]
        .chars()
        .take_while(|c| *c != '"' && *c != '\'')
        .all(|c| matches!(c, 'r' | 'R' | 'u' | 'U'))
}

/// JavaScript/TypeScript: name anonymous functions from their assignment
/// context (`const f = () => ...`, `{ f: function() {} }`, `x.f = ...`) or,
/// failing that, from the call they are passed to (`it("...", () => ...)`).
struct JsLikeAdapter;

impl LanguageAdapter for JsLikeAdapter {
    fn refine(&self, source: &str, unit: &mut PendingUnit<'_>) {
        if unit.name.is_some() {
            return;
        }
        let mut node = unit.node;
        while let Some(parent) = node.parent() {
            let name = match parent.kind() {
                "variable_declarator" => parent
                    .child_by_field_name("name")
                    .map(|n| source[n.byte_range()].to_string()),
                "pair" => parent
                    .child_by_field_name("key")
                    .map(|n| source[n.byte_range()].to_string()),
                "assignment_expression" => parent
                    .child_by_field_name("left")
                    .map(|n| source[n.byte_range()].to_string()),
                "parenthesized_expression" => {
                    node = parent;
                    continue;
                }
                _ => None,
            };
            if name.is_some() {
                unit.name = name;
                return;
            }
            break;
        }
        unit.name =
            name_from_call_argument(source, unit.node, "arguments", &["string"]).or_else(|| {
                name_from_enclosing_function(
                    source,
                    unit.node,
                    &["function_declaration", "method_definition"],
                )
            });
    }
}

fn name_from_call_argument(
    source: &str,
    node: Node<'_>,
    arguments_kind: &str,
    string_kinds: &[&str],
) -> Option<String> {
    let arguments = node.parent()?;
    if arguments.kind() != arguments_kind {
        return None;
    }
    let call = arguments.parent()?;
    if call.kind() != "call_expression" {
        return None;
    }
    let callee_node = call.child_by_field_name("function")?;
    let callee: String = source[callee_node.byte_range()]
        .split_whitespace()
        .collect();
    let callee = callee.strip_prefix("self.").unwrap_or(&callee);
    let mut callee = callee_tail(callee, 2);
    if callee.len() > 40 {
        callee = callee_tail(callee, 1);
    }
    if callee.len() > 40 {
        return None;
    }
    let mut label = None;
    for i in 0..arguments.named_child_count() as u32 {
        let child = arguments.named_child(i)?;
        if child.byte_range() == node.byte_range() {
            break;
        }
        if string_kinds.contains(&child.kind()) {
            let text = source[child.byte_range()]
                .trim_start_matches(['b', 'r'])
                .trim_matches(|c| c == '"' || c == '\'' || c == '`' || c == '#')
                .trim();
            let text: String = text.chars().take(48).collect();
            if !text.contains('"') {
                label = Some(text);
            }
            break;
        }
    }
    match label {
        Some(label) => Some(format!("{callee}(\"{label}\")")),
        None => Some(format!("{callee}(...)")),
    }
}

fn callee_tail(callee: &str, keep: usize) -> &str {
    let mut depth = 0usize;
    let mut dots = 0usize;
    for (i, b) in callee.bytes().enumerate().rev() {
        match b {
            b')' | b']' | b'>' => depth += 1,
            b'(' | b'[' | b'<' => depth = depth.saturating_sub(1),
            b'.' if depth == 0 => {
                dots += 1;
                if dots == keep {
                    return &callee[i + 1..];
                }
            }
            _ => {}
        }
    }
    callee
}

struct RustAdapter;

impl LanguageAdapter for RustAdapter {
    fn refine(&self, source: &str, unit: &mut PendingUnit<'_>) {
        if unit.node.kind() == "function_item"
            && unit.scope.is_none()
            && has_cfg_test_attribute(source, unit.node)
        {
            unit.scope = Some("tests".to_string());
        }
        if unit.name.is_some() || unit.node.kind() != "closure_expression" {
            return;
        }
        if let Some(parent) = unit.node.parent() {
            let name = match parent.kind() {
                "let_declaration" => parent.child_by_field_name("pattern"),
                "assignment_expression" => parent.child_by_field_name("left"),
                _ => None,
            }
            .map(|n| source[n.byte_range()].to_string())
            .filter(|n| n.len() <= 40 && !n.contains('\n'));
            if name.is_some() {
                unit.name = name;
                return;
            }
        }
        unit.name = name_from_call_argument(
            source,
            unit.node,
            "arguments",
            &["string_literal", "raw_string_literal"],
        );
    }
}

fn has_cfg_test_attribute(source: &str, node: Node<'_>) -> bool {
    let mut sibling = node.prev_named_sibling();
    while let Some(current) = sibling {
        match current.kind() {
            "attribute_item" => {
                if attribute_enables_cfg_test(source, current) {
                    return true;
                }
            }
            "line_comment" | "block_comment" => {}
            _ => return false,
        }
        sibling = current.prev_named_sibling();
    }
    false
}

/// Whether an `attribute_item` is `#[cfg(...)]` with a bare `test` token in an
/// enabled position, i.e. not inside a `not(...)` group.
fn attribute_enables_cfg_test(source: &str, item: Node<'_>) -> bool {
    let Some(attribute) = item.named_child(0) else {
        return false;
    };
    if attribute.kind() != "attribute" {
        return false;
    }
    let Some(path) = attribute.named_child(0) else {
        return false;
    };
    if &source[path.byte_range()] != "cfg" {
        return false;
    }
    let Some(arguments) = attribute.named_child(1) else {
        return false;
    };
    arguments.kind() == "token_tree" && token_tree_enables_test(source, arguments)
}

/// Scan a `cfg` token tree for a standalone `test` identifier token. Groups
/// guarded by `not` are skipped; string literals (`feature = "testing"`) are
/// distinct token kinds and never match.
fn token_tree_enables_test(source: &str, tree: Node<'_>) -> bool {
    let mut skip_next_group = false;
    for i in 0..tree.named_child_count() as u32 {
        let Some(child) = tree.named_child(i) else {
            continue;
        };
        match child.kind() {
            "identifier" => match &source[child.byte_range()] {
                "not" => skip_next_group = true,
                "test" => return true,
                _ => {}
            },
            "token_tree" => {
                if std::mem::take(&mut skip_next_group) {
                    continue;
                }
                if token_tree_enables_test(source, child) {
                    return true;
                }
            }
            _ => {}
        }
    }
    false
}

fn name_from_enclosing_function(
    source: &str,
    node: Node<'_>,
    declaration_kinds: &[&str],
) -> Option<String> {
    let mut current = node.parent()?;
    loop {
        if declaration_kinds.contains(&current.kind())
            && let Some(name) = current.child_by_field_name("name")
        {
            return Some(format!("{}.func", &source[name.byte_range()]));
        }
        current = current.parent()?;
    }
}

struct GoAdapter;

impl LanguageAdapter for GoAdapter {
    fn refine(&self, source: &str, unit: &mut PendingUnit<'_>) {
        if unit.node.kind() == "func_literal" && unit.name.is_none() {
            let assignment_name = unit.node.parent().and_then(|parent| {
                if parent.kind() != "expression_list" || parent.named_child_count() != 1 {
                    return None;
                }
                let declaration = parent.parent()?;
                match declaration.kind() {
                    "short_var_declaration" | "assignment_statement" => {
                        declaration.child_by_field_name("left")
                    }
                    "var_spec" => declaration.child_by_field_name("name"),
                    _ => None,
                }
            });
            if let Some(name_node) = assignment_name {
                let name = source[name_node.byte_range()].to_string();
                if name.len() <= 40 && !name.contains('\n') {
                    unit.name = Some(name);
                    return;
                }
            }
            unit.name = name_from_call_argument(
                source,
                unit.node,
                "argument_list",
                &["interpreted_string_literal", "raw_string_literal"],
            )
            .or_else(|| {
                name_from_enclosing_function(
                    source,
                    unit.node,
                    &["function_declaration", "method_declaration"],
                )
            });
            return;
        }
        if unit.node.kind() != "method_declaration" || unit.scope.is_some() {
            return;
        }
        if let Some(receiver) = unit.node.child_by_field_name("receiver") {
            let text = source[receiver.byte_range()]
                .trim_matches(|c| c == '(' || c == ')')
                .split_whitespace()
                .last()
                .map(|t| t.trim_start_matches(['*', '&']).to_string());
            unit.scope = text;
        }
    }
}

pub fn adapter_by_name(name: &str) -> Result<&'static dyn LanguageAdapter> {
    static PYTHON: PythonAdapter = PythonAdapter;
    static JS_LIKE: JsLikeAdapter = JsLikeAdapter;
    static GO: GoAdapter = GoAdapter;
    static RUST: RustAdapter = RustAdapter;
    match name {
        "python" => Ok(&PYTHON),
        "js-like" => Ok(&JS_LIKE),
        "go" => Ok(&GO),
        "rust" => Ok(&RUST),
        other => anyhow::bail!("unknown language adapter {other:?}"),
    }
}

pub struct LanguageDef {
    pub spec: LanguageSpec,
    pub language: Language,
    pub query: Query,
    /// Optional reference/call-site query for the Usage channel. Captures a
    /// `@ref.callee` node per call site. `None`/empty when a language has no
    /// reference support yet (its units still index; Usage stays empty).
    pub references: Option<Query>,
    pub adapter: Option<&'static dyn LanguageAdapter>,
}

macro_rules! bundled {
    ($id:literal, $lang:expr) => {
        (
            $id,
            include_str!(concat!("../assets/languages/", $id, ".toml")),
            include_str!(concat!("../assets/languages/", $id, "/units.scm")),
            include_str!(concat!("../assets/languages/", $id, "/references.scm")),
            Language::new($lang),
        )
    };
}

fn bundled_languages() -> Vec<(
    &'static str,
    &'static str,
    &'static str,
    &'static str,
    Language,
)> {
    vec![
        bundled!("c", tree_sitter_c::LANGUAGE),
        bundled!("cpp", tree_sitter_cpp::LANGUAGE),
        bundled!("rust", tree_sitter_rust::LANGUAGE),
        bundled!("python", tree_sitter_python::LANGUAGE),
        bundled!("javascript", tree_sitter_javascript::LANGUAGE),
        bundled!("typescript", tree_sitter_typescript::LANGUAGE_TYPESCRIPT),
        bundled!("java", tree_sitter_java::LANGUAGE),
        bundled!("kotlin", tree_sitter_kotlin_ng::LANGUAGE),
        bundled!("csharp", tree_sitter_c_sharp::LANGUAGE),
        bundled!("go", tree_sitter_go::LANGUAGE),
        bundled!("php", tree_sitter_php::LANGUAGE_PHP),
        bundled!("ruby", tree_sitter_ruby::LANGUAGE),
    ]
}

pub struct LanguageRegistry {
    languages: BTreeMap<String, LanguageDef>,
}

impl LanguageRegistry {
    fn load() -> Result<Self> {
        let mut languages = BTreeMap::new();
        for (id, spec_toml, query_src, references_src, language) in bundled_languages() {
            let spec: LanguageSpec = toml::from_str(spec_toml)
                .with_context(|| format!("parsing language spec for {id}"))?;
            anyhow::ensure!(spec.id == id, "language spec id mismatch for {id}");
            let query = Query::new(&language, query_src)
                .with_context(|| format!("compiling units.scm for {id}"))?;
            let references = {
                let query = Query::new(&language, references_src)
                    .with_context(|| format!("compiling references.scm for {id}"))?;
                (query.pattern_count() > 0).then_some(query)
            };
            let adapter = spec
                .adapter
                .as_deref()
                .map(adapter_by_name)
                .transpose()
                .with_context(|| format!("resolving adapter for {id}"))?;
            languages.insert(
                id.to_string(),
                LanguageDef {
                    spec,
                    language,
                    query,
                    references,
                    adapter,
                },
            );
        }
        Ok(Self { languages })
    }

    pub fn global() -> &'static LanguageRegistry {
        static REGISTRY: OnceLock<LanguageRegistry> = OnceLock::new();
        REGISTRY
            .get_or_init(|| LanguageRegistry::load().expect("bundled language assets must compile"))
    }

    pub fn get(&self, id: &str) -> Option<&LanguageDef> {
        self.languages.get(id)
    }

    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.languages.keys().map(|s| s.as_str())
    }

    pub fn by_extension(&self, extension: &str) -> Option<&LanguageDef> {
        self.languages
            .values()
            .find(|def| def.spec.extensions.iter().any(|e| e == extension))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_bundled_languages_load() {
        let registry = LanguageRegistry::global();
        let ids: Vec<&str> = registry.ids().collect();
        assert_eq!(
            ids,
            vec![
                "c",
                "cpp",
                "csharp",
                "go",
                "java",
                "javascript",
                "kotlin",
                "php",
                "python",
                "ruby",
                "rust",
                "typescript"
            ]
        );
    }

    #[test]
    fn registry_matches_bundled_language_ids() {
        let registry = LanguageRegistry::global();
        for id in crate::BUNDLED_LANGUAGE_IDS {
            assert!(registry.get(id).is_some(), "bundled id {id} missing");
        }
        assert_eq!(registry.ids().count(), crate::BUNDLED_LANGUAGE_IDS.len());
    }

    #[test]
    fn extension_resolution() {
        let registry = LanguageRegistry::global();
        assert_eq!(registry.by_extension("rs").unwrap().spec.id, "rust");
        assert_eq!(registry.by_extension("c").unwrap().spec.id, "c");
        assert_eq!(registry.by_extension("cpp").unwrap().spec.id, "cpp");
        assert_eq!(registry.by_extension("php").unwrap().spec.id, "php");
        assert_eq!(registry.by_extension("rb").unwrap().spec.id, "ruby");
        assert_eq!(registry.by_extension("tsx").unwrap().spec.id, "typescript");
        assert_eq!(registry.by_extension("mjs").unwrap().spec.id, "javascript");
        assert!(registry.by_extension("txt").is_none());
    }
}
