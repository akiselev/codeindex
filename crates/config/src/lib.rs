#![forbid(unsafe_code)]

//! Project discovery and configuration for the codeindex CLI and daemon.
//!
//! There is exactly one configuration file name — `.codeindex.toml` — and it
//! is both the project marker and the configuration. Commands walk up from
//! the working directory collecting every `.codeindex.toml` on the ancestor
//! path: the **topmost** file is the project root and the files below it are
//! folder overrides. Nearest-file-wins would misanchor a project at a
//! subfolder override like `vendor/`, which is why the walk continues to the
//! filesystem root; a file with `root = true` stops the walk early so nested
//! independent projects stay independent.
//!
//! Databases never live in the tree. The registry maps each added root to a
//! database under the user data directory, so there is nothing to gitignore.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use sha2::Digest;

/// The single configuration file name used at every level of a project.
pub const CONFIG_FILE: &str = ".codeindex.toml";

/// Root-level project configuration (the topmost `.codeindex.toml`).
///
/// Identity-level keys (`spaces`, models) are root-only by design: folder
/// overrides may narrow or tune, never fragment a project into incompatible
/// vector spaces.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct ProjectConfig {
    /// Stops upward discovery: this file is a project root even when an
    /// ancestor directory also carries a `.codeindex.toml`.
    #[serde(default)]
    pub root: bool,
    /// Project label used in the registry and search output. Defaults to the
    /// root directory name.
    #[serde(default)]
    pub label: Option<String>,
    #[serde(default)]
    pub index: IndexSection,
    #[serde(default)]
    pub spaces: BTreeMap<String, SpaceConfig>,
    #[serde(default)]
    pub search: SearchSection,
    #[serde(default)]
    pub storage: StorageSection,
    #[serde(default)]
    pub tasks: BTreeMap<String, TaskConfig>,
    #[serde(default)]
    pub lsp: BTreeMap<String, LspConfig>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct IndexSection {
    /// Language ids to index; `None` means every bundled language.
    #[serde(default)]
    pub languages: Option<Vec<String>>,
    /// Glob patterns excluded from indexing, relative to the config file's
    /// directory.
    #[serde(default)]
    pub exclude: Vec<String>,
}

/// One embedding space definition. `model` is a resolvable reference
/// (`hf:owner/name[@rev]`, `dir:/path`, `fastembed:Name`).
#[derive(Debug, Clone, Deserialize)]
pub struct SpaceConfig {
    pub channel: String,
    pub model: String,
    #[serde(default)]
    pub document_prompt: Option<String>,
    #[serde(default)]
    pub output_dimensions: Option<usize>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct SearchSection {
    /// Space searched when the caller does not name one.
    #[serde(default)]
    pub default_space: Option<String>,
    /// `include` | `exclude` | `only`.
    #[serde(default)]
    pub tests: Option<String>,
    /// `hybrid` | `dense` | `lexical`.
    #[serde(default)]
    pub retrieval: Option<String>,
    #[serde(default)]
    pub rerank: Option<bool>,
    #[serde(default)]
    pub rerank_model: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct StorageSection {
    /// Explicit database path (relative paths resolve against the project
    /// root). Default: `<data_root>/projects/<key>/index.db` outside the tree.
    #[serde(default)]
    pub path: Option<PathBuf>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TaskConfig {
    pub instruction: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LspConfig {
    pub server: String,
    #[serde(default)]
    pub args: Vec<String>,
}

/// A folder-level `.codeindex.toml`. Only narrowing keys are legal here;
/// unknown keys (for example a `[spaces]` table) are rejected loudly instead
/// of being silently ignored.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FolderOverride {
    /// Present only so a nested project root parses before discovery decides
    /// it terminates the walk; always false for genuine overrides.
    #[serde(default)]
    pub root: bool,
    #[serde(default)]
    pub index: Option<IndexOverride>,
    #[serde(default)]
    pub search: Option<SearchOverride>,
}

/// `index = false` disables the subtree; a table tunes it.
#[derive(Debug, Clone, Deserialize)]
#[serde(untagged)]
pub enum IndexOverride {
    Enabled(bool),
    Tune {
        #[serde(default)]
        exclude: Vec<String>,
        #[serde(default)]
        languages: Option<Vec<String>>,
    },
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SearchOverride {
    #[serde(default)]
    pub tests: Option<String>,
}

/// One override file, located by the directory that carries it.
#[derive(Debug, Clone)]
pub struct OverrideFile {
    /// Directory containing the file, relative to the project root.
    pub relative_dir: PathBuf,
    pub settings: FolderOverride,
}

/// A discovered project: the root, its parsed configuration, and the
/// override chain between the root and the starting directory (root-first).
#[derive(Debug, Clone)]
pub struct ProjectContext {
    pub root: PathBuf,
    pub config: ProjectConfig,
    pub overrides: Vec<OverrideFile>,
}

impl ProjectContext {
    pub fn label(&self) -> String {
        self.config
            .label
            .clone()
            .unwrap_or_else(|| default_label(&self.root))
    }

    /// Effective tests policy for the starting directory: nearest override
    /// wins, then the root `[search] tests` key.
    pub fn tests_policy(&self) -> Option<&str> {
        self.overrides
            .iter()
            .rev()
            .find_map(|file| {
                file.settings
                    .search
                    .as_ref()
                    .and_then(|search| search.tests.as_deref())
            })
            .or(self.config.search.tests.as_deref())
    }

    /// The database path for this project: `[storage] path` if configured
    /// (relative to the root), otherwise a stable location under the user
    /// data directory keyed by the root path.
    pub fn db_path(&self) -> PathBuf {
        match &self.config.storage.path {
            Some(path) if path.is_absolute() => path.clone(),
            Some(path) => self.root.join(path),
            None => default_db_path(&self.root),
        }
    }
}

/// Walk up from `start`, collecting every `.codeindex.toml` on the ancestor
/// path. The topmost file (or the first marked `root = true`) anchors the
/// project; files below it form the override chain, ordered root-first.
pub fn discover(start: &Path) -> Result<Option<ProjectContext>> {
    let start = if start.is_absolute() {
        start.to_path_buf()
    } else {
        std::env::current_dir()?.join(start)
    };
    let mut found: Vec<(PathBuf, String)> = Vec::new();
    for dir in start.ancestors() {
        let candidate = dir.join(CONFIG_FILE);
        if !candidate.is_file() {
            continue;
        }
        let text = std::fs::read_to_string(&candidate)
            .with_context(|| format!("reading {}", candidate.display()))?;
        let is_root = toml::from_str::<RootMarker>(&text)
            .map(|marker| marker.root)
            .unwrap_or(false);
        found.push((dir.to_path_buf(), text));
        if is_root {
            break;
        }
    }
    let Some((root, root_text)) = found.pop() else {
        return Ok(None);
    };
    let config: ProjectConfig = toml::from_str(&root_text)
        .with_context(|| format!("parsing {}", root.join(CONFIG_FILE).display()))?;
    // Remaining entries are below the root, currently nearest-first.
    let mut overrides = Vec::new();
    for (dir, text) in found.into_iter().rev() {
        let settings: FolderOverride = toml::from_str(&text).with_context(|| {
            format!(
                "parsing folder override {} (folder overrides may only narrow: \
                 index/search keys; spaces and models are root-only)",
                dir.join(CONFIG_FILE).display()
            )
        })?;
        let relative_dir = dir
            .strip_prefix(&root)
            .expect("override directories descend from the discovered root")
            .to_path_buf();
        overrides.push(OverrideFile {
            relative_dir,
            settings,
        });
    }
    Ok(Some(ProjectContext {
        root,
        config,
        overrides,
    }))
}

#[derive(Deserialize, Default)]
struct RootMarker {
    #[serde(default)]
    root: bool,
}

/// Walk the whole tree under `root` collecting folder overrides (excluding
/// the root's own file). Indexing needs every override in the project, not
/// just the ones on the working directory's ancestor chain. Directories
/// marked `root = true` are nested independent projects and are skipped
/// entirely (they exclude themselves from the outer project).
pub fn collect_overrides(root: &Path) -> Result<Vec<OverrideFile>> {
    let mut overrides = Vec::new();
    let mut nested_roots = Vec::new();
    collect_overrides_into(root, root, &mut overrides, &mut nested_roots)?;
    for nested in nested_roots {
        overrides.push(OverrideFile {
            relative_dir: nested,
            settings: FolderOverride {
                root: false,
                index: Some(IndexOverride::Enabled(false)),
                search: None,
            },
        });
    }
    overrides.sort_by(|a, b| a.relative_dir.cmp(&b.relative_dir));
    Ok(overrides)
}

fn collect_overrides_into(
    root: &Path,
    dir: &Path,
    overrides: &mut Vec<OverrideFile>,
    nested_roots: &mut Vec<PathBuf>,
) -> Result<()> {
    for entry in std::fs::read_dir(dir).with_context(|| format!("listing {}", dir.display()))? {
        let entry = entry?;
        let path = entry.path();
        let name = entry.file_name();
        if entry.file_type()?.is_dir() {
            if name.to_str().is_some_and(|name| name.starts_with('.')) {
                continue;
            }
            collect_overrides_into(root, &path, overrides, nested_roots)?;
        } else if name.to_str() == Some(CONFIG_FILE) && dir != root {
            let text = std::fs::read_to_string(&path)
                .with_context(|| format!("reading {}", path.display()))?;
            let relative_dir = dir
                .strip_prefix(root)
                .expect("walk stays under root")
                .to_path_buf();
            if toml::from_str::<RootMarker>(&text)
                .map(|marker| marker.root)
                .unwrap_or(false)
            {
                nested_roots.push(relative_dir);
                continue;
            }
            let settings: FolderOverride = toml::from_str(&text).with_context(|| {
                format!(
                    "parsing folder override {} (folder overrides may only narrow: \
                     index/search keys; spaces and models are root-only)",
                    path.display()
                )
            })?;
            overrides.push(OverrideFile {
                relative_dir,
                settings,
            });
        }
    }
    Ok(())
}

/// Effective exclude globs for indexing `root`: the root config's excludes
/// plus every folder override translated into root-relative patterns
/// (`index = false` becomes `<dir>/**`).
pub fn effective_excludes(config: &ProjectConfig, overrides: &[OverrideFile]) -> Vec<String> {
    let mut excludes = config.index.exclude.clone();
    for file in overrides {
        let prefix = file.relative_dir.to_string_lossy().replace('\\', "/");
        match &file.settings.index {
            Some(IndexOverride::Enabled(false)) => {
                excludes.push(if prefix.is_empty() {
                    "**".into()
                } else {
                    format!("{prefix}/**")
                });
            }
            Some(IndexOverride::Enabled(true)) | None => {}
            Some(IndexOverride::Tune { exclude, .. }) => {
                for pattern in exclude {
                    excludes.push(if prefix.is_empty() {
                        pattern.clone()
                    } else {
                        format!("{prefix}/{pattern}")
                    });
                }
            }
        }
    }
    excludes
}

/// The user data root: `$CODEINDEX_DATA_DIR`, else `$XDG_DATA_HOME/codeindex`,
/// else `~/.local/share/codeindex`.
pub fn data_root() -> PathBuf {
    if let Some(dir) = std::env::var_os("CODEINDEX_DATA_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(dir) = std::env::var_os("XDG_DATA_HOME")
        && !dir.is_empty()
    {
        return PathBuf::from(dir).join("codeindex");
    }
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."));
    home.join(".local/share/codeindex")
}

/// Stable out-of-tree database location for a project root.
pub fn default_db_path(root: &Path) -> PathBuf {
    let digest = sha2::Sha256::digest(root.to_string_lossy().as_bytes());
    let key = hex(&digest[..8]);
    data_root().join("projects").join(key).join("index.db")
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn default_label(root: &Path) -> String {
    root.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .unwrap_or_else(|| "project".into())
}

/// The registry of added project roots, stored at
/// `<data_root>/registry.json`. This is the only global mutable file; the
/// daemon rebuilds all runtime state from it plus the per-project databases.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Registry {
    #[serde(default)]
    pub projects: Vec<RegisteredProject>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegisteredProject {
    pub label: String,
    pub root: PathBuf,
    pub db: PathBuf,
}

pub fn registry_path() -> PathBuf {
    data_root().join("registry.json")
}

impl Registry {
    pub fn load() -> Result<Registry> {
        Self::load_from(&registry_path())
    }

    pub fn load_from(path: &Path) -> Result<Registry> {
        match std::fs::read(path) {
            Ok(bytes) => serde_json::from_slice(&bytes)
                .with_context(|| format!("parsing {}", path.display())),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(Registry::default()),
            Err(error) => Err(error).with_context(|| format!("reading {}", path.display())),
        }
    }

    pub fn save(&self) -> Result<()> {
        self.save_to(&registry_path())
    }

    pub fn save_to(&self, path: &Path) -> Result<()> {
        let parent = path.parent().context("registry path has no parent")?;
        std::fs::create_dir_all(parent)?;
        let temp = path.with_extension("json.tmp");
        std::fs::write(&temp, serde_json::to_vec_pretty(self)?)?;
        std::fs::rename(&temp, path).with_context(|| format!("replacing {}", path.display()))?;
        Ok(())
    }

    /// Register a root (idempotent by root path). Labels are unique: a
    /// clashing label from a different root gets a numeric suffix.
    pub fn register(
        &mut self,
        root: &Path,
        label_hint: &str,
        db: PathBuf,
    ) -> Result<RegisteredProject> {
        if let Some(existing) = self.projects.iter().find(|project| project.root == root) {
            return Ok(existing.clone());
        }
        let mut label = label_hint.to_string();
        let mut counter = 2;
        while self.projects.iter().any(|project| project.label == label) {
            label = format!("{label_hint}-{counter}");
            counter += 1;
        }
        let project = RegisteredProject {
            label,
            root: root.to_path_buf(),
            db,
        };
        self.projects.push(project.clone());
        Ok(project)
    }

    /// Find by label, exact root, or containment (deepest registered root
    /// containing the path wins, so nested registrations resolve correctly).
    pub fn find(&self, needle: &str) -> Option<&RegisteredProject> {
        if let Some(by_label) = self.projects.iter().find(|project| project.label == needle) {
            return Some(by_label);
        }
        let path = Path::new(needle);
        self.projects
            .iter()
            .filter(|project| path.starts_with(&project.root))
            .max_by_key(|project| project.root.components().count())
    }

    pub fn remove(&mut self, needle: &str) -> Result<RegisteredProject> {
        let index = self
            .projects
            .iter()
            .position(|project| project.label == needle || project.root == Path::new(needle))
            .with_context(|| format!("no registered project matches {needle:?}"))?;
        Ok(self.projects.remove(index))
    }
}

/// A minimal root `.codeindex.toml` written by `codeindex add` when the
/// directory has none yet.
pub fn starter_config(label: &str) -> String {
    format!(
        "# codeindex project configuration (see docs/daemon-cli-plan.md)\n\
         root = true\n\
         label = {label:?}\n\
         \n\
         [index]\n\
         exclude = [\"target/**\", \"node_modules/**\", \"dist/**\", \".venv/**\"]\n\
         \n\
         # Uncomment to define an embedding space (downloads the model):\n\
         # [spaces.code]\n\
         # channel = \"implementation\"\n\
         # model = \"hf:Qwen/Qwen3-Embedding-0.6B\"\n\
         \n\
         [search]\n\
         tests = \"exclude\"\n"
    )
}

/// Validate a config for use as a project root before registering it.
pub fn validate_root_config(config: &ProjectConfig) -> Result<()> {
    for (id, space) in &config.spaces {
        if space.channel.is_empty() {
            bail!("space {id:?} has an empty channel");
        }
        if space.model.is_empty() {
            bail!("space {id:?} has an empty model reference");
        }
    }
    if let Some(default_space) = &config.search.default_space
        && !config.spaces.contains_key(default_space)
    {
        bail!("[search] default_space {default_space:?} is not a defined space");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(path: &Path, text: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, text).unwrap();
    }

    #[test]
    fn discovery_anchors_at_the_topmost_file_not_the_nearest() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("repo");
        write(&root.join(CONFIG_FILE), "label = \"repo\"\n");
        write(&root.join("vendor").join(CONFIG_FILE), "index = false\n");
        let from_vendor = discover(&root.join("vendor")).unwrap().unwrap();
        assert_eq!(from_vendor.root, root);
        assert_eq!(from_vendor.label(), "repo");
        assert_eq!(from_vendor.overrides.len(), 1);
        assert_eq!(
            from_vendor.overrides[0].relative_dir,
            PathBuf::from("vendor")
        );
    }

    #[test]
    fn root_marker_stops_the_upward_walk() {
        let temp = tempfile::tempdir().unwrap();
        let outer = temp.path().join("outer");
        let inner = outer.join("libs/inner");
        write(&outer.join(CONFIG_FILE), "label = \"outer\"\n");
        write(&inner.join(CONFIG_FILE), "root = true\nlabel = \"inner\"\n");
        let context = discover(&inner.join("src")).unwrap().unwrap();
        assert_eq!(context.root, inner);
        assert_eq!(context.label(), "inner");
        assert!(context.overrides.is_empty());
    }

    #[test]
    fn override_chain_orders_root_first_and_tests_policy_is_nearest_wins() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("repo");
        write(&root.join(CONFIG_FILE), "[search]\ntests = \"exclude\"\n");
        write(
            &root.join("suite").join(CONFIG_FILE),
            "[search]\ntests = \"include\"\n",
        );
        let context = discover(&root.join("suite/deep")).unwrap().unwrap();
        assert_eq!(context.tests_policy(), Some("include"));
        let at_root = discover(&root).unwrap().unwrap();
        assert_eq!(at_root.tests_policy(), Some("exclude"));
    }

    #[test]
    fn folder_overrides_reject_identity_keys() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("repo");
        write(&root.join(CONFIG_FILE), "");
        write(
            &root.join("sub").join(CONFIG_FILE),
            "[spaces.code]\nchannel = \"implementation\"\nmodel = \"hf:x/y\"\n",
        );
        let error = discover(&root.join("sub")).unwrap_err().to_string();
        assert!(
            error.contains("folder overrides may only narrow"),
            "{error}"
        );
    }

    #[test]
    fn collected_overrides_translate_into_root_relative_excludes() {
        let temp = tempfile::tempdir().unwrap();
        let root = temp.path().join("repo");
        write(
            &root.join(CONFIG_FILE),
            "[index]\nexclude = [\"target/**\"]\n",
        );
        write(&root.join("vendor").join(CONFIG_FILE), "index = false\n");
        write(
            &root.join("docs").join(CONFIG_FILE),
            "[index]\nexclude = [\"fixtures/**\"]\n",
        );
        // A nested independent project excludes itself from the outer index.
        write(
            &root.join("embedded").join(CONFIG_FILE),
            "root = true\nlabel = \"embedded\"\n",
        );
        let context = discover(&root).unwrap().unwrap();
        let overrides = collect_overrides(&root).unwrap();
        let excludes = effective_excludes(&context.config, &overrides);
        assert_eq!(
            excludes,
            vec![
                "target/**".to_string(),
                "docs/fixtures/**".to_string(),
                "embedded/**".to_string(),
                "vendor/**".to_string(),
            ]
        );
    }

    #[test]
    fn registry_round_trips_and_dedupes_labels() {
        let temp = tempfile::tempdir().unwrap();
        let path = temp.path().join("registry.json");
        let mut registry = Registry::default();
        let first = registry
            .register(
                &temp.path().join("a/app"),
                "app",
                PathBuf::from("/tmp/a.db"),
            )
            .unwrap();
        let second = registry
            .register(
                &temp.path().join("b/app"),
                "app",
                PathBuf::from("/tmp/b.db"),
            )
            .unwrap();
        assert_eq!(first.label, "app");
        assert_eq!(second.label, "app-2");
        // Idempotent by root.
        let again = registry
            .register(&temp.path().join("a/app"), "app", PathBuf::from("/ignored"))
            .unwrap();
        assert_eq!(again.label, "app");
        registry.save_to(&path).unwrap();
        let loaded = Registry::load_from(&path).unwrap();
        assert_eq!(loaded.projects.len(), 2);
        // Containment lookup picks the deepest root.
        let deep = temp.path().join("a/app/src/lib.rs");
        assert_eq!(loaded.find(deep.to_str().unwrap()).unwrap().label, "app");
        assert_eq!(loaded.find("app-2").unwrap().label, "app-2");
        assert!(loaded.find("missing").is_none());
    }
}
