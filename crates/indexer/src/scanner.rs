use std::collections::HashSet;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use ignore::WalkBuilder;
use ignore::overrides::OverrideBuilder;

use codeindex_tree_sitter::LanguageRegistry;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScannedFile {
    pub absolute_path: PathBuf,
    /// Forward-slash path relative to the project root.
    pub relative_path: String,
    pub language_id: String,
}

/// Walk a project root and return files whose language is enabled.
/// Respects `.gitignore` files plus the project's exclude patterns, including
/// unpacked source trees that are not themselves Git repositories.
pub fn scan_files(
    root: &Path,
    exclude: &[String],
    enabled_languages: &HashSet<String>,
) -> Result<Vec<ScannedFile>> {
    let mut overrides = OverrideBuilder::new(root);
    for pattern in exclude {
        // Override syntax whitelists by default; `!` makes it an exclude.
        overrides
            .add(&format!("!{pattern}"))
            .with_context(|| format!("bad exclude pattern {pattern:?}"))?;
    }
    let overrides = overrides.build()?;

    let registry = LanguageRegistry::global();
    let mut files = Vec::new();
    let mut walk = WalkBuilder::new(root);
    walk.overrides(overrides).require_git(false);
    for entry in walk.build() {
        let entry = entry?;
        if !entry.file_type().is_some_and(|t| t.is_file()) {
            continue;
        }
        let path = entry.path();
        let Some(extension) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        let Some(def) = registry.by_extension(&extension.to_lowercase()) else {
            continue;
        };
        if !enabled_languages.contains(&def.spec.id) {
            continue;
        }
        let relative = path
            .strip_prefix(root)
            .with_context(|| format!("{} outside root {}", path.display(), root.display()))?;
        let relative_path = relative
            .components()
            .map(|c| c.as_os_str().to_string_lossy())
            .collect::<Vec<_>>()
            .join("/");
        files.push(ScannedFile {
            absolute_path: path.to_path_buf(),
            relative_path,
            language_id: def.spec.id.clone(),
        });
    }
    files.sort_by(|a, b| a.relative_path.cmp(&b.relative_path));
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(root: &Path, rel: &str, content: &str) {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, content).unwrap();
    }

    fn all_languages() -> HashSet<String> {
        LanguageRegistry::global()
            .ids()
            .map(str::to_string)
            .collect()
    }

    #[test]
    fn scans_enabled_languages_only() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "a.rs", "fn a() {}");
        write(dir.path(), "b.py", "def b(): pass");
        write(dir.path(), "c.txt", "not code");
        let mut enabled = HashSet::new();
        enabled.insert("rust".to_string());
        let files = scan_files(dir.path(), &[], &enabled).unwrap();
        assert_eq!(files.len(), 1);
        assert_eq!(files[0].relative_path, "a.rs");
        assert_eq!(files[0].language_id, "rust");
    }

    #[test]
    fn exclude_patterns_and_gitignore() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "src/keep.rs", "fn a() {}");
        write(dir.path(), "vendor/skip.rs", "fn b() {}");
        write(dir.path(), "generated.rs", "fn c() {}");
        write(dir.path(), "ignored/d.rs", "fn d() {}");
        write(dir.path(), ".gitignore", "ignored/\n");
        let files = scan_files(
            dir.path(),
            &["vendor/**".to_string(), "generated.rs".to_string()],
            &all_languages(),
        )
        .unwrap();
        let paths: Vec<&str> = files.iter().map(|f| f.relative_path.as_str()).collect();
        assert_eq!(paths, vec!["src/keep.rs"]);
    }

    #[test]
    fn relative_paths_use_forward_slashes() {
        let dir = tempfile::tempdir().unwrap();
        write(dir.path(), "nested/deep/file.go", "package x");
        let files = scan_files(dir.path(), &[], &all_languages()).unwrap();
        assert_eq!(files[0].relative_path, "nested/deep/file.go");
    }
}
