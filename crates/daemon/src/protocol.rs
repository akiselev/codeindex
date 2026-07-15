//! The application protocol spoken over daemonkit's authenticated streams:
//! JSON-RPC-shaped requests/responses with LSP-style `Content-Length`
//! framing. daemonkit owns transport authentication; this layer owns only
//! message shapes, so it is trivially bridgeable to MCP later.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub const PROTOCOL_VERSION: u32 = 1;

/// Upper bound for one framed message. Search responses are bounded by
/// `limit`; nothing legitimate approaches this.
const MAX_FRAME_BYTES: usize = 32 * 1024 * 1024;

#[derive(Debug, Serialize, Deserialize)]
pub struct Request {
    pub id: u64,
    pub method: String,
    #[serde(default)]
    pub params: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Response {
    pub id: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

pub async fn write_frame<W: AsyncWrite + Unpin>(
    writer: &mut W,
    payload: &[u8],
) -> std::io::Result<()> {
    let header = format!("Content-Length: {}\r\n\r\n", payload.len());
    writer.write_all(header.as_bytes()).await?;
    writer.write_all(payload).await?;
    writer.flush().await
}

/// Read one framed message. `Ok(None)` on clean EOF before the first header
/// byte (the peer hung up between messages).
pub async fn read_frame<R: AsyncRead + Unpin>(reader: &mut R) -> std::io::Result<Option<Vec<u8>>> {
    let mut header = Vec::with_capacity(64);
    let mut byte = [0u8; 1];
    loop {
        match reader.read(&mut byte).await? {
            0 => {
                if header.is_empty() {
                    return Ok(None);
                }
                return Err(std::io::Error::new(
                    std::io::ErrorKind::UnexpectedEof,
                    "connection closed mid-header",
                ));
            }
            _ => header.push(byte[0]),
        }
        if header.ends_with(b"\r\n\r\n") {
            break;
        }
        if header.len() > 4096 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "oversized frame header",
            ));
        }
    }
    let text = std::str::from_utf8(&header)
        .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "non-utf8 header"))?;
    let length = text
        .lines()
        .find_map(|line| {
            line.split_once(':').and_then(|(name, value)| {
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse::<usize>().ok())
                    .flatten()
            })
        })
        .ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::InvalidData, "missing Content-Length")
        })?;
    if length > MAX_FRAME_BYTES {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "frame exceeds maximum size",
        ));
    }
    let mut payload = vec![0u8; length];
    reader.read_exact(&mut payload).await?;
    Ok(Some(payload))
}

// ---- method payloads ----------------------------------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct AddParams {
    pub root: PathBuf,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AddResult {
    pub label: String,
    pub root: PathBuf,
    pub db: PathBuf,
    /// `indexing` when a job was spawned, `already-registered` otherwise.
    pub job: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RemoveParams {
    pub needle: String,
    #[serde(default)]
    pub purge: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RemoveResult {
    pub label: String,
    pub root: PathBuf,
    /// Path deleted by `--purge`, when it was safe to delete.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub purged: Option<PathBuf>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct NeedleParams {
    pub needle: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JobState {
    pub phase: String,
    pub detail: String,
    pub finished: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ProjectSummary {
    pub label: String,
    pub root: PathBuf,
    pub db: PathBuf,
    /// `None` until the first index run creates the database.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub units: Option<i64>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub spaces: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub job: Option<JobState>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct StatusResult {
    pub version: String,
    pub protocol: u32,
    pub pid: u32,
    pub uptime_seconds: u64,
    pub projects: Vec<ProjectSummary>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
pub struct SearchParams {
    /// Label or path; when absent, `cwd` containment resolves the project.
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub cwd: Option<PathBuf>,
    pub query: String,
    /// Space id; defaults to the project's `[search] default_space`, falling
    /// back to lexical-only retrieval when no space is available.
    #[serde(default)]
    pub space: Option<String>,
    #[serde(default)]
    pub task: Option<String>,
    #[serde(default)]
    pub instruction: Option<String>,
    #[serde(default)]
    pub filter: Option<String>,
    #[serde(default = "default_limit")]
    pub limit: usize,
    /// `hybrid` | `dense` | `lexical`; defaults per project config.
    #[serde(default)]
    pub retrieval: Option<String>,
    /// `auto` | `off` | `always`.
    #[serde(default)]
    pub compress: Option<String>,
    #[serde(default)]
    pub no_graph: bool,
    /// `include` | `exclude` | `only`; resolved by the client from flags and
    /// the folder-override chain (the daemon does not know the caller's cwd
    /// position within the tree).
    #[serde(default)]
    pub tests: Option<String>,
}

fn default_limit() -> usize {
    10
}

/// Search output. This is the single JSON shape shared by the direct CLI
/// path, the daemon protocol, and any future MCP surface.
#[derive(Debug, Serialize, Deserialize)]
pub struct SearchResults {
    pub query: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compressed_query: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub space: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<codeindex_core::EmbeddingTask>,
    pub matched: usize,
    pub hits: Vec<SearchHit>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SearchHit {
    pub selector: String,
    pub score: f32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rerank_score: Option<f32>,
    /// `source#rank` contributions from each fused list.
    pub sources: Vec<String>,
    pub project: String,
    pub path: String,
    pub lines: [usize; 2],
    pub language: String,
    pub kind: String,
    pub name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn frames_round_trip() {
        let mut buffer = Vec::new();
        write_frame(&mut buffer, br#"{"id":1}"#).await.unwrap();
        write_frame(&mut buffer, br#"{"id":2}"#).await.unwrap();
        let mut cursor = std::io::Cursor::new(buffer);
        assert_eq!(
            read_frame(&mut cursor).await.unwrap().unwrap(),
            br#"{"id":1}"#
        );
        assert_eq!(
            read_frame(&mut cursor).await.unwrap().unwrap(),
            br#"{"id":2}"#
        );
        assert!(read_frame(&mut cursor).await.unwrap().is_none());
    }

    #[tokio::test]
    async fn header_case_is_insensitive_and_garbage_is_rejected() {
        let mut cursor =
            std::io::Cursor::new(b"content-length: 2\r\nX-Other: y\r\n\r\nok".to_vec());
        assert_eq!(read_frame(&mut cursor).await.unwrap().unwrap(), b"ok");
        let mut bad = std::io::Cursor::new(b"Content-Type: json\r\n\r\n{}".to_vec());
        assert!(read_frame(&mut bad).await.is_err());
        let mut truncated = std::io::Cursor::new(b"Content-Length: 5\r\n".to_vec());
        assert!(read_frame(&mut truncated).await.is_err());
    }
}
