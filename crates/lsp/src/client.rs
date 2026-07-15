//! Minimal blocking LSP client: Content-Length framed JSON-RPC over a child
//! process's stdio. Requests are sequential — adequate for batch enrichment
//! passes; an async client only pays off once requests overlap.

use std::io::{BufRead, BufReader, Read, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

use anyhow::{Context, Result, bail};
use serde_json::{Value, json};

pub struct LspClient {
    child: Child,
    stdin: ChildStdin,
    reader: BufReader<ChildStdout>,
    next_id: i64,
    /// The server's advertised capabilities from the `initialize` response.
    capabilities: Value,
}

impl LspClient {
    /// Spawn a language server and complete the `initialize` handshake with
    /// `root` as the workspace folder.
    pub fn start(command: &str, args: &[String], root: &Path) -> Result<LspClient> {
        let mut child = Command::new(command)
            .args(args)
            .current_dir(root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("spawning language server {command:?}"))?;
        let stdin = child.stdin.take().context("language server has no stdin")?;
        let stdout = child
            .stdout
            .take()
            .context("language server has no stdout")?;
        let mut client = LspClient {
            child,
            stdin,
            reader: BufReader::new(stdout),
            next_id: 0,
            capabilities: Value::Null,
        };
        let root_uri = file_uri(root);
        let initialized = client.request(
            "initialize",
            json!({
                "processId": Value::Null,
                "rootUri": root_uri,
                "workspaceFolders": [{"uri": root_uri, "name": "codeindex"}],
                "capabilities": {
                    "textDocument": {
                        "hover": {"contentFormat": ["markdown", "plaintext"]},
                        "definition": {},
                        "callHierarchy": {}
                    }
                }
            }),
        )?;
        client.capabilities = initialized
            .get("capabilities")
            .cloned()
            .unwrap_or(Value::Null);
        client.notify("initialized", json!({}))?;
        Ok(client)
    }

    /// Whether the server advertised support for a capability, by its key in
    /// the `initialize` response (e.g. `callHierarchyProvider`).
    pub fn supports(&self, capability: &str) -> bool {
        match self.capabilities.get(capability) {
            None | Some(Value::Null) | Some(Value::Bool(false)) => false,
            Some(_) => true,
        }
    }

    pub fn notify(&mut self, method: &str, params: Value) -> Result<()> {
        self.send(&json!({"jsonrpc": "2.0", "method": method, "params": params}))
    }

    /// Send a request and block until its response arrives. Server-initiated
    /// requests received in the meantime are acknowledged with `null` so the
    /// server never stalls waiting on the client; notifications are ignored.
    pub fn request(&mut self, method: &str, params: Value) -> Result<Value> {
        self.next_id += 1;
        let id = self.next_id;
        self.send(&json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params}))?;
        loop {
            let message = self.read_message()?;
            if message.get("method").is_none()
                && message.get("id").and_then(Value::as_i64) == Some(id)
            {
                if let Some(error) = message.get("error") {
                    bail!("{method} failed: {error}");
                }
                return Ok(message.get("result").cloned().unwrap_or(Value::Null));
            }
            if let (Some(request_id), Some(_)) = (message.get("id"), message.get("method")) {
                let request_id = request_id.clone();
                self.send(&json!({"jsonrpc": "2.0", "id": request_id, "result": Value::Null}))?;
            }
        }
    }

    pub fn shutdown(mut self) -> Result<()> {
        let _ = self.request("shutdown", Value::Null);
        let _ = self.notify("exit", Value::Null);
        let _ = self.child.wait();
        Ok(())
    }

    fn send(&mut self, value: &Value) -> Result<()> {
        let body = serde_json::to_string(value)?;
        write!(self.stdin, "Content-Length: {}\r\n\r\n{body}", body.len())?;
        self.stdin.flush()?;
        Ok(())
    }

    fn read_message(&mut self) -> Result<Value> {
        let mut content_length: Option<usize> = None;
        loop {
            let mut line = String::new();
            let read = self.reader.read_line(&mut line)?;
            if read == 0 {
                bail!("language server closed its output");
            }
            let line = line.trim_end();
            if line.is_empty() {
                break;
            }
            if let Some(rest) = line.strip_prefix("Content-Length:") {
                content_length = Some(rest.trim().parse()?);
            }
        }
        let length = content_length.context("message missing Content-Length header")?;
        let mut buffer = vec![0_u8; length];
        self.reader.read_exact(&mut buffer)?;
        Ok(serde_json::from_slice(&buffer)?)
    }
}

/// A `file://` URI for an absolute path. Minimal encoding: spaces only, which
/// covers real project layouts without pulling in a URL crate.
pub fn file_uri(path: &Path) -> String {
    format!("file://{}", path.display().to_string().replace(' ', "%20"))
}

/// The absolute path of a `file://` URI produced by [`file_uri`] or a server.
pub fn uri_to_path(uri: &str) -> Option<std::path::PathBuf> {
    uri.strip_prefix("file://")
        .map(|rest| std::path::PathBuf::from(rest.replace("%20", " ")))
}
