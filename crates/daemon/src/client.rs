//! Blocking client used by the CLI: ensure/attach the daemon through
//! daemonkit, then speak the framed protocol over one short-lived
//! authenticated stream per command.

use anyhow::{Context, Result, anyhow, bail};
use daemonkit::{Daemon, DaemonClient, DaemonSpec, Spawn};
use serde::Serialize;
use serde_json::Value;

use crate::protocol::{self, Request, Response};

/// Reverse-DNS identity of the codeindex daemon instance (one per user;
/// `CODEINDEX_ISOLATION` scopes tests away from the real instance).
const APP_ID: &str = "com.akiselev.codeindex";

fn spec() -> Result<DaemonSpec> {
    let mut spec = DaemonSpec::new(APP_ID).map_err(|error| anyhow!("daemon spec: {error}"))?;
    if let Ok(isolation) = std::env::var("CODEINDEX_ISOLATION")
        && !isolation.is_empty()
    {
        let isolation = daemonkit::IsolationId::try_from(isolation.as_str())
            .map_err(|error| anyhow!("CODEINDEX_ISOLATION: {error}"))?;
        spec = spec.isolation(isolation);
    }
    Ok(spec)
}

fn spawn_recipe() -> Result<Spawn> {
    let mut spawn = Spawn::current_exe()
        .map_err(|error| anyhow!("resolving current executable: {error}"))?
        .arg("__daemon");
    // The bootstrap channel itself is daemonkit's (env-validated); these are
    // plain settings the daemon must share with the launching CLI.
    for key in ["CODEINDEX_DATA_DIR", "XDG_DATA_HOME", "HF_TOKEN"] {
        if let Some(value) = std::env::var_os(key) {
            spawn = spawn.env(key, value);
        }
    }
    Ok(spawn)
}

pub struct Connection {
    runtime: tokio::runtime::Runtime,
    stream: daemonkit::AuthenticatedStream,
    next_id: u64,
}

impl Connection {
    /// Attach to a running daemon or start one (daemonkit serializes
    /// concurrent starts and authenticates the stream).
    pub fn ensure() -> Result<Connection> {
        let runtime = client_runtime()?;
        let daemon = Daemon::embedded(spec()?, spawn_recipe()?)
            .map_err(|error| anyhow!("configuring daemon: {error}"))?;
        let stream = runtime.block_on(async {
            match ensure_connect(&daemon).await {
                // daemonkit 0.1.0 cannot replace a generation wedged in
                // QUIESCING (an abandoned drain leaves it there forever;
                // fixed upstream on the first-consumer-hardening branch).
                // Consumer-side recovery: force-stop the wedged instance
                // and ensure again.
                Err(daemonkit::Error::BusyQuiescing) => {
                    let _ = daemon.stop().await;
                    ensure_connect(&daemon).await
                }
                other => other,
            }
        });
        let stream = stream.map_err(startup_error)?;
        Ok(Connection {
            runtime,
            stream,
            next_id: 0,
        })
    }

    /// Connect only if a compatible daemon is already running.
    pub fn attach() -> Result<Option<Connection>> {
        let runtime = client_runtime()?;
        let client = DaemonClient::<daemonkit::Embedded>::embedded(spec()?)
            .map_err(|error| anyhow!("configuring daemon client: {error}"))?;
        let stream = runtime.block_on(async {
            match client.attach().await? {
                Some(instance) => instance.connect().await.map(Some),
                None => Ok(None),
            }
        });
        match stream {
            Ok(Some(stream)) => Ok(Some(Connection {
                runtime,
                stream,
                next_id: 0,
            })),
            Ok(None) => Ok(None),
            Err(daemonkit::Error::Unavailable { .. }) => Ok(None),
            Err(error) => Err(anyhow!("attaching to daemon: {error}")),
        }
    }

    pub fn call<P: Serialize>(&mut self, method: &str, params: &P) -> Result<Value> {
        self.next_id += 1;
        let request = Request {
            id: self.next_id,
            method: method.to_string(),
            params: serde_json::to_value(params)?,
        };
        let payload = serde_json::to_vec(&request)?;
        let stream = &mut self.stream;
        let response: Response = self.runtime.block_on(async {
            protocol::write_frame(stream, &payload).await?;
            let bytes = protocol::read_frame(stream)
                .await?
                .ok_or_else(|| std::io::Error::from(std::io::ErrorKind::UnexpectedEof))?;
            Ok::<_, std::io::Error>(serde_json::from_slice(&bytes)?)
        })?;
        if response.id != request.id {
            bail!(
                "daemon answered request {} while {} was pending",
                response.id,
                request.id
            );
        }
        match (response.result, response.error) {
            (Some(result), None) => Ok(result),
            (_, Some(error)) => Err(anyhow!(error)),
            (None, None) => bail!("daemon returned an empty response"),
        }
    }
}

/// Stop the daemon if it is running. Returns a human-readable outcome.
pub fn stop() -> Result<String> {
    let runtime = client_runtime()?;
    let client = DaemonClient::<daemonkit::Embedded>::embedded(spec()?)
        .map_err(|error| anyhow!("configuring daemon client: {error}"))?;
    match runtime.block_on(client.stop()) {
        Ok(outcome) => Ok(format!("{outcome:?}")),
        Err(daemonkit::Error::Unavailable { .. }) => Ok("not running".to_string()),
        Err(error) => Err(anyhow!("stopping daemon: {error}")),
    }
}

/// Lifecycle status without creating an instance.
pub fn lifecycle_status() -> Result<String> {
    let runtime = client_runtime()?;
    let client = DaemonClient::<daemonkit::Embedded>::embedded(spec()?)
        .map_err(|error| anyhow!("configuring daemon client: {error}"))?;
    let status = runtime
        .block_on(client.status())
        .map_err(|error| anyhow!("querying daemon status: {error}"))?;
    Ok(format!("{status:?}"))
}

async fn ensure_connect(
    daemon: &Daemon<daemonkit::Embedded>,
) -> Result<daemonkit::AuthenticatedStream, daemonkit::Error> {
    let instance = daemon.ensure().await?;
    instance.connect().await
}

fn client_runtime() -> Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .context("building client runtime")
}

fn startup_error(error: daemonkit::Error) -> anyhow::Error {
    match &error {
        daemonkit::Error::Startup { diagnostics, .. } => {
            anyhow!("daemon failed to start: {diagnostics:?}")
        }
        _ => anyhow!("ensuring daemon: {error}"),
    }
}
