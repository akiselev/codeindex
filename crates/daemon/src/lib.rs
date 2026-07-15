#![forbid(unsafe_code)]

//! The codeindex daemon: a single stateful background process per user that
//! owns warm embedding models and per-project search state for every
//! registered root. daemonkit owns the lifecycle (identity, locks,
//! authenticated streams, shutdown); this crate owns the application
//! protocol and the resident state.
//!
//! The same binary hosts both sides: the CLI calls [`bootstrap_entry`]
//! first, which detects daemonkit's private bootstrap channel and runs the
//! service instead of the CLI when present.

pub mod client;
pub mod pipeline;
pub mod protocol;
pub mod server;
pub mod state;

use anyhow::{Context, Result, anyhow};

/// Run the daemon service when this process was launched as the daemonkit
/// bootstrap child. Returns `false` when this is a normal CLI invocation.
pub fn bootstrap_entry() -> Result<bool> {
    let Some(bootstrap) =
        daemonkit::Bootstrap::detect().map_err(|error| anyhow!("bootstrap detect: {error}"))?
    else {
        return Ok(false);
    };
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .enable_all()
        .build()
        .context("building daemon runtime")?;
    runtime.block_on(server::run(bootstrap))?;
    Ok(true)
}
