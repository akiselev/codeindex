//! System-memory probing for the embed memory guard.
//!
//! Embedding a large corpus can run for hours next to other workloads. The
//! guard lets a run yield instead of pushing the machine into the OOM killer:
//! when available memory drops below a floor, the embed loop releases the
//! model (weights, ONNX Runtime session and its memory arena) and waits for
//! memory to recover before reloading and continuing.

use std::time::Duration;

/// Pause-and-resume thresholds for an embed run, in MiB of *available*
/// system memory (Linux `MemAvailable`: free memory plus reclaimable cache).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryGuard {
    /// Pause (and release the model) when available memory falls below this.
    pub pause_below_mib: u64,
    /// Resume only once available memory is back at or above this. Must be
    /// at least `pause_below_mib`; the gap is hysteresis so a reload does not
    /// immediately re-trigger a pause.
    pub resume_above_mib: u64,
    /// How often to re-check while paused.
    pub poll_interval: Duration,
}

impl MemoryGuard {
    pub fn new(pause_below_mib: u64, resume_above_mib: u64) -> Self {
        Self {
            pause_below_mib,
            resume_above_mib: resume_above_mib.max(pause_below_mib),
            poll_interval: Duration::from_secs(5),
        }
    }

    /// `Some(available)` when memory is below the pause floor.
    pub fn should_pause(&self) -> Option<u64> {
        available_memory_mib().filter(|available| *available < self.pause_below_mib)
    }

    /// Whether a paused run may resume.
    pub fn may_resume(&self) -> bool {
        available_memory_mib().is_none_or(|available| available >= self.resume_above_mib)
    }
}

/// Available system memory in MiB, or `None` where it cannot be measured
/// (non-Linux platforms, or `/proc` unavailable). Callers treat `None` as
/// "no pressure", so the guard degrades to a no-op rather than stalling.
pub fn available_memory_mib() -> Option<u64> {
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    parse_mem_available_kib(&meminfo).map(|kib| kib / 1024)
}

fn parse_mem_available_kib(meminfo: &str) -> Option<u64> {
    meminfo
        .lines()
        .find_map(|line| line.strip_prefix("MemAvailable:"))
        .and_then(|rest| rest.split_whitespace().next())
        .and_then(|value| value.parse().ok())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_mem_available() {
        let meminfo = "MemTotal:       65000000 kB\nMemFree:  1000 kB\n\
                       MemAvailable:   17500000 kB\nBuffers: 10 kB\n";
        assert_eq!(parse_mem_available_kib(meminfo), Some(17_500_000));
        assert_eq!(parse_mem_available_kib("MemTotal: 1 kB\n"), None);
    }

    #[test]
    fn resume_threshold_never_below_pause() {
        let guard = MemoryGuard::new(4096, 1024);
        assert_eq!(guard.resume_above_mib, 4096);
    }
}
