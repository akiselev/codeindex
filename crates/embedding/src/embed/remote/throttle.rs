//! Shared admission control for a remote backend's request workers.
//!
//! One [`RateGate`] is shared by every worker of a backend. It does two
//! things the per-request retry loop cannot:
//! - **Global pause.** A 429 means the *account* (or upstream) is over its
//!   limit, not just that request. The gate records a pause deadline — the
//!   server's `Retry-After`, else a jittered exponential delay — and every
//!   worker waits it out before sending again, instead of each hammering the
//!   API on its own schedule.
//! - **Adaptive concurrency (AIMD).** Each 429 halves the number of requests
//!   allowed in flight (once per pause window, so one burst of 429s from
//!   requests already in flight counts as one signal); every `limit`
//!   consecutive successes raise it by one, back up to the configured
//!   ceiling.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Exponential backoff with equal jitter: half the step is fixed, half is
/// random, so concurrent retries spread out but never retry immediately.
#[derive(Debug, Clone, Copy)]
pub struct Backoff {
    pub initial: Duration,
    pub max: Duration,
}

impl Backoff {
    pub fn delay(&self, attempt: u32) -> Duration {
        let step = self
            .initial
            .saturating_mul(1u32 << attempt.min(16))
            .min(self.max);
        let half = step / 2;
        half + half.mul_f64(jitter_unit())
    }
}

/// A cheap uniform-ish value in `[0, 1)`; jitter needs spread, not quality.
/// A global counter decorrelates workers that wake in the same nanosecond.
fn jitter_unit() -> f64 {
    static CALLS: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |elapsed| elapsed.subsec_nanos() as u64);
    let mut x = nanos
        ^ CALLS
            .fetch_add(1, Ordering::Relaxed)
            .wrapping_add(1)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    (x % 1_000_000) as f64 / 1_000_000.0
}

#[derive(Debug)]
struct GateState {
    ceiling: usize,
    limit: usize,
    in_flight: usize,
    paused_until: Option<Instant>,
    /// Consecutive 429s without a success, driving the fallback delay.
    rate_limit_streak: u32,
    successes: usize,
    /// Lowest limit reached, for the usage summary.
    floor_reached: usize,
    rate_limited: u64,
}

pub struct RateGate {
    state: Mutex<GateState>,
    changed: Condvar,
    adaptive: bool,
    backoff: Backoff,
}

/// Held while a request is in flight; dropping it without reporting counts
/// as a neutral outcome (transport errors, non-rate-limit failures).
pub struct Permit<'a> {
    gate: &'a RateGate,
    done: bool,
}

impl RateGate {
    pub fn new(ceiling: usize, adaptive: bool, backoff: Backoff) -> Self {
        let ceiling = ceiling.max(1);
        Self {
            state: Mutex::new(GateState {
                ceiling,
                limit: ceiling,
                in_flight: 0,
                paused_until: None,
                rate_limit_streak: 0,
                successes: 0,
                floor_reached: ceiling,
                rate_limited: 0,
            }),
            changed: Condvar::new(),
            adaptive,
            backoff,
        }
    }

    /// Block until a request may be sent: no global pause is active and
    /// fewer than `limit` requests are in flight.
    pub fn acquire(&self) -> Permit<'_> {
        let mut state = self.state.lock().unwrap();
        loop {
            let now = Instant::now();
            if let Some(until) = state.paused_until {
                if now < until {
                    state = self.changed.wait_timeout(state, until - now).unwrap().0;
                    continue;
                }
                state.paused_until = None;
            }
            if state.in_flight < state.limit {
                state.in_flight += 1;
                return Permit {
                    gate: self,
                    done: false,
                };
            }
            state = self.changed.wait(state).unwrap();
        }
    }

    /// Current allowed in-flight requests (tests, diagnostics).
    pub fn limit(&self) -> usize {
        self.state.lock().unwrap().limit
    }

    /// `(429 responses seen, lowest concurrency limit reached)`.
    pub fn summary(&self) -> (u64, usize) {
        let state = self.state.lock().unwrap();
        (state.rate_limited, state.floor_reached)
    }

    fn release(&self, outcome: Outcome) {
        let mut state = self.state.lock().unwrap();
        state.in_flight = state.in_flight.saturating_sub(1);
        match outcome {
            Outcome::Success => {
                state.rate_limit_streak = 0;
                state.successes += 1;
                if self.adaptive && state.limit < state.ceiling && state.successes >= state.limit {
                    state.limit += 1;
                    state.successes = 0;
                }
            }
            Outcome::RateLimited(retry_after) => {
                state.rate_limited += 1;
                state.successes = 0;
                let now = Instant::now();
                // Requests already in flight when the limit was hit report
                // 429 too; only the first in a pause window cuts concurrency.
                let in_window = state.paused_until.is_some_and(|until| now < until);
                if !in_window {
                    if self.adaptive {
                        state.limit = (state.limit / 2).max(1);
                        state.floor_reached = state.floor_reached.min(state.limit);
                    }
                    state.rate_limit_streak += 1;
                }
                let delay = retry_after.unwrap_or_else(|| {
                    self.backoff
                        .delay(state.rate_limit_streak.saturating_sub(1))
                });
                let until = now + delay;
                state.paused_until = Some(state.paused_until.map_or(until, |at| at.max(until)));
            }
            Outcome::Neutral => {}
        }
        drop(state);
        self.changed.notify_all();
    }
}

enum Outcome {
    Success,
    RateLimited(Option<Duration>),
    Neutral,
}

impl Permit<'_> {
    pub fn success(mut self) {
        self.done = true;
        self.gate.release(Outcome::Success);
    }

    /// Report a 429, with the server's `Retry-After` when it sent one.
    pub fn rate_limited(mut self, retry_after: Option<Duration>) {
        self.done = true;
        self.gate.release(Outcome::RateLimited(retry_after));
    }
}

impl Drop for Permit<'_> {
    fn drop(&mut self) {
        if !self.done {
            self.gate.release(Outcome::Neutral);
        }
    }
}

/// Parse `Retry-After`: delta-seconds (possibly fractional, which some
/// gateways send) or an IMF-fixdate HTTP date.
pub fn parse_retry_after(value: &str) -> Option<Duration> {
    let value = value.trim();
    if let Ok(seconds) = value.parse::<f64>() {
        return (seconds.is_finite() && seconds >= 0.0).then(|| Duration::from_secs_f64(seconds));
    }
    let target = parse_http_date(value)?;
    let now = SystemTime::now().duration_since(UNIX_EPOCH).ok()?.as_secs();
    Some(Duration::from_secs(target.saturating_sub(now)))
}

/// `Sun, 06 Nov 1994 08:49:37 GMT` → Unix seconds.
fn parse_http_date(value: &str) -> Option<u64> {
    let mut parts = value.split_whitespace();
    let _weekday = parts.next()?;
    let day: u64 = parts.next()?.parse().ok()?;
    let month = match parts.next()? {
        "Jan" => 1,
        "Feb" => 2,
        "Mar" => 3,
        "Apr" => 4,
        "May" => 5,
        "Jun" => 6,
        "Jul" => 7,
        "Aug" => 8,
        "Sep" => 9,
        "Oct" => 10,
        "Nov" => 11,
        "Dec" => 12,
        _ => return None,
    };
    let year: i64 = parts.next()?.parse().ok()?;
    let mut clock = parts
        .next()?
        .split(':')
        .map(|part| part.parse::<u64>().ok());
    let (hour, minute, second) = (clock.next()??, clock.next()??, clock.next()??);
    // Days from civil (Howard Hinnant's algorithm).
    let (y, m) = if month <= 2 {
        (year - 1, month + 9)
    } else {
        (year, month - 3)
    };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * m as i64 + 2) / 5 + day as i64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146_097 + doe - 719_468;
    u64::try_from(days)
        .ok()
        .map(|days| days * 86_400 + hour * 3600 + minute * 60 + second)
}

#[cfg(test)]
mod tests {
    use super::*;

    const BACKOFF: Backoff = Backoff {
        initial: Duration::from_millis(1),
        max: Duration::from_millis(8),
    };

    #[test]
    fn backoff_grows_caps_and_jitters_within_bounds() {
        for attempt in 0..10 {
            let step = Duration::from_millis(1u64 << attempt.min(3)).min(BACKOFF.max);
            let delay = BACKOFF.delay(attempt);
            assert!(delay >= step / 2 && delay <= step, "{attempt}: {delay:?}");
        }
    }

    #[test]
    fn rate_limits_halve_once_per_window_and_successes_recover() {
        let gate = RateGate::new(8, true, BACKOFF);
        let a = gate.acquire();
        let b = gate.acquire();
        // Two in-flight requests hit the same limit: one signal, one cut.
        a.rate_limited(Some(Duration::from_millis(20)));
        b.rate_limited(Some(Duration::from_millis(20)));
        assert_eq!(gate.limit(), 4);
        assert_eq!(gate.summary(), (2, 4));

        // The pause holds every worker back.
        let started = Instant::now();
        let permit = gate.acquire();
        assert!(started.elapsed() >= Duration::from_millis(15));
        permit.success();

        // `limit` consecutive successes raise the limit by one.
        for _ in 0..3 {
            gate.acquire().success();
        }
        assert_eq!(gate.limit(), 5);
    }

    #[test]
    fn fixed_concurrency_never_adapts_and_limit_bounds_in_flight() {
        let gate = RateGate::new(2, false, BACKOFF);
        gate.acquire().rate_limited(Some(Duration::ZERO));
        assert_eq!(gate.limit(), 2);

        let first = gate.acquire();
        let _second = gate.acquire();
        std::thread::scope(|scope| {
            let waiter = scope.spawn(|| {
                let started = Instant::now();
                gate.acquire().success();
                started.elapsed()
            });
            std::thread::sleep(Duration::from_millis(20));
            drop(first);
            assert!(waiter.join().unwrap() >= Duration::from_millis(15));
        });
    }

    #[test]
    fn retry_after_seconds_and_http_dates() {
        assert_eq!(parse_retry_after("7"), Some(Duration::from_secs(7)));
        assert_eq!(parse_retry_after("1.5"), Some(Duration::from_millis(1500)));
        assert_eq!(parse_retry_after("-1"), None);
        assert_eq!(
            parse_http_date("Sun, 06 Nov 1994 08:49:37 GMT"),
            Some(784_111_777)
        );
        // A date in the past means "now".
        assert_eq!(
            parse_retry_after("Sun, 06 Nov 1994 08:49:37 GMT"),
            Some(Duration::ZERO)
        );
        assert_eq!(parse_retry_after("soon"), None);
    }
}
