//! `XCL_WATCH_AHEAD=1`: most of the look for CPU writes before each title submit
//! (`refresh_once`: `GetWriteWatch` with reset over the watched ranges) is done ahead of the
//! submit, on a thread of its own. `XCL_WATCH_AHEAD=2` also takes the guards of physical-page
//! views (`XCL_PAGE_ALIAS`) ahead; with 1 they stay with the submit (a view taken ahead is
//! guarded again early, and a second write to it before the submit faults again).
//!
//! The thread makes one pass after each submit's look, timed for shortly before the next submit:
//! `XCL_WATCH_AHEAD_US` after it, or by default 60 % of the recent time between submits
//! (0.2 – 20 ms). The pages it finds are kept; the next submit takes them and looks once more
//! itself for what was written since. The cost of a look grows with the written pages it resets
//! (a retail game's run: 1.1 ms a submit, about 0.7 µs a page), so the submit's own look then has
//! mostly the pages written since the pass. Its fixed cost (walking the ranges) is paid twice.
//!
//! What it relies on: a page the kernel reports and resets is reported again when written again
//! (`GetWriteWatch` with `WRITE_WATCH_FLAG_RESET`), and a guarded view is guarded again before
//! its page is reported (`guards::take_dirty`). The pass holds `PASS` from its look until its
//! pages are kept, and the submit holds it from its own look until it has taken them, so no page
//! a pass reset is missing from the submit that follows it. The bytes are read at the submit, as
//! before (`sparse::before_execute`, the texture uploads): every CPU write made before the
//! submit's own look is in what it uploads – the guarantee the look without the pass gives.
//!
//! The idea of asking the kernel ahead of the submission follows VIctory-NVIDIA's
//! `crates/victory-gpu/src/watch.rs` (`ask_ahead` / `ask_once`, trustno1-coder/VIctory-NVIDIA
//! @9ce3cf9e, GPL-3.0-or-later); no code is taken from it.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::super::wait_stats::{self, Stat};

/// 0: off, 1: write-watch ahead, 2: the guards too
fn mode() -> u8 {
    static M: OnceLock<u8> = OnceLock::new();
    *M.get_or_init(|| {
        let m = match std::env::var("XCL_WATCH_AHEAD").as_deref().map(str::trim) {
            Ok("1") => 1,
            Ok("2") => 2,
            _ => 0,
        };
        if m > 0 {
            let when = fixed_delay().map_or("60 % of the time between submits after each".to_string(), |d| format!("{} µs after each", d.as_micros()));
            crate::xlog_info!("title memory: XCL_WATCH_AHEAD={m} – CPU writes are looked for ahead of the submits ({when}{})", if m == 2 { ", guards too" } else { "" });
        }
        m
    })
}

pub(super) fn on() -> bool {
    mode() > 0
}

fn fixed_delay() -> Option<Duration> {
    static D: OnceLock<Option<Duration>> = OnceLock::new();
    *D.get_or_init(|| std::env::var("XCL_WATCH_AHEAD_US").ok().and_then(|v| v.trim().parse().ok()).map(Duration::from_micros))
}

/// The time between submits' looks, smoothed (µs), and when the last one looked.
struct Pace {
    gap_us: f64,
    last: Option<Instant>,
}

static PACE: Mutex<Pace> = Mutex::new(Pace { gap_us: 0.0, last: None });

/// When the pass after a look at `now` is made: `fixed_delay`, or 60 % of the smoothed gap.
fn delay_after(now: Instant) -> Duration {
    let mut p = PACE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(last) = p.last.replace(now) {
        let gap = now.saturating_duration_since(last).as_micros() as f64;
        p.gap_us = if p.gap_us == 0.0 { gap } else { p.gap_us * 0.875 + gap * 0.125 };
    }
    fixed_delay().unwrap_or_else(|| pace_delay(p.gap_us))
}

fn pace_delay(gap_us: f64) -> Duration {
    Duration::from_micros((gap_us * 0.6).clamp(200.0, 20_000.0) as u64)
}

/// The pages the passes found since the last submit took them.
struct Found {
    pages: Vec<usize>,
}

static PASS: Mutex<Found> = Mutex::new(Found { pages: Vec::new() });
/// a submit looked: the thread makes its next pass this long after
static KICKED: Mutex<Option<Duration>> = Mutex::new(None);
static KICK: Condvar = Condvar::new();
static THREAD: AtomicBool = AtomicBool::new(false);

fn pass_lock() -> std::sync::MutexGuard<'static, Found> {
    PASS.lock().unwrap_or_else(|e| e.into_inner())
}

/// The submit's look (`look`: the look without the pass) and the passes' pages, into `dirty`.
pub(super) fn take_with_final_pass(dirty: &mut Vec<usize>, look: impl FnOnce(&mut Vec<usize>)) {
    let mut found = pass_lock();
    look(dirty);
    wait_stats::add(Stat::AheadPagesUsed, found.pages.len() as u64);
    dirty.append(&mut found.pages);
}

/// After a submit's look: the thread makes a pass ahead of the next one.
pub(super) fn kick() {
    if !THREAD.swap(true, Ordering::AcqRel) && std::thread::Builder::new().name("xcl-watch-ahead".into()).spawn(thread).is_err() {
        THREAD.store(false, Ordering::Release);
        return;
    }
    *KICKED.lock().unwrap_or_else(|e| e.into_inner()) = Some(delay_after(Instant::now()));
    KICK.notify_one();
}

fn thread() {
    // a submit waits for a pass that holds `PASS`: the pass is not to be put off for the title's
    // own threads
    unsafe {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_ABOVE_NORMAL};
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);
    }
    loop {
        let after = {
            let mut k = KICKED.lock().unwrap_or_else(|e| e.into_inner());
            loop {
                if let Some(d) = k.take() {
                    break d;
                }
                k = KICK.wait(k).unwrap_or_else(|e| e.into_inner());
            }
        };
        std::thread::sleep(after);
        unsafe { pass() };
    }
}

/// One pass: the look the submit makes (write-watch; with mode 2 the guards too), its pages
/// kept for the next submit.
unsafe fn pass() {
    let Some(u) = crate::gpu::uma::get() else { return };
    // the ranges first (they take other locks), then the look under `PASS`
    let ranges = super::cached_watch_ranges(u);
    let mut page = 0x1000u32;
    let mut found = pass_lock();
    let t0 = Instant::now();
    let before = found.pages.len();
    if mode() == 2 {
        super::take_all_written(&ranges, &mut found.pages, &mut page);
    } else {
        super::take_watched(&ranges, &mut found.pages, &mut page);
    }
    let n = found.pages.len() - before;
    drop(found);
    wait_stats::add(Stat::AheadPasses, 1);
    wait_stats::add(Stat::AheadPages, n as u64);
    wait_stats::add_us(Stat::AheadUs, t0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{VirtualAlloc, VirtualFree, MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_READWRITE};
    const MEM_WRITE_WATCH: u32 = 0x20_0000;

    /// A page written before a pass, one written before and after it, and one written only after
    /// it all reach the submit's look; a pass's pages are not given to a second submit.
    /// (Write-watch only: the guards' state is the process's, and other tests use it.)
    #[test]
    fn a_submit_gets_the_pages_of_the_pass_and_those_written_after_it() {
        unsafe {
            const N: usize = 8;
            let base = VirtualAlloc(std::ptr::null(), N * 4096, MEM_RESERVE | MEM_COMMIT | MEM_WRITE_WATCH, PAGE_READWRITE) as usize;
            assert_ne!(base, 0);
            let ranges = [(base, N * 4096)];
            let mine = |v: &mut Vec<usize>| {
                v.retain(|&p| p >= base && p < base + N * 4096);
                v.sort_unstable();
                v.dedup();
            };
            let mut page = 0x1000u32;
            let mut d = Vec::new();
            super::super::take_watched(&ranges, &mut d, &mut page);

            ((base + 4096) as *mut u8).write_volatile(1);
            ((base + 5 * 4096) as *mut u8).write_volatile(1);
            {
                // the pass, as the thread makes it
                let mut found = pass_lock();
                super::super::take_watched(&ranges, &mut found.pages, &mut page);
            }
            ((base + 5 * 4096) as *mut u8).write_volatile(2);
            ((base + 3 * 4096) as *mut u8).write_volatile(1);
            let mut dirty = Vec::new();
            take_with_final_pass(&mut dirty, |d| super::super::take_watched(&ranges, d, &mut page));
            mine(&mut dirty);
            assert_eq!(dirty, vec![base + 4096, base + 3 * 4096, base + 5 * 4096]);
            let mut again = Vec::new();
            take_with_final_pass(&mut again, |d| super::super::take_watched(&ranges, d, &mut page));
            mine(&mut again);
            assert!(again.is_empty(), "{again:x?}");
            assert_ne!(VirtualFree(base as *mut _, 0, MEM_RELEASE), 0);
        }
    }

    #[test]
    fn the_pass_comes_at_60_percent_of_the_gap_within_bounds() {
        assert_eq!(pace_delay(5000.0), Duration::from_micros(3000));
        assert_eq!(pace_delay(10.0), Duration::from_micros(200));
        assert_eq!(pace_delay(1e9), Duration::from_micros(20_000));
    }
}
