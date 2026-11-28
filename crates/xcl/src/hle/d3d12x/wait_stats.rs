//! Where the title's threads wait for the GPU in XCL, counted for the `XCL_ACTIVITY` line
//! (`waits (2 s): …`). Every counter only goes up; the activity thread prints the differences.
//!
//! - `Wait32/64BitValueX` at a submit whose memory did not hold yet: bridged to a GPU
//!   `queue.Wait` on the per-address fences (`list_x::try_gpu_wait`), watched on the CPU with a
//!   GPU `queue.Wait` (`gpu_watch`, `XCL_GPU_SIDE_WAITS=1`), or polled on the CPU by the
//!   submitting thread (the time it spent, and how many ran out of time);
//! - `Write64` signals put on the CPU to break a cycle, GPU waits the watchdog released;
//! - the sparse windows' CPU waits for their own queue (readbacks after a submit, upload slots);
//! - `PresentX` waiting for a present slot;
//! - binds waiting for a compute pipeline being built;
//! - the time submits spend in XCL before the driver has the lists (`title_textures::refresh`:
//!   write-watch and uploads) and in all of `ExecuteCommandLists`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Stat {
    /// a wait bridged to the GPU (per-address fence `queue.Wait`, same-queue order, released)
    Bridged,
    /// a wait handed to `gpu_watch` (GPU `queue.Wait`, the CPU signals once the memory holds)
    Watched,
    /// µs from a watched wait's arming to the CPU's signal
    WatchedUs,
    /// watched waits signalled after the timeout without the memory holding
    WatchedTimeouts,
    /// a wait polled on the submitting thread
    CpuPolled,
    /// µs the submitting threads spent polling
    CpuPolledUs,
    /// polls that ran out of time
    CpuTimeouts,
    /// a same-batch self-wait skipped
    CycleSkips,
    /// a Write64 signal done on the CPU to break a GPU cycle
    CycleBreaks,
    /// GPU waits the watchdog released on the CPU
    WatchdogReleases,
    /// sparse windows: CPU waits for their own queue, and their µs
    SparseWaits,
    SparseWaitUs,
    /// PresentX: waits for a present slot, and their µs
    PresentWaits,
    PresentWaitUs,
    /// binds that waited for a pipeline being built, and their µs
    PsoWaits,
    PsoWaitUs,
    /// µs of `title_textures::refresh` in the title's submits
    RefreshUs,
    /// µs of the title's `ExecuteCommandLists` calls in all
    SubmitUs,
    /// µs of the look for CPU writes in the submits (write-watch and guards; part of `RefreshUs`)
    WatchUs,
    /// `XCL_WATCH_AHEAD`: passes ahead of the submits, the pages they found, their µs, and the
    /// pages submits took from them
    AheadPasses,
    AheadPages,
    AheadUs,
    AheadPagesUsed,
    /// `XCL_RELAY_READBACK`: copies back handed to the relay's thread
    RelayJobs,
    /// `XCL_WC_NO_BACK`: KiB of GPU writes into write-combined memory not copied back
    WcLeftKb,
}

pub const COUNT: usize = Stat::WcLeftKb as usize + 1;

static STATS: [AtomicU64; COUNT] = [const { AtomicU64::new(0) }; COUNT];

pub fn add(s: Stat, n: u64) {
    STATS[s as usize].fetch_add(n, Ordering::Relaxed);
}

pub fn add_us(s: Stat, since: Instant) {
    add(s, since.elapsed().as_micros() as u64);
}

pub fn snapshot() -> [u64; COUNT] {
    STATS.each_ref().map(|a| a.load(Ordering::Relaxed))
}

/// The `waits (…)` line for the differences `d` (`snapshot` minus an earlier one).
pub fn line(d: &[u64; COUNT]) -> String {
    let ms = |s: Stat| d[s as usize] / 1000;
    let n = |s: Stat| d[s as usize];
    format!(
        "Wait32/64 at submit: {} GPU-bridged, {} GPU-watched ({} ms to release, {} timed out), {} CPU-polled ({} ms on the submit thread, {} timed out), {} self-waits skipped; {} Write64 cycle breaks, {} watchdog releases; \
         sparse CPU waits {} ({} ms); PresentX slot waits {} ({} ms); pipeline-build waits {} ({} ms); submits {} ms in XCL ({} ms write-watch and uploads, {} ms of it the look for writes); looks ahead {} ({} ms, {} pages found, {} taken by submits); copies back relayed {}, {} KiB write-combined not copied back",
        n(Stat::Bridged),
        n(Stat::Watched),
        ms(Stat::WatchedUs),
        n(Stat::WatchedTimeouts),
        n(Stat::CpuPolled),
        ms(Stat::CpuPolledUs),
        n(Stat::CpuTimeouts),
        n(Stat::CycleSkips),
        n(Stat::CycleBreaks),
        n(Stat::WatchdogReleases),
        n(Stat::SparseWaits),
        ms(Stat::SparseWaitUs),
        n(Stat::PresentWaits),
        ms(Stat::PresentWaitUs),
        n(Stat::PsoWaits),
        ms(Stat::PsoWaitUs),
        ms(Stat::SubmitUs),
        ms(Stat::RefreshUs),
        ms(Stat::WatchUs),
        n(Stat::AheadPasses),
        ms(Stat::AheadUs),
        n(Stat::AheadPages),
        n(Stat::AheadPagesUsed),
        n(Stat::RelayJobs),
        n(Stat::WcLeftKb),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_line_names_every_counter_in_its_place() {
        let mut d = [0u64; COUNT];
        for (i, v) in d.iter_mut().enumerate() {
            *v = (i as u64 + 1) * 1000;
        }
        let l = line(&d);
        assert!(l.starts_with("Wait32/64 at submit: 1000 GPU-bridged, 2000 GPU-watched (3 ms to release, 4000 timed out), 5000 CPU-polled (6 ms"), "{l}");
        assert!(l.contains("submits 18 ms in XCL (17 ms write-watch and uploads, 19 ms of it the look for writes)"), "{l}");
        assert!(l.ends_with("looks ahead 20000 (22 ms, 21000 pages found, 23000 taken by submits); copies back relayed 24000, 25000 KiB write-combined not copied back"), "{l}");
    }
}
