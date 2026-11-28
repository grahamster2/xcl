//! Two switches for the copies of GPU writes in the sparse windows back into their CPU view
//! (`readback_after_execute`), which the title's submitting thread waits for:
//!
//! - `XCL_RELAY_READBACK=1`: the copy back is done by a thread of its own ("lands"), not by the
//!   submit. The title queue gets a GPU `Wait` on a relay fence right behind its lists, which the
//!   thread signals once the bytes are in the CPU view – so the title's own `Signal`s after the
//!   submit, and with them its fence waits, complete only after the copy back, as they did. Any
//!   copy of CPU-view bytes to the GPU (`before_execute`, `readback_now`) first waits for the
//!   copies back still going (`land`): no upload reads a CPU view older than the GPU's bytes.
//!   What changes: the submit returns before the copy; a title thread that polls a GPU-written
//!   marker in other memory (not a fence) can see it before the copy back – as it already could
//!   while the submitting thread waited for the copy.
//! - `XCL_WC_NO_BACK=1` (`XCL_WC_NO_BACK_MIN_KB`, default 1024): GPU writes of at least that
//!   size into sparse memory the title reserved write-combined (`PAGE_WRITECOMBINE`) are not
//!   copied back. A console title does not read write-combined memory back with the CPU. Risk:
//!   one that does reads stale bytes there; and a later CPU write to a chunk of such a range
//!   uploads the chunk's CPU bytes over the GPU's.
//!
//! Ported in idea from trustno1-coder/VIctory-NVIDIA @9ce3cf9e (GPL-3.0-or-later): the relay's
//! landing thread (`crates/victory-gpu/src/host.rs` `relay` / `relay_worker`, docs at
//! :2875-2924) and `VICTORY_WC_NO_BACK` (`crates/victory-gpu/src/shadow.rs:2366-2404`,
//! :3456-3466); written for XCL's sparse windows, no code taken, 2026-11-28.

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::Instant;

use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::*;

use super::super::wait_stats::{self, Stat};

pub(super) fn on() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_RELAY_READBACK").is_ok_and(|v| v.trim() == "1");
        if on {
            crate::xlog_info!("sparse windows: XCL_RELAY_READBACK=1 – GPU writes are copied back by a thread of their own, the title queue waits for that on the GPU");
        }
        on
    })
}

struct Job {
    /// the title queue's lists are done when `rb_fence` is there
    rb_fence: ID3D12Fence,
    rb_value: u64,
    ranges: Vec<(usize, usize)>,
    /// signalled once the bytes are back
    relay_value: u64,
}

struct Relay {
    fence: ID3D12Fence,
    device: usize,
    issued: u64,
    jobs: VecDeque<Job>,
    thread: bool,
}

// SAFETY: D3D12 fences are free-threaded.
unsafe impl Send for Relay {}

static RELAY: Mutex<Option<Relay>> = Mutex::new(None);
static WORK: Condvar = Condvar::new();

fn relay() -> std::sync::MutexGuard<'static, Option<Relay>> {
    RELAY.lock().unwrap_or_else(|e| e.into_inner())
}

/// Instead of copying `ranges` back now: a job for the thread. `queue` signalled `rb_fence` =
/// `rb_value` after its lists; it now waits on the GPU for the copy back. false: not done (no
/// fence, the queue refused the wait) – the caller copies back itself.
pub(super) unsafe fn send(dev: &ID3D12Device, queue: &ID3D12CommandQueue, rb_fence: &ID3D12Fence, rb_value: u64, ranges: Vec<(usize, usize)>) -> bool {
    let dev_raw = windows::core::Interface::as_raw(dev) as usize;
    let mut g = relay();
    if g.as_ref().is_some_and(|r| r.device != dev_raw) {
        return false;
    }
    if g.is_none() {
        let Ok(fence) = dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) else { return false };
        *g = Some(Relay { fence, device: dev_raw, issued: 0, jobs: VecDeque::new(), thread: false });
    }
    let r = g.as_mut().unwrap();
    let relay_value = r.issued + 1;
    if queue.Wait(&r.fence, relay_value).is_err() {
        return false;
    }
    r.issued = relay_value;
    r.jobs.push_back(Job { rb_fence: rb_fence.clone(), rb_value, ranges, relay_value });
    wait_stats::add(Stat::RelayJobs, 1);
    if !r.thread {
        r.thread = std::thread::Builder::new().name("xcl-sparse-relay".into()).spawn(worker).is_ok();
        if !r.thread {
            // no thread: undo nothing (the queue waits), land it here and now
            let job = r.jobs.pop_back().unwrap();
            let fence = r.fence.clone();
            drop(g);
            land_job(job, &fence);
            return true;
        }
    }
    WORK.notify_one();
    true
}

unsafe fn wait_fence(fence: &ID3D12Fence, value: u64, event: HANDLE, ms: u32) -> bool {
    if fence.GetCompletedValue() >= value {
        return true;
    }
    let _ = fence.SetEventOnCompletion(value, event);
    windows::Win32::System::Threading::WaitForSingleObject(event, ms);
    fence.GetCompletedValue() >= value
}

fn thread_event() -> HANDLE {
    thread_local! {
        static E: HANDLE = unsafe { windows::Win32::System::Threading::CreateEventW(None, false, false, None).unwrap_or_default() };
    }
    E.with(|e| *e)
}

unsafe fn land_job(job: Job, relay_fence: &ID3D12Fence) {
    // the title's lists first, with no lock held (the submits go on meanwhile)
    if !wait_fence(&job.rb_fence, job.rb_value, thread_event(), 5000) {
        crate::xlog_warn!("sparse windows: a title queue has not reached its copy-back point {} after 5 s – copied back anyway", job.rb_value);
    }
    {
        let mut g = super::lock_state();
        if let Some(s) = g.as_mut() {
            s.readback_ranges(&job.ranges);
        }
    }
    let _ = relay_fence.Signal(job.relay_value);
}

fn worker() {
    loop {
        let (job, fence) = {
            let mut g = relay();
            loop {
                if let Some(r) = g.as_mut() {
                    if let Some(j) = r.jobs.pop_front() {
                        break (j, r.fence.clone());
                    }
                }
                g = WORK.wait(g).unwrap_or_else(|e| e.into_inner());
            }
        };
        unsafe { land_job(job, &fence) };
    }
}

/// Before CPU-view bytes of the sparse windows go to the GPU, or are read for the title: the
/// copies back sent so far are done. Called without `STATE` held (the thread takes it).
pub(super) unsafe fn land() {
    let pending = relay().as_ref().filter(|r| r.issued > 0).map(|r| (r.fence.clone(), r.issued));
    let Some((fence, issued)) = pending else { return };
    if fence.GetCompletedValue() >= issued {
        return;
    }
    let t0 = Instant::now();
    if !wait_fence(&fence, issued, thread_event(), 5000) {
        crate::xlog_warn!("sparse windows: copies back not landed after 5 s (at {}, sent {issued})", fence.GetCompletedValue());
    }
    wait_stats::add(Stat::SparseWaits, 1);
    wait_stats::add_us(Stat::SparseWaitUs, t0);
}

// ---------------------------------------------------------------------------------------
// XCL_WC_NO_BACK
// ---------------------------------------------------------------------------------------

fn wc_min() -> Option<usize> {
    static M: OnceLock<Option<usize>> = OnceLock::new();
    *M.get_or_init(|| {
        if !std::env::var("XCL_WC_NO_BACK").is_ok_and(|v| v.trim() == "1") {
            return None;
        }
        let kb = std::env::var("XCL_WC_NO_BACK_MIN_KB").ok().and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(1024);
        crate::xlog_info!("sparse windows: XCL_WC_NO_BACK=1 – GPU writes of {kb} KiB or more into write-combined memory are not copied back");
        Some(kb << 10)
    })
}

/// write-combined sparse reservations: start → length
static WC: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());
static WC_SEEN: AtomicBool = AtomicBool::new(false);
static WC_LEFT: AtomicU64 = AtomicU64::new(0);

const PAGE_WRITECOMBINE: u32 = 0x400;

/// The title reserved or committed `[start, start + len)` in a sparse window with protection
/// `protect`. What was known of the range before is forgotten first (an address given back and
/// reserved again, or committed with another protection, is not taken for write-combined).
pub fn reserved(start: usize, len: usize, protect: u32) {
    if len == 0 {
        return;
    }
    let mut m = WC.lock().unwrap_or_else(|e| e.into_inner());
    if !m.is_empty() {
        let end = start + len;
        let over: Vec<usize> = m.range(..end).filter(|(&s, &l)| s + l > start).map(|(&s, _)| s).collect();
        for s in over {
            m.remove(&s);
        }
    }
    if protect & PAGE_WRITECOMBINE == 0 {
        return;
    }
    if !WC_SEEN.swap(true, Ordering::Relaxed) {
        crate::xlog_info!("sparse windows: write-combined reservation at {start:#x} ({} KiB, protection {protect:#x})", len >> 10);
    }
    m.insert(start, len);
}

/// The reservation at `start` went back.
pub fn released(start: usize) {
    WC.lock().unwrap_or_else(|e| e.into_inner()).remove(&start);
}

fn in_wc(map: &BTreeMap<usize, usize>, a: usize, n: usize) -> bool {
    map.range(..=a).next_back().is_some_and(|(&s, &l)| a + n <= s + l)
}

/// `XCL_WC_NO_BACK`: the GPU write `[a, a + n)` is not copied back.
pub(super) fn skip_back(a: usize, n: usize) -> bool {
    let Some(min) = wc_min() else { return false };
    if n < min || !WC_SEEN.load(Ordering::Relaxed) || !in_wc(&WC.lock().unwrap_or_else(|e| e.into_inner()), a, n) {
        return false;
    }
    let left = WC_LEFT.fetch_add(n as u64, Ordering::Relaxed) + n as u64;
    wait_stats::add(Stat::WcLeftKb, (n >> 10) as u64);
    static N: AtomicU64 = AtomicU64::new(0);
    if N.fetch_add(1, Ordering::Relaxed).is_power_of_two() {
        crate::xlog_info!("sparse windows: {} MB of GPU writes into write-combined memory not copied back so far", left >> 20);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_range_counts_as_write_combined_only_inside_one_reservation() {
        let mut m = BTreeMap::new();
        m.insert(0x1000, 0x1000);
        m.insert(0x4000, 0x2000);
        assert!(in_wc(&m, 0x1000, 0x1000));
        assert!(in_wc(&m, 0x4800, 0x800));
        assert!(!in_wc(&m, 0x1800, 0x1000), "runs past its reservation");
        assert!(!in_wc(&m, 0x2000, 0x100), "between reservations");
        assert!(!in_wc(&m, 0x800, 0x100), "before the first");
    }

    /// A reservation again at a write-combined address without the protection, or a commit in
    /// it with another protection, is not write-combined any more; one given back neither.
    #[test]
    fn the_registry_forgets_what_is_reserved_again_or_given_back() {
        const A: usize = 0x7FF0_0000_0000;
        let wc = |a, n| in_wc(&WC.lock().unwrap(), a, n);
        reserved(A, 0x10_0000, 0x404);
        assert!(wc(A + 0x1000, 0x1000));
        reserved(A + 0x2000, 0x1000, 0x4);
        assert!(!wc(A + 0x1000, 0x1000), "a commit with another protection ends it");
        reserved(A, 0x10_0000, 0x404);
        released(A);
        assert!(!wc(A, 0x1000));
        reserved(A, 0x10_0000, 0x404);
        reserved(A, 0x10_0000, 0x4);
        assert!(!wc(A, 0x1000), "reserved again without write-combining");
    }
}
