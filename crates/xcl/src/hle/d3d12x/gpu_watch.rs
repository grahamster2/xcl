//! `XCL_GPU_SIDE_WAITS=1`: a `Wait32BitValueX` / `Wait64BitValueX` whose memory does not hold
//! at the submit and that the per-address fences cannot bridge (`list_x::try_gpu_wait`: a
//! 32-bit wait, another comparison or mask, a counter no recorded `Write64` produces – the CPU
//! writes it, or a shader) waits on the GPU instead of on the submitting thread.
//!
//! The queue gets `queue.Wait(fence, v)` on a fence of a small pool, in the place where the
//! submitting thread would have polled (before the list that waits); a thread of XCL's polls the
//! memory as that thread did and signals `v` on the CPU once it holds, or once the poll's time is
//! up (`list_x::wait_timeout`: the CPU poll goes on after it too). The waiting queue so starts its
//! list at the same point of the memory as before; what changes is that the submitting thread
//! goes on at once, with the title's other queues' submits (a title submits all its queues from
//! one thread: one CPU poll held every queue behind it).
//!
//! Off by default.

use std::ffi::c_void;
use std::sync::{Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use super::wait_stats::{self, Stat};

pub fn enabled() -> bool {
    #[cfg(test)]
    if FORCE.with(|f| f.get()) {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_GPU_SIDE_WAITS").is_ok_and(|v| v.trim() == "1");
        if on {
            crate::xlog_info!("D3D12.X: XCL_GPU_SIDE_WAITS=1 – Wait32/64BitValueX the per-address fences cannot bridge wait on the GPU, watched on the CPU");
        }
        on
    })
}

#[cfg(test)]
thread_local! {
    static FORCE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Tests: `enabled` on this thread while `f` runs.
#[cfg(test)]
pub fn with_enabled<R>(f: impl FnOnce() -> R) -> R {
    let was = FORCE.with(|c| c.replace(true));
    let r = f();
    FORCE.with(|c| c.set(was));
    r
}

/// A fence of the pool: the last value signalled on the CPU (a queue waits for one more).
struct Slot {
    fence: ID3D12Fence,
    device: usize,
    value: u64,
}

struct Watch {
    slot: Slot,
    ready: Box<dyn Fn() -> bool + Send>,
    /// what is waited for, for the log
    what: String,
    armed: Instant,
    timeout: Duration,
}

// SAFETY: `ID3D12Fence` is free-threaded.
unsafe impl Send for Slot {}

struct State {
    free: Vec<Slot>,
    watches: Vec<Watch>,
    thread: bool,
}

static STATE: Mutex<State> = Mutex::new(State { free: Vec::new(), watches: Vec::new(), thread: false });
static WAKE: Condvar = Condvar::new();

/// (a panic elsewhere must not stop the title's submits here)
fn state() -> std::sync::MutexGuard<'static, State> {
    STATE.lock().unwrap_or_else(|e| e.into_inner())
}

/// Make `queue` wait on the GPU until `ready` holds (polled on XCL's thread) or `timeout` is up.
/// false: not done (no device or fence, or the queue refused the wait) – the caller polls on the CPU.
pub unsafe fn arm(queue: *mut c_void, ready: Box<dyn Fn() -> bool + Send>, what: String, timeout: Duration) -> bool {
    let Some(q) = ID3D12CommandQueue::from_raw_borrowed(&queue) else { return false };
    let mut dev: Option<ID3D12Device> = None;
    if q.GetDevice(&mut dev).is_err() {
        return false;
    }
    let Some(dev) = dev else { return false };
    let dev_raw = dev.as_raw() as usize;
    let pooled = {
        let mut g = state();
        g.free.iter().position(|s| s.device == dev_raw).map(|i| g.free.swap_remove(i))
    };
    let mut slot = match pooled {
        Some(s) => s,
        None => {
            let Ok(fence) = dev.CreateFence::<ID3D12Fence>(0, D3D12_FENCE_FLAG_NONE) else { return false };
            Slot { fence, device: dev_raw, value: 0 }
        }
    };
    let target = slot.value + 1;
    if q.Wait(&slot.fence, target).is_err() {
        state().free.push(slot);
        return false;
    }
    slot.value = target;
    super::list_x::note_internal_wait(queue as usize, "Wait32/64BitValueX watched on the CPU", &slot.fence, target);
    wait_stats::add(Stat::Watched, 1);
    let mut g = state();
    g.watches.push(Watch { slot, ready, what, armed: Instant::now(), timeout });
    if !g.thread {
        g.thread = std::thread::Builder::new().name("xcl-gpu-watch".into()).spawn(watch_thread).is_ok();
        if !g.thread {
            // no thread to signal it: release it now (the wait holds what the CPU poll held at its timeout)
            let w = g.watches.pop().unwrap();
            drop(g);
            let _ = w.slot.fence.Signal(w.slot.value);
            crate::xlog_warn!("D3D12.X: no thread to watch {} – released at once", w.what);
            return true;
        }
    }
    WAKE.notify_one();
    true
}

/// The watches still waiting (tests).
#[cfg(test)]
pub fn pending() -> usize {
    state().watches.len()
}

fn watch_thread() {
    loop {
        // the conditions are read outside the lock (`shader::readable` can take long): `arm` on
        // the title's submitting thread does not wait for them
        let watches: Vec<Watch> = {
            let mut g = state();
            while g.watches.is_empty() {
                g = WAKE.wait(g).unwrap_or_else(|e| e.into_inner());
            }
            std::mem::take(&mut g.watches)
        };
        let mut released: Vec<(Watch, bool)> = Vec::new();
        let mut kept: Vec<Watch> = Vec::new();
        let mut youngest: Option<Duration> = None;
        for w in watches {
            // a condition that panics counts as holding (the CPU poll's unreadable memory does)
            let held = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| (w.ready)())).unwrap_or(true);
            let age = w.armed.elapsed();
            if held || age >= w.timeout {
                released.push((w, held));
            } else {
                youngest = Some(youngest.map_or(age, |y| y.min(age)));
                kept.push(w);
            }
        }
        state().watches.extend(kept);
        for (w, held) in released {
            unsafe {
                let _ = w.slot.fence.Signal(w.slot.value);
            }
            wait_stats::add_us(Stat::WatchedUs, w.armed);
            if !held {
                wait_stats::add(Stat::WatchedTimeouts, 1);
                static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 40 {
                    crate::xlog_warn!("D3D12.X: {} did not hold within {} ms – the GPU wait released on the CPU", w.what, w.timeout.as_millis());
                }
            }
            state().free.push(w.slot);
        }
        // the CPU poll's pace (`PendingWait::wait_until_satisfied`): 100 µs steps while a wait
        // is in its first 10 ms, then 1 ms
        match youngest {
            None => {}
            Some(age) if age < Duration::from_millis(10) => std::thread::sleep(Duration::from_micros(100)),
            Some(_) => std::thread::sleep(Duration::from_millis(1)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    unsafe fn device() -> Option<ID3D12Device> {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Dxgi::*;
        let mut dev: Option<ID3D12Device> = None;
        if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().ok()?;
            let warp: IDXGIAdapter = factory.EnumWarpAdapter().ok()?;
            D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).ok()?;
        }
        dev
    }

    /// The queue stands at the watched wait until the condition holds, and the arming thread
    /// does not wait for it; the second round takes a fence of the pool if one is free.
    #[test]
    fn the_queue_waits_until_the_condition_holds_and_the_caller_goes_on() {
        unsafe {
            let Some(dev) = device() else { return crate::gpu::recomp::runner::skip_test("no GPU") };
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_COMPUTE, ..Default::default() }).unwrap();
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            for round in 1..=2u64 {
                let flag = Arc::new(AtomicBool::new(false));
                let f = flag.clone();
                let t0 = Instant::now();
                assert!(arm(q.as_raw(), Box::new(move || f.load(Ordering::Acquire)), format!("test round {round}"), Duration::from_secs(10)));
                assert!(t0.elapsed() < Duration::from_millis(200), "arming does not wait");
                q.Signal(&done, round).unwrap();
                std::thread::sleep(Duration::from_millis(150));
                assert_eq!(done.GetCompletedValue(), round - 1, "the queue is held at the wait while the condition does not hold");
                flag.store(true, Ordering::Release);
                crate::gpu::fence::wait(&done, round, &dev, None, 3000).expect("the queue goes on once the condition holds");
            }
        }
    }

    /// A condition that never holds lets the queue go on after the timeout, as the CPU poll does.
    #[test]
    fn a_condition_that_never_holds_is_released_after_the_timeout() {
        unsafe {
            let Some(dev) = device() else { return crate::gpu::recomp::runner::skip_test("no GPU") };
            let q: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let done: ID3D12Fence = dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).unwrap();
            let before = wait_stats::snapshot()[Stat::WatchedTimeouts as usize];
            assert!(arm(q.as_raw(), Box::new(|| false), "test never".into(), Duration::from_millis(100)));
            q.Signal(&done, 1).unwrap();
            crate::gpu::fence::wait(&done, 1, &dev, None, 3000).expect("released after the timeout");
            assert!(wait_stats::snapshot()[Stat::WatchedTimeouts as usize] > before);
        }
    }
}
