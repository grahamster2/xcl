//! Where the process's local video memory went, for the log: the change in local video memory
//! in use (`QueryVideoMemoryInfo` CurrentUsage) across each creation of a kind, summed per kind.
//! What none of them made is the rest (the driver, the title's own heaps and resources, the
//! swap chain). Cheap and approximate: a creation on another thread at the same time counts
//! for both, and memory given back is only known for the heaps whose size XCL knows (`freed`).
//!
//! Pipelines are sampled: one is measured at a time (others built meanwhile are not), and the
//! sum is scaled to all pipelines made.

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::ID3D12Device;
use windows::Win32::Graphics::Dxgi::*;

/// What a measured creation made.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cat {
    /// CreateGraphicsPipelineState / CreateComputePipelineState / CreatePipelineState
    Pipelines = 0,
    /// CreateCommittedResource(1, 2): two-pass and compute-split buffers, fallback textures
    Committed = 1,
    /// the sparse windows' tile pool heaps
    Sparse = 2,
    /// the pool heaps of title texture pages (`vram::alloc_pool_heap`)
    TargetPools = 3,
    /// `vram` arenas
    Arenas = 4,
}

const CATS: usize = 5;
const NAMES: [&str; CATS] = ["pipelines", "committed resources", "sparse heaps", "target pool heaps", "arenas"];

/// change in local use across the measured creations (bytes, may be negative)
static DELTA: [AtomicI64; CATS] = [const { AtomicI64::new(0) }; CATS];
/// creations of each kind
static MADE: [AtomicU64; CATS] = [const { AtomicU64::new(0) }; CATS];
/// of them measured
static MEASURED: [AtomicU64; CATS] = [const { AtomicU64::new(0) }; CATS];
/// local video memory given back (bytes of the heaps let go)
static FREED: [AtomicU64; CATS] = [const { AtomicU64::new(0) }; CATS];
/// a pipeline is being measured
static PIPELINE_BUSY: AtomicBool = AtomicBool::new(false);
/// the device's adapter (an `IDXGIAdapter3` kept for the process), 0: not asked yet, 1: none
static ADAPTER: AtomicUsize = AtomicUsize::new(0);

/// Measured only with `XCL_ACTIVITY` (whose memory lines show it).
fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("XCL_ACTIVITY").is_some())
}

/// The process's local video memory in use now.
pub fn usage() -> Option<u64> {
    let mut raw = ADAPTER.load(Ordering::Acquire);
    if raw == 0 {
        let dev = super::real::device_raw();
        if dev.is_null() {
            return None;
        }
        let made = unsafe {
            let dev = ID3D12Device::from_raw_borrowed(&dev)?;
            CreateDXGIFactory1::<IDXGIFactory4>().and_then(|f| f.EnumAdapterByLuid::<IDXGIAdapter3>(dev.GetAdapterLuid())).ok()
        };
        let new = made.map_or(1, |a| a.into_raw() as usize);
        raw = match ADAPTER.compare_exchange(0, new, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => new,
            Err(other) => {
                if new > 1 {
                    // (another thread's is kept)
                    drop(unsafe { IDXGIAdapter3::from_raw(new as *mut _) });
                }
                other
            }
        };
    }
    if raw <= 1 {
        return None;
    }
    let p = raw as *mut std::ffi::c_void;
    let adapter = unsafe { IDXGIAdapter3::from_raw_borrowed(&p) }?;
    let mut i = DXGI_QUERY_VIDEO_MEMORY_INFO::default();
    unsafe { adapter.QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mut i) }.ok()?;
    Some(i.CurrentUsage)
}

/// A creation of `cat` begun: local use before it (None: not measured).
pub fn begin(cat: Cat) -> Option<u64> {
    MADE[cat as usize].fetch_add(1, Ordering::Relaxed);
    if !enabled() {
        return None;
    }
    if cat == Cat::Pipelines && PIPELINE_BUSY.swap(true, Ordering::AcqRel) {
        return None;
    }
    let before = usage();
    if before.is_none() && cat == Cat::Pipelines {
        PIPELINE_BUSY.store(false, Ordering::Release);
    }
    before
}

/// The creation `begin` gave `before` for is done.
pub fn end(cat: Cat, before: Option<u64>) {
    let Some(before) = before else { return };
    if let Some(after) = usage() {
        DELTA[cat as usize].fetch_add(after as i64 - before as i64, Ordering::Relaxed);
        MEASURED[cat as usize].fetch_add(1, Ordering::Relaxed);
    }
    if cat == Cat::Pipelines {
        PIPELINE_BUSY.store(false, Ordering::Release);
    }
}

/// `f`, a creation of `cat`, measured.
pub fn measure<R>(cat: Cat, f: impl FnOnce() -> R) -> R {
    let before = begin(cat);
    let r = f();
    end(cat, before);
    r
}

/// `bytes` of local video memory of `cat` given back.
pub fn freed(cat: Cat, bytes: u64) {
    FREED[cat as usize].fetch_add(bytes, Ordering::Relaxed);
}

/// What each kind holds by these sums: (change measured, scaled to all made for pipelines) less
/// what was given back.
fn net(delta: i64, made: u64, measured: u64, freed: u64, scaled: bool) -> i64 {
    let d = if scaled && measured > 0 { (delta as i128 * made as i128 / measured as i128) as i64 } else { delta };
    d - freed as i64
}

/// The log line: local video memory by what made it, and the rest. None: no use known.
pub fn line() -> Option<String> {
    let local = usage()?;
    let mb = |b: i64| b >> 20;
    let mut parts = Vec::with_capacity(CATS);
    let mut sum = 0i64;
    for k in 0..CATS {
        let (delta, made, measured, freed) =
            (DELTA[k].load(Ordering::Relaxed), MADE[k].load(Ordering::Relaxed), MEASURED[k].load(Ordering::Relaxed), FREED[k].load(Ordering::Relaxed));
        let scaled = k == Cat::Pipelines as usize;
        let n = net(delta, made, measured, freed, scaled);
        sum += n;
        let mut p = format!("{} {} MB ({made} made", NAMES[k], mb(n));
        if scaled {
            p += &format!(", {measured} measured");
        }
        if freed != 0 {
            p += &format!(", {} MB given back", freed >> 20);
        }
        p.push(')');
        parts.push(p);
    }
    Some(format!(
        "video memory by cause (change in local use at creation, since the start): {}; the rest {} MB of {} MB local",
        parts.join(", "),
        mb(local as i64 - sum),
        local >> 20
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipelines_scale_to_all_made_and_heaps_give_back() {
        assert_eq!(net(100 << 20, 8, 2, 0, true), 400 << 20, "2 of 8 measured");
        assert_eq!(net(100 << 20, 8, 0, 0, true), 100 << 20, "none measured: as it is");
        assert_eq!(net(640 << 20, 10, 10, 128 << 20, false), 512 << 20);
        assert_eq!(net(-(1 << 20), 1, 1, 0, false), -(1 << 20), "memory freed meanwhile");
    }
}
