//! Residency priorities for XCL's own heaps (`XCL_RESIDENCY`, `XCL_RESIDENCY_LOW`).
//!
//! When a process asks for more video memory than its budget, Windows pages heaps out to
//! system memory, and the GPU then reaches them over PCIe. By default every heap has NORMAL
//! priority, so the sparse tile pools (title buffers in the windows, GPU-view mirrors) are as
//! likely to be paged as anything else.
//!
//! * `XCL_RESIDENCY=max` (or `high`): the tile pools behind the sparse windows and the GPU-view
//!   mirrors, and the video-memory copies of the shader windows, get MAXIMUM (HIGH) priority
//!   right after they are made. Off by default.
//! * `XCL_RESIDENCY_LOW=1`: memory that is fine to page gets MINIMUM priority so it goes first –
//!   the pool heaps of title texture pages in video memory (`vram::alloc_pool_heap`), and a
//!   title's buffers-only heap once its buffers live in unified memory (its PC heap is then
//!   never used).
//! * By default (`XCL_TARGET_RESIDENCY=0` turns it off): GPU-owned targets, the reserve pool
//!   heaps of their pages (`XCL_TARGET_VRAM_MB`) and the video-memory arenas, get HIGH priority,
//!   and the sparse tile pools LOW (unless `XCL_RESIDENCY` asks for more): over the budget the
//!   OS pages sparse tiles out before the render targets, depth and UAVs the GPU writes every
//!   frame. Not everything at MAXIMUM: `XCL_RESIDENCY=max` made frames slower.

use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::OnceLock;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

/// The priority `XCL_RESIDENCY` asks for (None: off, or a value it does not know).
fn parse_hot(v: Option<&str>) -> Option<D3D12_RESIDENCY_PRIORITY> {
    match v.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
        Some("max" | "maximum") => Some(D3D12_RESIDENCY_PRIORITY_MAXIMUM),
        Some("high") => Some(D3D12_RESIDENCY_PRIORITY_HIGH),
        _ => None,
    }
}

fn parse_cold(v: Option<&str>) -> bool {
    matches!(v.map(str::trim), Some("1" | "on" | "true" | "yes"))
}

fn hot_priority() -> Option<D3D12_RESIDENCY_PRIORITY> {
    static P: OnceLock<Option<D3D12_RESIDENCY_PRIORITY>> = OnceLock::new();
    *P.get_or_init(|| {
        let v = std::env::var("XCL_RESIDENCY").ok();
        let p = parse_hot(v.as_deref());
        if p.is_none() {
            if let Some(v) = v.filter(|v| !v.trim().is_empty() && v.trim() != "0" && v.trim() != "off") {
                crate::xlog_warn!("XCL_RESIDENCY={v}: not max or high – left off");
            }
        }
        p
    })
}

fn parse_targeted(v: Option<&str>) -> bool {
    !matches!(v.map(str::trim), Some("0" | "off" | "false" | "no"))
}

fn targeted_on() -> bool {
    static P: OnceLock<bool> = OnceLock::new();
    *P.get_or_init(|| parse_targeted(std::env::var("XCL_TARGET_RESIDENCY").ok().as_deref()))
}

fn cold_on() -> bool {
    static P: OnceLock<bool> = OnceLock::new();
    *P.get_or_init(|| parse_cold(std::env::var("XCL_RESIDENCY_LOW").ok().as_deref()))
}

struct Tally {
    name: &'static str,
    /// Set once SetResidencyPriority is found missing or refused: no further tries for this
    /// kind (a title's heap that will not answer does not turn `XCL_RESIDENCY` off).
    broken: AtomicBool,
    heaps: AtomicU32,
    bytes: AtomicU64,
}

static HOT: Tally = Tally { name: "XCL_RESIDENCY", broken: AtomicBool::new(false), heaps: AtomicU32::new(0), bytes: AtomicU64::new(0) };
static TARGET: Tally = Tally { name: "XCL_TARGET_RESIDENCY (GPU-owned targets)", broken: AtomicBool::new(false), heaps: AtomicU32::new(0), bytes: AtomicU64::new(0) };
static SPARSE: Tally = Tally { name: "XCL_TARGET_RESIDENCY (sparse tile pools)", broken: AtomicBool::new(false), heaps: AtomicU32::new(0), bytes: AtomicU64::new(0) };
static COLD: Tally = Tally { name: "XCL_RESIDENCY_LOW", broken: AtomicBool::new(false), heaps: AtomicU32::new(0), bytes: AtomicU64::new(0) };

fn priority_name(p: D3D12_RESIDENCY_PRIORITY) -> &'static str {
    match p {
        D3D12_RESIDENCY_PRIORITY_MAXIMUM => "MAXIMUM",
        D3D12_RESIDENCY_PRIORITY_HIGH => "HIGH",
        D3D12_RESIDENCY_PRIORITY_NORMAL => "NORMAL",
        D3D12_RESIDENCY_PRIORITY_LOW => "LOW",
        D3D12_RESIDENCY_PRIORITY_MINIMUM => "MINIMUM",
        _ => "?",
    }
}

/// `heap` (`bytes` of it, `what` it holds) gets priority `p` from the device that made it.
unsafe fn set(heap: &ID3D12Heap, bytes: u64, p: D3D12_RESIDENCY_PRIORITY, what: &str, tally: &Tally) {
    if tally.broken.load(Ordering::Relaxed) {
        return;
    }
    let mut dev: Option<ID3D12Device1> = None;
    let got = heap.GetDevice(&mut dev);
    let dev = match (got, dev) {
        (Ok(()), Some(d)) => d,
        (r, _) => {
            if !tally.broken.swap(true, Ordering::Relaxed) {
                let why = r.err().map(|e| e.to_string()).unwrap_or_else(|| "none returned".into());
                crate::xlog_warn!("{}: no ID3D12Device1 for the heaps ({why}) – residency priorities left NORMAL", tally.name);
            }
            return;
        }
    };
    let Ok(pageable) = heap.cast::<ID3D12Pageable>() else {
        return;
    };
    let objects = [Some(pageable)];
    let priorities = [p];
    if let Err(e) = dev.SetResidencyPriority(1, objects.as_ptr(), priorities.as_ptr()) {
        if !tally.broken.swap(true, Ordering::Relaxed) {
            crate::xlog_warn!("{}: SetResidencyPriority ({what}): {e} – residency priorities left NORMAL from here on", tally.name);
        }
        return;
    }
    let n = tally.heaps.fetch_add(1, Ordering::Relaxed) + 1;
    let total = tally.bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
    // the first heap, then at 2, 4, 8 … heaps: a running total, not a line per heap
    if n.is_power_of_two() {
        crate::xlog_info!("{}: {n} heaps, {} MB in all, at residency priority {} (latest: {what}, {} MB)", tally.name, total >> 20, priority_name(p), bytes >> 20);
    }
}

/// `XCL_RESIDENCY`: a heap that must stay in video memory (a sparse tile pool, a shader
/// window's copy in video memory) right after it is made. Nothing when off.
pub unsafe fn keep(heap: &ID3D12Heap, bytes: u64, what: &str) {
    if let Some(p) = hot_priority() {
        set(heap, bytes, p, what, &HOT);
    }
}

/// A heap of GPU-owned targets (a reserve pool heap of their pages, an arena) gets HIGH
/// priority right after it is made. Nothing with `XCL_TARGET_RESIDENCY=0`.
pub unsafe fn target(heap: &ID3D12Heap, bytes: u64, what: &str) {
    if targeted_on() {
        set(heap, bytes, D3D12_RESIDENCY_PRIORITY_HIGH, what, &TARGET);
    }
}

/// A sparse tile pool right after it is made: `XCL_RESIDENCY`'s priority when set (`keep`),
/// else LOW, so it pages out before GPU-owned targets (`target`). Nothing with
/// `XCL_TARGET_RESIDENCY=0` and no `XCL_RESIDENCY`.
pub unsafe fn sparse_pool(heap: &ID3D12Heap, bytes: u64, what: &str) {
    if hot_priority().is_some() {
        keep(heap, bytes, what);
    } else if targeted_on() {
        set(heap, bytes, D3D12_RESIDENCY_PRIORITY_LOW, what, &SPARSE);
    }
}

/// `XCL_RESIDENCY_LOW`: a heap that is fine to page (texture pages, a title heap whose buffers
/// live elsewhere) goes out first. Nothing when off.
pub unsafe fn spare(heap: &ID3D12Heap, bytes: u64, what: &str) {
    if cold_on() {
        set(heap, bytes, D3D12_RESIDENCY_PRIORITY_MINIMUM, what, &COLD);
    }
}

/// `spare` for a raw heap pointer (a title's heap); nothing if it is not an ID3D12Heap.
pub unsafe fn spare_raw(heap: usize, bytes: u64, what: &str) {
    if !cold_on() || heap == 0 {
        return;
    }
    let raw = heap as *mut std::ffi::c_void;
    let Some(unk) = windows::core::IUnknown::from_raw_borrowed(&raw) else { return };
    if let Ok(h) = unk.cast::<ID3D12Heap>() {
        spare(&h, bytes, what);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values() {
        assert_eq!(parse_hot(Some("max")), Some(D3D12_RESIDENCY_PRIORITY_MAXIMUM));
        assert_eq!(parse_hot(Some(" MAXIMUM ")), Some(D3D12_RESIDENCY_PRIORITY_MAXIMUM));
        assert_eq!(parse_hot(Some("high")), Some(D3D12_RESIDENCY_PRIORITY_HIGH));
        assert_eq!(parse_hot(Some("1")), None);
        assert_eq!(parse_hot(Some("off")), None);
        assert_eq!(parse_hot(None), None);
        assert!(parse_cold(Some("1")));
        assert!(!parse_cold(Some("0")));
        assert!(!parse_cold(None));
        assert!(parse_targeted(None), "on by default");
        assert!(parse_targeted(Some("1")));
        assert!(!parse_targeted(Some(" 0 ")));
        assert!(!parse_targeted(Some("off")));
    }

    #[test]
    fn a_heap_gets_its_priority() {
        let Ok(dev) = crate::gpu::agility::create_device(None, windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0) else {
            return crate::gpu::recomp::runner::skip_test("no GPU");
        };
        unsafe {
            let desc = D3D12_HEAP_DESC { SizeInBytes: 1 << 16, Properties: D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, ..Default::default() }, Alignment: 0, Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS };
            let mut heap: Option<ID3D12Heap> = None;
            dev.CreateHeap(&desc, &mut heap).unwrap();
            let heap = heap.unwrap();
            let tally = Tally { name: "test", broken: AtomicBool::new(false), heaps: AtomicU32::new(0), bytes: AtomicU64::new(0) };
            set(&heap, 1 << 16, D3D12_RESIDENCY_PRIORITY_MAXIMUM, "test heap", &tally);
            assert_eq!(tally.heaps.load(Ordering::Relaxed), 1);
            assert_eq!(tally.bytes.load(Ordering::Relaxed), 1 << 16);
        }
    }
}
