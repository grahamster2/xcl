//! `XCL_TRACE_CS=<dump hash>[,<dump hash>…]` (hex, as the `[SHADER] cs …` lines and the GPU
//! times' "dump hash" name them): what a traced compute program's dispatches bind – each root
//! view, each buffer of its descriptor tables, the addresses its constants and root buffers
//! hold – with the memory each lies in: the unified-memory window (sparse: video memory;
//! coherent: system memory the GPU reaches over the bus; view: a GPU view of the title's CPU
//! memory; plain: process memory outside the windows; none: not readable) and, where XCL knows
//! it, what made that memory (AllocateUploadMemory, a console committed buffer or texture, a
//! heap's block, XCL's own pool, a title reservation). The first `FIRST` dispatches of each
//! program are traced, then each one with more groups than any before (`LARGER` of them).
//!
//! `XCL_UPLOAD_SPARSE=1` (`upload_sparse`): AllocateUploadMemory pieces in the sparse windows.
//!
//! Unset, both cost one load of a `OnceLock` where they are asked.

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

use super::console_list::{Layout, Param};

/// dispatches of a traced program logged from its first one
const FIRST: u32 = 3;
/// dispatches logged after those for having more groups than any before
const LARGER: u32 = 6;
/// descriptor table slots looked at
const TABLE_SLOTS: usize = 16;
/// dwords of a root view (a constant buffer, mostly) looked at for addresses it holds
const VIEW_DWORDS: usize = 64;
/// upload pieces remembered (more: the registry starts over)
const PIECES_MAX: usize = 1 << 17;

/// The hashes `XCL_TRACE_CS` lists (hex, `0x` optional; empty: off).
pub fn parse_hashes(v: &str) -> Vec<u64> {
    v.split(',').filter_map(|h| u64::from_str_radix(h.trim().trim_start_matches("0x"), 16).ok()).collect()
}

fn hashes() -> &'static [u64] {
    static H: OnceLock<Vec<u64>> = OnceLock::new();
    H.get_or_init(|| std::env::var("XCL_TRACE_CS").map(|v| parse_hashes(&v)).unwrap_or_default())
}

/// Whether `XCL_TRACE_CS` names any program.
pub fn on() -> bool {
    !hashes().is_empty()
}

/// `XCL_UPLOAD_SPARSE=1`: AllocateUploadMemory pieces small enough for an arena are cut from
/// blocks of the sparse windows (video memory; the CPU's writes reach it with the write-watch
/// upload before each submit) instead of the coherent window (system memory the GPU reaches
/// over the bus). Larger pieces go to the sparse windows already. Off by default.
pub fn upload_sparse() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_UPLOAD_SPARSE").is_ok_and(|v| v.trim() == "1");
        if on {
            crate::xlog_info!("D3D12.X: XCL_UPLOAD_SPARSE=1 – AllocateUploadMemory pieces in the sparse windows (video memory) rather than the coherent window");
        }
        on
    })
}

/// `XCL_UPLOAD_SPARSE`: a piece of a list's sparse upload arena. `carve` cuts it from the list's
/// arena (`Arena::take_with` under the list's lock) with the block source it is given: first
/// none (the current block's room), then one fresh sparse block, which is made here, outside
/// the list's lock (`sparse::alloc` takes the sparse windows' lock). A block `carve` did not
/// take goes back.
pub(super) unsafe fn sparse_upload_piece(align: usize, mut carve: impl FnMut(&mut dyn FnMut(usize, usize) -> Option<usize>) -> Option<*mut u8>) -> Option<*mut u8> {
    if let Some(p) = carve(&mut |_, _| None) {
        return Some(p);
    }
    let block_align = align.max(crate::gpu::uma::GRAIN).checked_next_power_of_two()?;
    let (b, _) = super::sparse::alloc(crate::gpu::uma::ARENA_BLOCK, block_align)?;
    let mut fresh = Some(b);
    let p = carve(&mut |_, _| fresh.take());
    if let Some(b) = fresh {
        super::sparse::free(b);
    }
    p
}

/// Which dispatches of a program get logged: the first `FIRST`, then each with more groups than
/// any before (`LARGER` of them).
#[derive(Default, Debug)]
pub struct Pick {
    seen: u32,
    most: u64,
    larger: u32,
}

impl Pick {
    /// Why this dispatch of `groups` is logged (None: it is not).
    pub fn take(&mut self, groups: [u32; 3]) -> Option<&'static str> {
        let n = groups.iter().map(|&g| g as u64).product::<u64>();
        self.seen = self.seen.saturating_add(1);
        let more = n > self.most;
        self.most = self.most.max(n);
        if self.seen <= FIRST {
            return Some("first");
        }
        if more && self.larger < LARGER {
            self.larger += 1;
            return Some("largest so far");
        }
        None
    }
}

/// What made a piece of upload memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Upload {
    /// a piece of the list's arena in the coherent window (shared with XCL's per-draw data)
    Arena,
    /// a piece of the list's upload arena in the sparse windows (`XCL_UPLOAD_SPARSE`)
    SparseArena,
    /// a block of its own in the sparse windows
    Sparse,
    /// a block of its own in the coherent window
    Coherent,
}

/// AllocateUploadMemory pieces: start → (bytes, what made it, number)
#[derive(Default)]
pub struct Pieces {
    m: BTreeMap<usize, (usize, Upload, u64)>,
    n: u64,
}

impl Pieces {
    /// A piece of `bytes` at `p`: pieces it overlaps (memory used again) are forgotten.
    pub fn add(&mut self, p: usize, bytes: usize, how: Upload) {
        self.forget(p, bytes.max(1));
        if self.m.len() >= PIECES_MAX {
            self.m.clear();
        }
        self.n += 1;
        self.m.insert(p, (bytes.max(1), how, self.n));
    }

    /// Pieces overlapping `[p, p + bytes)` are forgotten (their memory was freed).
    pub fn forget(&mut self, p: usize, bytes: usize) {
        let end = p.saturating_add(bytes);
        if let Some((&s, &(n, _, _))) = self.m.range(..p).next_back() {
            if s + n > p {
                self.m.remove(&s);
            }
        }
        let inside: Vec<usize> = self.m.range(p..end).map(|(&s, _)| s).collect();
        for s in inside {
            self.m.remove(&s);
        }
    }

    /// The piece holding `va`: (start, bytes, what made it, its number).
    pub fn at(&self, va: usize) -> Option<(usize, usize, Upload, u64)> {
        self.m.range(..=va).next_back().filter(|(&s, &(n, _, _))| va < s + n).map(|(&s, &(n, h, k))| (s, n, h, k))
    }
}

static PIECES: Mutex<Option<Pieces>> = Mutex::new(None);

/// AllocateUploadMemory handed out `bytes` at `p` (remembered while tracing).
pub fn note_upload(p: usize, bytes: usize, how: Upload) {
    if on() && p != 0 {
        PIECES.lock().unwrap().get_or_insert_with(Pieces::default).add(p, bytes, how);
    }
}

/// A block of unified memory was freed: upload pieces in it are forgotten (while tracing).
pub fn note_freed(p: usize, bytes: usize) {
    if on() {
        if let Some(g) = PIECES.lock().unwrap().as_mut() {
            g.forget(p, bytes);
        }
    }
}

/// 64-bit values at each dword of `words` (pairs `words[i]`, `words[i + 1]`) that `is_address`
/// takes: (dword offset, value), each value once.
pub fn addresses_in(words: &[u32], is_address: impl Fn(u64) -> bool) -> Vec<(usize, u64)> {
    let mut out: Vec<(usize, u64)> = Vec::new();
    for i in 0..words.len().saturating_sub(1) {
        let v = words[i] as u64 | (words[i + 1] as u64) << 32;
        if v != 0 && is_address(v) && !out.iter().any(|&(_, x)| x == v) {
            out.push((i, v));
        }
    }
    out
}

/// The bytes a buffer V# reaches (raw: num_records bytes; else num_records × stride).
pub fn buffer_bytes(b: &crate::gpu::desc::Buffer) -> u64 {
    if b.stride == 0 { b.num_records as u64 } else { b.num_records as u64 * b.stride as u64 }
}

/// The window `va` lies in, by kind: (`sparse` / `coherent` / `view` with the window's index),
/// `plain` (readable memory outside the windows) or `none`.
fn window_text(u: &crate::gpu::uma::Uma, va: usize) -> String {
    match u.window_of(va) {
        Some(k) if u.is_sparse_window(k) => format!("sparse window {k}"),
        Some(k) if u.is_committed_window(k) => format!("coherent window {k}"),
        Some(k) if u.is_view_window(k) => format!("view window {k}"),
        Some(k) => format!("window {k}"),
        None if crate::gpu::shader::readable(va, 4) => "plain".to_string(),
        None => "none".to_string(),
    }
}

/// What XCL knows made the memory at `va` (all it finds, outermost last).
fn origin_text(u: &crate::gpu::uma::Uma, va: usize) -> String {
    let mut what: Vec<String> = Vec::new();
    if let Some((s, n, how, k)) = PIECES.lock().unwrap().as_ref().and_then(|g| g.at(va)) {
        let how = match how {
            Upload::Arena => "a piece of the list's arena",
            Upload::SparseArena => "a piece of the list's sparse upload arena",
            Upload::Sparse => "a sparse block of its own",
            Upload::Coherent => "a coherent block of its own",
        };
        what.push(format!("upload memory #{k} ({n} bytes at {s:#x} +{:#x}, {how})", va - s));
    }
    if let Some((r, s, n, xcl)) = super::real::console_resource_at(va) {
        let d = unsafe { ID3D12Resource::from_raw_borrowed(&(r as *mut c_void)).map(|x| x.GetDesc()) };
        let kind = match d.map(|d| d.Dimension) {
            Some(D3D12_RESOURCE_DIMENSION_BUFFER) => "buffer",
            Some(_) => "texture",
            None => "resource",
        };
        let rb = if super::real::is_sparse_readback(r) { ", read back on Map" } else { "" };
        let made = if xcl { "committed: XCL's memory" } else { "placed at the title's address" };
        what.push(format!("console {kind} {r:#x} ({made}, {n:#x} bytes at {s:#x} +{:#x}{rb})", va - s));
    }
    if let Some((h, s, n, rb)) = super::real::heap_block_at(va) {
        what.push(format!("the block of heap {h:#x} ({n:#x} bytes at {s:#x} +{:#x}{})", va - s, if rb { ", a CPU-readback heap" } else { "" }));
    }
    if u.in_pool(va) {
        what.push("XCL's arena pool (per-draw data; small upload pieces)".to_string());
    } else if let Some((s, n)) = u.block_at(va) {
        what.push(format!("a unified-memory block ({:#x} bytes at {s:#x} +{:#x})", n, va - s));
    }
    if u.is_sparse(va) {
        match u.sparse_reservation(va) {
            Some((s, n)) => what.push(format!("a sparse reservation ({} MB at {s:#x} +{:#x})", n >> 20, va - s)),
            None => what.push("no sparse reservation (nothing there)".to_string()),
        }
    }
    if u.is_view(va) {
        match u.view_reservation(va) {
            Some((s, n)) => what.push(format!("a GPU view ({} MB at {s:#x}; CPU view {:#x})", n >> 20, super::sparse::cpu_of(va))),
            None => what.push("no GPU view reserved there".to_string()),
        }
    }
    if what.is_empty() {
        "nothing XCL knows".to_string()
    } else {
        what.join(" / ")
    }
}

/// `va`'s window and origin, as the trace lines give them.
fn where_text(u: &crate::gpu::uma::Uma, va: usize) -> String {
    format!("{}: {}", window_text(u, va), origin_text(u, va))
}

/// Up to `n` dwords the CPU can read at `va` (a GPU view's CPU view for a view-window address).
unsafe fn read_words(va: usize, n: usize) -> Vec<u32> {
    let at = super::sparse::cpu_of(va);
    if at == 0 || !crate::gpu::shader::readable(at, 4 * n) {
        return Vec::new();
    }
    (0..n).map(|k| ((at + 4 * k) as *const u32).read_unaligned()).collect()
}

/// A dispatch of `groups` with `pso`, on the list whose compute root signature is `layout` with
/// `args`: logged if `XCL_TRACE_CS` names its program and `Pick` takes it.
pub(super) unsafe fn dispatch(pso: usize, groups: [u32; 3], layout: Option<&Layout>, args: &[u8]) {
    let Some((hash, _)) = super::real::pso_shaders(pso) else { return };
    if !hashes().contains(&hash) {
        return;
    }
    let (n, why) = {
        static PICKS: Mutex<Option<HashMap<u64, Pick>>> = Mutex::new(None);
        let mut g = PICKS.lock().unwrap();
        let p = g.get_or_insert_with(HashMap::new).entry(hash).or_default();
        let Some(why) = p.take(groups) else { return };
        (p.seen, why)
    };
    let Some(u) = crate::gpu::uma::get() else { return };
    let [x, y, z] = groups;
    let Some(layout) = layout else {
        crate::xlog_info!("cs trace: cs {hash:016x} pipeline {pso:#x} dispatch #{n} ({why}) {x}x{y}x{z} groups: no root signature layout");
        return;
    };
    crate::xlog_info!(
        "cs trace: cs {hash:016x} pipeline {pso:#x} dispatch #{n} ({why}) {x}x{y}x{z} groups: {} root parameters, {} bytes of arguments{}",
        layout.params.len(),
        args.len(),
        if layout.heaps[0] { " (the shader indexes the view heap itself: its own heap indices are not followed)" } else { "" }
    );
    let word = |at: usize| args.get(at..at + 4).map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()));
    let is_address = |v: u64| u.window_of(v as usize).is_some();
    for (i, p) in layout.params.iter().enumerate() {
        match *p {
            Param::Constants { at, count } => {
                let w: Vec<u32> = (0..count as usize).map(|k| word(at as usize + 4 * k)).collect();
                crate::xlog_info!("cs trace: … p{i} constants {:x?}", &w[..w.len().min(32)]);
                for (k, v) in addresses_in(&w, is_address) {
                    crate::xlog_info!("cs trace: … p{i} constants [{k}..{}] hold address {v:#x} – {}", k + 1, where_text(u, v as usize));
                }
            }
            Param::Descriptor { at } => {
                let va = (word(at as usize) as u64 | (word(at as usize + 4) as u64) << 32) as usize;
                crate::xlog_info!("cs trace: … p{i} root view {va:#x} – {}", where_text(u, va));
                if va != 0 {
                    for (k, v) in addresses_in(&read_words(va, VIEW_DWORDS), is_address) {
                        crate::xlog_info!("cs trace: … p{i} root view {va:#x} holds address {v:#x} at +{:#x} – {}", 4 * k, where_text(u, v as usize));
                    }
                }
            }
            Param::Table { at } => {
                let t = word(at as usize);
                let mut empty = 0;
                for k in 0..TABLE_SLOTS {
                    let slot = t as usize + 32 * k;
                    let Some(w) = super::console_heap::words_at(slot as u32) else {
                        if k == 0 {
                            crate::xlog_info!("cs trace: … p{i} table {t:#x}: not in a console view heap");
                        }
                        break;
                    };
                    let res = super::console_heap::resource_at(u.base() + slot).map_or(String::new(), |r| format!(" (view of {r:#x})"));
                    if w[..4].iter().all(|&x| x == 0) {
                        empty += 1;
                        continue;
                    }
                    // (a T#'s type, word 3 [31:28], is 8 or more; a V#'s bits there are its
                    // out-of-bounds mode, [29:28])
                    if w[3] >> 28 >= 8 {
                        crate::xlog_info!("cs trace: … p{i} table {t:#x}[{k}] image{res}");
                        continue;
                    }
                    let b = crate::gpu::desc::buffer(&[w[0], w[1], w[2], w[3]]);
                    crate::xlog_info!(
                        "cs trace: … p{i} table {t:#x}[{k}] buffer {:#x} +{:#x} stride {}{res} – {}",
                        b.address,
                        buffer_bytes(&b),
                        b.stride,
                        where_text(u, b.address as usize)
                    );
                }
                if empty > 0 {
                    crate::xlog_info!("cs trace: … p{i} table {t:#x}: {empty} of the first {TABLE_SLOTS} slots empty");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_are_hex_with_or_without_0x() {
        assert_eq!(parse_hashes("24de2aef6c31d55e, 0x1,zz,"), vec![0x24de_2aef_6c31_d55e, 1]);
        assert!(parse_hashes("").is_empty());
    }

    /// The first three dispatches, then each with more groups than any before (six of them).
    #[test]
    fn first_dispatches_and_then_the_largest_are_picked() {
        let mut p = Pick::default();
        assert_eq!(p.take([100, 1, 1]), Some("first"));
        assert_eq!(p.take([10, 1, 1]), Some("first"));
        assert_eq!(p.take([10, 2, 1]), Some("first"));
        assert_eq!(p.take([100, 1, 1]), None, "as many as the largest: not larger");
        assert_eq!(p.take([17608, 1, 1]), Some("largest so far"));
        assert_eq!(p.take([5, 1, 1]), None);
        for k in 0..5 {
            assert_eq!(p.take([20000 + k, 1, 1]), Some("largest so far"));
        }
        assert_eq!(p.take([1 << 20, 1, 1]), None, "six larger ones at most");
    }

    /// Pieces are found by any address in them; memory handed out again forgets what overlapped it.
    #[test]
    fn upload_pieces_are_found_and_forgotten() {
        let mut g = Pieces::default();
        g.add(0x1000, 0x40, Upload::Arena);
        g.add(0x1040, 0x100, Upload::SparseArena);
        assert_eq!(g.at(0x103f).map(|x| (x.0, x.2)), Some((0x1000, Upload::Arena)));
        assert_eq!(g.at(0x1040).map(|x| (x.0, x.2)), Some((0x1040, Upload::SparseArena)));
        assert_eq!(g.at(0x1140), None);
        assert_eq!(g.at(0xfff), None);
        // a block freed and its memory a new piece: the old pieces in it are gone
        g.add(0x1020, 0x30, Upload::Coherent);
        assert_eq!(g.at(0x1000), None);
        assert_eq!(g.at(0x1100), None);
        assert_eq!(g.at(0x1049).map(|x| (x.0, x.1, x.2)), Some((0x1020, 0x30, Upload::Coherent)));
        g.forget(0x1000, 0x1_0000);
        assert_eq!(g.at(0x1020), None);
    }

    /// 64-bit addresses at any dword of a constant buffer, each once.
    #[test]
    fn addresses_held_in_words() {
        let w = [0x1c41_0000, 0x13, 0x5, 0x1c41_0000, 0x13, 0, 0x10_0000, 0x10];
        let a = addresses_in(&w, |v| (0x10_0000_0000..0x20_0000_0000).contains(&v));
        assert_eq!(a, vec![(0, 0x13_1c41_0000), (6, 0x10_0010_0000)]);
        assert!(addresses_in(&[1], |_| true).is_empty());
    }

    #[test]
    fn a_buffer_reaches_its_records() {
        let raw = crate::gpu::desc::buffer(&crate::gpu::desc::buffer_words(0x13_0000_0000, 0, 0x400, 4));
        assert_eq!(buffer_bytes(&raw), 0x400);
        let structured = crate::gpu::desc::buffer(&crate::gpu::desc::buffer_words(0x13_0000_0000, 64, 0x10, 4));
        assert_eq!(buffer_bytes(&structured), 0x400);
    }
}
