//! Residency priorities by use (`XCL_HOT_RESIDENCY=1`, off by default).
//!
//! Past its budget Windows pages a process's heaps out of video memory, and VidMm cannot tell
//! which of them a frame uses: XCL can only give it priorities. `XCL_RESIDENCY=max` (every pool
//! heap at the top) made frames slower, so the priorities follow what the title's lists use:
//!
//! * each PC command list keeps a log of what it touches, deduplicated by 64 KiB tile – vertex
//!   and index buffers, the source and destination of copies and memory operations, the arguments
//!   and count of ExecuteIndirect, render targets and depth, pipelines, the root views (64 KiB at
//!   their address, 4 MiB when the pipeline may store), 64-bit addresses among root constants and
//!   the first slots of descriptor tables;
//! * at ExecuteCommandLists (and KickoffX) the logs of the lists submitted go into the frame's
//!   set (a list submitted again counts again; its log goes at its Reset);
//! * at a present, at most every `UPDATE_EVERY`, the set is mapped onto heaps – a sparse window
//!   tile onto its pool heap, a target page or a texture onto its pool heap, an address or
//!   texture onto its arena – and the heaps whose class changed get it in one batched
//!   `SetResidencyPriority`:
//!   HIGH: touched in the last `HOT`, and target pool heaps holding pages of GPU-owned targets,
//!   arenas in use, two-pass buffers (at their creation);
//!   NORMAL: untouched for `HOT` to `COLD`; LOW: untouched for `COLD` or more;
//!   MINIMUM: empty pool heaps (spare pools, `XCL_VRAM_KEEP_SPARE_POOLS`; sparse pools whose every
//!   tile is free), retired heaps, arenas with no resource left.
//!   No heap is evicted: a sparse pool heap with a tile mapped into a window must stay readable,
//!   and MINIMUM already makes an empty one go first.
//!
//! Every `LOG_EVERY` a line says how much is at each priority and the hot set: the bytes of the
//! heaps touched in the last `HOT` against the budget less `XCL_VRAM_HEADROOM_MB` – whether what
//! the frames use fits at all.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::*;

#[cfg(test)]
mod tests;

/// a heap touched this recently is HIGH
pub(super) const HOT: Duration = Duration::from_secs(2);
/// one untouched this long is LOW (between `HOT` and this: NORMAL)
pub(super) const COLD: Duration = Duration::from_secs(4);
/// the frame's set is mapped onto heaps at a present this long after the last time at the earliest
const UPDATE_EVERY: Duration = Duration::from_millis(500);
const LOG_EVERY: Duration = Duration::from_secs(10);

const TILE: u64 = 0x1_0000;
/// key tags: a PC resource, a pipeline, a descriptor table's first slot; untagged: a 64 KiB tile
/// of a title address (`va / TILE`)
const RES: u64 = 1 << 63;
const PSO: u64 = 1 << 62;
const TABLE: u64 = 1 << 61;
const TAGS: u64 = RES | PSO | TABLE;
/// bytes a root view is taken for (it names an address, no size), and when the pipeline may
/// store through it (as `sparse::gpu_write` asks for an unsized write)
pub(super) const ROOT_VIEW_BYTES: u64 = 64 << 10;
pub(super) const ROOT_VIEW_WRITE_BYTES: u64 = 4 << 20;
/// tiles a range counts at most (256 MiB); a resource looked up whole, at most this many of its
const RANGE_TILES_MAX: u64 = 4096;
pub(super) const RES_TILES_MAX: usize = 4096;
/// descriptor table slots looked at (32 bytes each)
const TABLE_SLOTS: usize = 8;
/// keys a thread holds before they go to their list's log
const LOCAL_MAX: usize = 512;
/// keys a list's log and the frame's set hold at most, and keys mapped at an update
const LIST_KEYS_MAX: usize = 1 << 16;
const FRAME_KEYS_MAX: usize = 1 << 18;
const RESOLVE_MAX: usize = 1 << 18;

#[derive(Default)]
pub(super) struct Fx(u64);

impl Hasher for Fx {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.write_u64(b as u64);
        }
    }
    fn write_u64(&mut self, i: u64) {
        self.0 = (self.0.rotate_left(5) ^ i).wrapping_mul(0x51_7C_C1_B7_27_22_0A_95);
    }
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

pub(super) type FxSet<K> = HashSet<K, BuildHasherDefault<Fx>>;
type FxMap<K, V> = HashMap<K, V, BuildHasherDefault<Fx>>;

fn parse_on(v: Option<&str>) -> bool {
    matches!(v.map(str::trim), Some("1" | "on" | "true" | "yes"))
}

/// `XCL_HOT_RESIDENCY=1`
pub(super) fn on() -> bool {
    #[cfg(test)]
    if tests::ON_FOR_TEST.with(|c| c.get()) {
        return true;
    }
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = parse_on(std::env::var("XCL_HOT_RESIDENCY").ok().as_deref());
        if on {
            crate::xlog_info!("D3D12.X: XCL_HOT_RESIDENCY=1: residency priorities by use – HIGH touched in the last {} s, LOW untouched for {} s, MINIMUM empty and retired heaps", HOT.as_secs(), COLD.as_secs());
        }
        on
    })
}

// ---------------------------------------------------------------------------------------
// recording
// ---------------------------------------------------------------------------------------

/// The keys of the tiles `[va, va + bytes)` touches (`RANGE_TILES_MAX` at most; an empty range
/// counts its first tile), and whether the range was longer.
pub(super) fn va_keys(va: u64, bytes: u64) -> (std::ops::RangeInclusive<u64>, bool) {
    let first = va / TILE;
    let last = va.saturating_add(bytes.max(1) - 1) / TILE;
    let cut = last - first >= RANGE_TILES_MAX;
    (first..=last.min(first + RANGE_TILES_MAX - 1), cut)
}

/// What a key stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Key {
    /// a 64 KiB tile's title address
    Tile(usize),
    Res(usize),
    Pso(usize),
    /// a descriptor table's first slot
    Table(usize),
}

pub(super) fn decode(k: u64) -> Key {
    let v = (k & !TAGS) as usize;
    if k & RES != 0 {
        Key::Res(v)
    } else if k & PSO != 0 {
        Key::Pso(v)
    } else if k & TABLE != 0 {
        Key::Table(v)
    } else {
        Key::Tile(v * TILE as usize)
    }
}

/// a list's touches
#[derive(Default)]
struct ListLog {
    keys: FxSet<u64>,
}

static LOGS: super::sharded::Sharded<ListLog> = super::sharded::Sharded::new();

/// keys this thread recorded for `list` and has not put in its log yet
struct Local {
    list: usize,
    keys: Vec<u64>,
    /// spans of keys (first, last) pushed for `list` lately, the next one's place, and the
    /// `RESETS` they were pushed in: a span inside one is in the list's log already (a title
    /// binds the same buffers and views draw after draw)
    recent: [(u64, u64); RECENT],
    next: usize,
    resets: u64,
}

const RECENT: usize = 32;
const NO_SPAN: (u64, u64) = (u64::MAX, 0);

thread_local! {
    static LOCAL: RefCell<Local> = const { RefCell::new(Local { list: 0, keys: Vec::new(), recent: [NO_SPAN; RECENT], next: 0, resets: 0 }) };
}

/// lists' logs let go so far (`reset`): a thread's `Local::recent` from before one is not trusted
static RESETS: AtomicU64 = AtomicU64::new(0);

/// keys recorded, cut (past a range's, a log's or the frame's most), and the time put in the
/// frame's set at the submits (ns)
static RECORDED: AtomicU64 = AtomicU64::new(0);
static CUT: AtomicU64 = AtomicU64::new(0);
static MERGE_NS: AtomicU64 = AtomicU64::new(0);

fn flush(l: &mut Local) {
    if l.keys.is_empty() {
        return;
    }
    RECORDED.fetch_add(l.keys.len() as u64, Ordering::Relaxed);
    let mut cut = 0;
    let keys = &mut l.keys;
    LOGS.with(l.list, |log| {
        for k in keys.drain(..) {
            if log.keys.len() < LIST_KEYS_MAX || log.keys.contains(&k) {
                log.keys.insert(k);
            } else {
                cut += 1;
            }
        }
    });
    if cut != 0 {
        CUT.fetch_add(cut, Ordering::Relaxed);
    }
}

/// The keys `first..=last` for `list` (a span of tiles, or one tagged key).
fn push(list: usize, first: u64, last: u64) {
    if list == 0 {
        return;
    }
    let resets = RESETS.load(Ordering::Relaxed);
    let _ = LOCAL.try_with(|l| {
        let Ok(mut l) = l.try_borrow_mut() else { return };
        if l.list != list || l.resets != resets {
            if l.list != list {
                flush(&mut l);
                l.list = list;
            }
            l.recent = [NO_SPAN; RECENT];
            l.resets = resets;
        }
        if l.recent.iter().any(|&(a, b)| a <= first && last <= b) {
            return;
        }
        let n = l.next;
        l.recent[n] = (first, last);
        l.next = (n + 1) % RECENT;
        l.keys.extend(first..=last);
        if l.keys.len() >= LOCAL_MAX {
            flush(&mut l);
        }
    });
}

/// What this thread recorded goes to the lists' logs.
fn flush_local() {
    let _ = LOCAL.try_with(|l| {
        if let Ok(mut l) = l.try_borrow_mut() {
            flush(&mut l);
        }
    });
}

/// `list` reads or writes `[va, va + bytes)` (title addresses).
pub(super) fn touch_va(list: usize, va: u64, bytes: u64) {
    if va == 0 || !on() {
        return;
    }
    let (keys, cut) = va_keys(va, bytes);
    if cut {
        CUT.fetch_add(1, Ordering::Relaxed);
    }
    push(list, *keys.start(), *keys.end());
}

/// `list` uses the PC resource `res` (a texture, or a buffer whose bytes are not known here).
pub(super) fn touch_res(list: usize, res: usize) {
    if res != 0 && on() {
        push(list, RES | res as u64, RES | res as u64);
    }
}

/// `list` reads or writes `bytes` from `offset` of the PC resource `res`: at its title address
/// when it is a console buffer (`u64::MAX`: to its end), else the resource.
pub(super) fn touch_buffer(list: usize, res: usize, offset: u64, bytes: u64) {
    if res == 0 || !on() {
        return;
    }
    match super::real::console_buffer(res) {
        Some((p, n)) => {
            let offset = offset.min(n as u64);
            touch_va(list, p as u64 + offset, bytes.min(n as u64 - offset));
        }
        None => touch_res(list, res),
    }
}

pub(super) fn touch_pso(list: usize, pso: usize) {
    if pso != 0 && on() {
        push(list, PSO | pso as u64, PSO | pso as u64);
    }
}

/// `list` binds a descriptor table whose first slot is at `at` (unified memory).
pub(super) fn touch_table(list: usize, at: usize) {
    if at != 0 && on() {
        push(list, TABLE | at as u64, TABLE | at as u64);
    }
}

/// `list` closed: what this thread recorded goes to its log.
pub(super) fn closed(_list: usize) {
    if on() {
        flush_local();
    }
}

/// `list` records anew: its log goes.
pub(super) fn reset(list: usize) {
    if !on() || list == 0 {
        return;
    }
    let _ = LOCAL.try_with(|l| {
        if let Ok(mut l) = l.try_borrow_mut() {
            if l.list == list {
                l.keys.clear();
            }
        }
    });
    LOGS.remove(list);
    // (after the log went: a thread that pushes for the list from here on pushes again)
    RESETS.fetch_add(1, Ordering::Relaxed);
}

/// what the lists submitted since the last update touched
struct FrameKeys {
    keys: Option<FxSet<u64>>,
}

static FRAME: Mutex<FrameKeys> = Mutex::new(FrameKeys { keys: None });

/// The PC lists `lists` were submitted: their logs go in the frame's set (they stay with the
/// lists: one submitted again counts again).
pub(super) fn submitted(lists: &[*mut std::ffi::c_void]) {
    if !on() || lists.is_empty() {
        return;
    }
    let t = Instant::now();
    flush_local();
    let mut keys = Vec::new();
    for &l in lists {
        LOGS.peek(l as usize, |log| keys.extend(log.keys.iter().copied()));
    }
    if !keys.is_empty() {
        let mut cut = 0;
        let mut f = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        let set = f.keys.get_or_insert_with(FxSet::default);
        for k in keys {
            if set.len() < FRAME_KEYS_MAX || set.contains(&k) {
                set.insert(k);
            } else {
                cut += 1;
            }
        }
        drop(f);
        if cut != 0 {
            CUT.fetch_add(cut, Ordering::Relaxed);
        }
    }
    MERGE_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

fn take_frame() -> FxSet<u64> {
    FRAME.lock().unwrap_or_else(|e| e.into_inner()).keys.take().unwrap_or_default()
}

// ---------------------------------------------------------------------------------------
// classes
// ---------------------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(super) enum Class {
    Minimum = 0,
    Low = 1,
    Normal = 2,
    High = 3,
}

impl Class {
    pub(super) fn priority(self) -> D3D12_RESIDENCY_PRIORITY {
        match self {
            Class::Minimum => D3D12_RESIDENCY_PRIORITY_MINIMUM,
            Class::Low => D3D12_RESIDENCY_PRIORITY_LOW,
            Class::Normal => D3D12_RESIDENCY_PRIORITY_NORMAL,
            Class::High => D3D12_RESIDENCY_PRIORITY_HIGH,
        }
    }
}

/// A heap as its owner (`sparse`, `vram`) has it at an update.
#[derive(Clone, Copy, Debug)]
pub(super) struct HeapView {
    pub ptr: usize,
    pub bytes: u64,
    /// a key of the frames since the last update maps onto it
    pub touched: bool,
    /// HIGH whatever its use: pages of GPU-owned targets, an arena in use
    pub pinned: bool,
    /// nothing of it is in use (no tile mapped, no page held, no resource placed)
    pub empty: bool,
}

/// The class of a heap untouched for `idle`.
pub(super) fn classify(idle: Duration, pinned: bool, empty: bool) -> Class {
    if empty {
        Class::Minimum
    } else if pinned || idle < HOT {
        Class::High
    } else if idle < COLD {
        Class::Normal
    } else {
        Class::Low
    }
}

struct HeapState {
    last_touch: Instant,
    /// the class it was given (None: none yet – whatever it was made with)
    class: Option<Class>,
    empty: bool,
    seen: u64,
}

/// bytes and heaps of each class at an update, and of the hot set (touched in the last `HOT`)
#[derive(Clone, Copy, Default, Debug)]
pub(super) struct Tally {
    pub bytes: [u64; 4],
    pub heaps: [u32; 4],
    pub hot_bytes: u64,
    pub hot_heaps: u32,
}

/// Every heap seen, with when it was touched last and what it was given.
#[derive(Default)]
pub(super) struct Tracker {
    heaps: FxMap<usize, HeapState>,
    gen: u64,
    pub tally: Tally,
}

impl Tracker {
    /// An update begins: the tallies anew.
    pub(super) fn begin(&mut self) {
        self.gen += 1;
        self.tally = Tally::default();
    }

    /// A new heap at `ptr` (one let go may have been there): what was known of it goes.
    pub(super) fn forget(&mut self, ptr: usize) {
        self.heaps.remove(&ptr);
    }

    /// `v` at `now`: its class, when it is not the one it was given last.
    pub(super) fn see(&mut self, v: &HeapView, now: Instant) -> Option<Class> {
        let gen = self.gen;
        let e = self.heaps.entry(v.ptr).or_insert(HeapState { last_touch: now, class: None, empty: v.empty, seen: gen });
        // (a heap that comes into use again, as one made, is new work: touched now)
        if v.touched || (e.empty && !v.empty) {
            e.last_touch = now;
        }
        e.empty = v.empty;
        e.seen = gen;
        let idle = now.saturating_duration_since(e.last_touch);
        let c = classify(idle, v.pinned, v.empty);
        self.tally.bytes[c as usize] += v.bytes;
        self.tally.heaps[c as usize] += 1;
        if !v.empty && idle < HOT {
            self.tally.hot_bytes += v.bytes;
            self.tally.hot_heaps += 1;
        }
        (e.class != Some(c)).then(|| {
            e.class = Some(c);
            c
        })
    }

    /// The update is over: heaps not seen in it are gone.
    pub(super) fn end(&mut self) {
        let gen = self.gen;
        self.heaps.retain(|_, e| e.seen == gen);
    }
}

// ---------------------------------------------------------------------------------------
// updates
// ---------------------------------------------------------------------------------------

/// what the frames since the last update touched, by kind (tiles 64 KiB aligned, sorted)
#[derive(Default)]
pub(super) struct Touched {
    pub tiles: Vec<usize>,
    pub res: Vec<usize>,
    pub psos: usize,
}

/// `keys` sorted out (`RESOLVE_MAX` at most); a table's slots name resources by `resource_at`,
/// and a resource of a console buffer is its first `RES_TILES_MAX` tiles (`console_buffer`).
pub(super) fn sort_keys(keys: impl IntoIterator<Item = u64>, resource_at: impl Fn(usize) -> Option<usize>, console_buffer: impl Fn(usize) -> Option<(usize, usize)>) -> (Touched, u64) {
    let mut t = Touched::default();
    let mut tiles = FxSet::default();
    let mut res = FxSet::default();
    let mut cut = 0u64;
    for (n, k) in keys.into_iter().enumerate() {
        if n >= RESOLVE_MAX {
            cut += 1;
            continue;
        }
        match decode(k) {
            Key::Tile(a) => {
                tiles.insert(a);
            }
            Key::Res(r) => {
                res.insert(r);
            }
            Key::Pso(_) => t.psos += 1,
            Key::Table(at) => res.extend((0..TABLE_SLOTS).filter_map(|s| resource_at(at + 32 * s))),
        }
    }
    for &r in &res {
        if let Some((p, n)) = console_buffer(r) {
            let (keys, _) = va_keys(p as u64, n.min(RES_TILES_MAX * TILE as usize) as u64);
            tiles.extend(keys.map(|k| k as usize * TILE as usize));
        }
    }
    t.tiles = tiles.into_iter().collect();
    t.tiles.sort_unstable();
    t.res = res.into_iter().collect();
    t.res.sort_unstable();
    (t, cut)
}

/// whether a tile of `tiles` (sorted) lies in `[base, base + size)`
pub(super) fn any_in(tiles: &[usize], base: usize, size: usize) -> bool {
    let i = tiles.partition_point(|&a| a < base & !(TILE as usize - 1));
    tiles.get(i).is_some_and(|&a| a < base + size)
}

/// heaps made since the last update (`made`): what was known at their address goes
static MADE: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// A heap was made at `ptr` (sparse pool, target pool, arena).
pub(super) fn made(ptr: usize) {
    if on() {
        MADE.lock().unwrap_or_else(|e| e.into_inner()).push(ptr);
    }
}

struct Hot {
    tracker: Tracker,
    last_update: Option<Instant>,
    last_log: Instant,
    /// since the last log line: presents, updates, their time (ns, and the longest), keys mapped,
    /// heaps given a class and `SetResidencyPriority` calls, pipelines touched
    frames: u64,
    updates: u64,
    update_ns: u64,
    update_max_ns: u64,
    mapped: u64,
    changed: u64,
    calls: u64,
    psos: u64,
    /// SetResidencyPriority refused: nothing more is set
    broken: bool,
}

// (D3D12 objects are free-threaded; none is held here)
static HOT_STATE: Mutex<Option<Hot>> = Mutex::new(None);

/// A frame ends (a present): at most every `UPDATE_EVERY`, the touches are mapped onto heaps and
/// their classes set; every `LOG_EVERY` a line.
pub(super) fn frame() {
    if !on() {
        return;
    }
    flush_local();
    let now = Instant::now();
    // (one update at a time; a present meanwhile does not wait)
    let Ok(mut g) = HOT_STATE.try_lock() else { return };
    let h = g.get_or_insert_with(|| Hot {
        tracker: Tracker::default(),
        last_update: None,
        last_log: now,
        frames: 0,
        updates: 0,
        update_ns: 0,
        update_max_ns: 0,
        mapped: 0,
        changed: 0,
        calls: 0,
        psos: 0,
        broken: false,
    });
    h.frames += 1;
    if h.last_update.is_none_or(|t| now.duration_since(t) >= UPDATE_EVERY) {
        h.last_update = Some(now);
        unsafe { update(h, now) };
        let ns = now.elapsed().as_nanos() as u64;
        h.updates += 1;
        h.update_ns += ns;
        h.update_max_ns = h.update_max_ns.max(ns);
    }
    if now.duration_since(h.last_log) >= LOG_EVERY {
        log(h, now);
    }
}

unsafe fn update(h: &mut Hot, now: Instant) {
    let keys = take_frame();
    h.mapped += keys.len() as u64;
    for p in std::mem::take(&mut *MADE.lock().unwrap_or_else(|e| e.into_inner())) {
        h.tracker.forget(p);
    }
    let (t, cut) = sort_keys(keys, super::console_heap::resource_at, super::real::console_buffer);
    if cut != 0 {
        CUT.fetch_add(cut, Ordering::Relaxed);
    }
    h.psos += t.psos as u64;
    h.tracker.begin();
    let mut out: Vec<(ID3D12Heap, Class)> = Vec::new();
    let tracker = &mut h.tracker;
    let mut see = |v: &HeapView| tracker.see(v, now);
    super::sparse::hot_scan(&t.tiles, &t.res, &mut see, &mut out);
    super::vram::hot_scan(&t.tiles, &t.res, &mut see, &mut out);
    h.tracker.end();
    if out.is_empty() || h.broken {
        return;
    }
    h.changed += out.len() as u64;
    let objects: Vec<Option<ID3D12Pageable>> = out.iter().map(|(heap, _)| heap.cast::<ID3D12Pageable>().ok()).collect();
    let priorities: Vec<D3D12_RESIDENCY_PRIORITY> = out.iter().map(|&(_, c)| c.priority()).collect();
    let (objects, priorities): (Vec<_>, Vec<_>) = objects.into_iter().zip(priorities).filter(|(o, _)| o.is_some()).unzip();
    let Some(Some(first)) = objects.first() else { return };
    h.calls += 1;
    if let Err(e) = super::residency::set_batch(first, &objects, &priorities) {
        h.broken = true;
        crate::xlog_warn!("hot residency: SetResidencyPriority of {} heaps: {e} – priorities left as they are from here on", objects.len());
    }
}

/// Heaps given back (`vram` retiring empty pool heaps and idle arenas): MINIMUM until they go.
pub(super) unsafe fn retiring(heaps: &[ID3D12Heap]) {
    if !on() || heaps.is_empty() {
        return;
    }
    let objects: Vec<Option<ID3D12Pageable>> = heaps.iter().filter_map(|h| h.cast::<ID3D12Pageable>().ok()).map(Some).collect();
    let Some(Some(first)) = objects.first() else { return };
    let priorities = vec![D3D12_RESIDENCY_PRIORITY_MINIMUM; objects.len()];
    let _ = super::residency::set_batch(first, &objects, &priorities);
}

/// A buffer XCL keeps for work in use (two-pass records and arguments): HIGH from its creation.
pub(super) unsafe fn keep_high(res: &ID3D12Resource) {
    if !on() {
        return;
    }
    if let Ok(p) = res.cast::<ID3D12Pageable>() {
        let objects = [Some(p)];
        let _ = super::residency::set_batch(objects[0].as_ref().unwrap(), &objects, &[D3D12_RESIDENCY_PRIORITY_HIGH]);
    }
}

pub(super) fn line(t: &Tally, two_pass: u64, budget: u64, headroom: u64, local: u64) -> String {
    let mb = |b: u64| b >> 20;
    format!(
        "hot residency: HIGH {} MB ({} heaps), LOW {} MB ({} heaps), MIN {} MB; visible hot set {} MB vs target {} MB (budget {} − headroom {}); local {} MB; NORMAL {} MB ({} heaps), MIN {} heaps, two-pass buffers HIGH {} MB",
        mb(t.bytes[Class::High as usize]),
        t.heaps[Class::High as usize],
        mb(t.bytes[Class::Low as usize]),
        t.heaps[Class::Low as usize],
        mb(t.bytes[Class::Minimum as usize]),
        mb(t.hot_bytes),
        mb(budget.saturating_sub(headroom)),
        mb(budget),
        mb(headroom),
        mb(local),
        mb(t.bytes[Class::Normal as usize]),
        t.heaps[Class::Normal as usize],
        t.heaps[Class::Minimum as usize],
        mb(two_pass),
    )
}

fn log(h: &mut Hot, now: Instant) {
    let (local, _, budget) = super::real::video_memory().unwrap_or((0, 0, 0));
    let headroom = super::vram::vram_headroom();
    let text = line(&h.tracker.tally, super::bind::two_pass_alive_bytes(), budget, headroom, local);
    let frames = h.frames.max(1);
    let merge = MERGE_NS.swap(0, Ordering::Relaxed);
    let recorded = RECORDED.swap(0, Ordering::Relaxed);
    let cut = CUT.swap(0, Ordering::Relaxed);
    crate::xlog_info!(
        "{text}; cost {:.1} µs a frame on the presenting thread ({} updates, longest {} µs) + {:.1} µs a frame at the submits; {} keys recorded and {} mapped a frame, {} cut, {} pipelines touched an update; {} heaps changed class in {} SetResidencyPriority calls over {} frames",
        h.update_ns as f64 / 1e3 / frames as f64,
        h.updates,
        h.update_max_ns / 1000,
        merge as f64 / 1e3 / frames as f64,
        recorded / frames,
        h.mapped / frames,
        cut,
        h.psos / h.updates.max(1),
        h.changed,
        h.calls,
        h.frames
    );
    h.last_log = now;
    h.frames = 0;
    h.updates = 0;
    h.update_ns = 0;
    h.update_max_ns = 0;
    h.mapped = 0;
    h.changed = 0;
    h.calls = 0;
    h.psos = 0;
}
