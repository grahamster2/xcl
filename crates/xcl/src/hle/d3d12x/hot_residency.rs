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
//! * at a present, at most every `UPDATE_EVERY`, a worker thread is woken (the present only counts
//!   the frame and signals): it takes the set and maps it onto heaps – a sparse window tile onto
//!   its pool heap, a target page or a texture onto its pool heap, an address or texture onto its
//!   arena – and the heaps whose class changed get it in one batched `SetResidencyPriority`. The
//!   tiles are a bitmap (`TileSet`), and a heap owner's map is looked up tile by tile or walked
//!   whole, whichever is shorter (`for_each_touched`): its lock is held for a walk of it at most.
//!   The classes:
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
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, Once, OnceLock};
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
/// keys a list's log and the frame's set hold at most
const LIST_KEYS_MAX: usize = 1 << 16;
const FRAME_KEYS_MAX: usize = 1 << 18;

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
        // (the product's low bits, where the table's index is taken, are poor: without the
        // rotation 64 KiB aligned addresses all fell in a few buckets)
        self.0.rotate_left(26)
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

/// 64 KiB tiles of title addresses, as a bitmap of each GiB with one in it.
#[derive(Default)]
pub(super) struct TileSet {
    chunks: FxMap<u64, Box<[u64; CHUNK_WORDS]>>,
    len: usize,
}

/// tiles of a chunk (1 GiB) as a power of two, and its words
const CHUNK_SHIFT: u32 = 14;
const CHUNK_MASK: u64 = (1 << CHUNK_SHIFT) - 1;
const CHUNK_WORDS: usize = (1 << CHUNK_SHIFT) / 64;

/// the bits `from..=to` of a word
fn bits(from: usize, to: usize) -> u64 {
    (u64::MAX >> (63 - to)) & (u64::MAX << from)
}

impl TileSet {
    pub(super) fn len(&self) -> usize {
        self.len
    }

    fn chunk(&mut self, c: u64) -> &mut [u64; CHUNK_WORDS] {
        self.chunks.entry(c).or_insert_with(|| Box::new([0; CHUNK_WORDS]))
    }

    /// The tile `i` (`va / TILE`); whether it is new.
    pub(super) fn insert(&mut self, i: u64) -> bool {
        let w = &mut self.chunk(i >> CHUNK_SHIFT)[(i & CHUNK_MASK) as usize / 64];
        let bit = 1u64 << (i % 64);
        let new = *w & bit == 0;
        *w |= bit;
        self.len += new as usize;
        new
    }

    /// The tiles `first..=last`.
    pub(super) fn insert_range(&mut self, first: u64, last: u64) {
        let mut i = first;
        while i <= last {
            let c = i >> CHUNK_SHIFT;
            let end = last.min(i | CHUNK_MASK);
            let (lo, hi) = ((i & CHUNK_MASK) as usize, (end & CHUNK_MASK) as usize);
            let mut added = 0;
            let words = self.chunk(c);
            for w in lo / 64..=hi / 64 {
                let mask = bits(if w == lo / 64 { lo % 64 } else { 0 }, if w == hi / 64 { hi % 64 } else { 63 });
                added += (mask & !words[w]).count_ones() as usize;
                words[w] |= mask;
            }
            self.len += added;
            if end == u64::MAX {
                break;
            }
            i = end + 1;
        }
    }

    pub(super) fn contains_index(&self, i: u64) -> bool {
        self.chunks.get(&(i >> CHUNK_SHIFT)).is_some_and(|w| w[(i & CHUNK_MASK) as usize / 64] >> (i % 64) & 1 != 0)
    }

    /// whether the tile at `a` is in (a 64 KiB aligned address; any other is no tile's)
    pub(super) fn contains(&self, a: usize) -> bool {
        a % TILE as usize == 0 && self.contains_index((a / TILE as usize) as u64)
    }

    /// the tiles' addresses
    pub(super) fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.chunks.iter().flat_map(|(&c, words)| {
            words.iter().enumerate().flat_map(move |(w, &b)| {
                let base = (c << CHUNK_SHIFT) + w as u64 * 64;
                BitIter(b).map(move |k| ((base + k as u64) * TILE) as usize)
            })
        })
    }

    /// whether a tile lies in `[base, base + size)` (from the middle of a tile on: that tile too)
    pub(super) fn any_in(&self, base: usize, size: usize) -> bool {
        // the tiles from the one `base` is in to the last that starts before the end
        let first = (base / TILE as usize) as u64;
        let end = (base.saturating_add(size) as u64).div_ceil(TILE);
        if first >= end {
            return false;
        }
        // the part of chunk `c` in the range
        let hit = |c: u64, words: &[u64; CHUNK_WORDS]| {
            let (from, to) = (first.max(c << CHUNK_SHIFT), (end - 1).min(c << CHUNK_SHIFT | CHUNK_MASK));
            let (lo, hi) = ((from & CHUNK_MASK) as usize, (to & CHUNK_MASK) as usize);
            (lo / 64..=hi / 64).any(|w| words[w] & bits(if w == lo / 64 { lo % 64 } else { 0 }, if w == hi / 64 { hi % 64 } else { 63 }) != 0)
        };
        let (c0, c1) = (first >> CHUNK_SHIFT, (end - 1) >> CHUNK_SHIFT);
        // (the chunks of the range looked up, or the chunks there are walked: never the
        // address space's 2^34 empty ones)
        if c1 - c0 < self.chunks.len() as u64 {
            (c0..=c1).any(|c| self.chunks.get(&c).is_some_and(|w| hit(c, w)))
        } else {
            self.chunks.iter().any(|(&c, w)| (c0..=c1).contains(&c) && hit(c, w))
        }
    }

    /// (tests) the tiles at `addrs`
    #[cfg(test)]
    pub(super) fn of(addrs: &[usize]) -> TileSet {
        let mut t = TileSet::default();
        for &a in addrs {
            t.insert((a / TILE as usize) as u64);
        }
        t
    }
}

struct BitIter(u64);

impl Iterator for BitIter {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        (self.0 != 0).then(|| {
            let b = self.0.trailing_zeros();
            self.0 &= self.0 - 1;
            b
        })
    }
}

/// `f` with the value of each tile of `tiles` in `map` (keyed by 64 KiB aligned addresses): the
/// tiles looked up, or the map walked when it is the shorter – the same values either way.
pub(super) fn for_each_touched<V, S: std::hash::BuildHasher>(tiles: &TileSet, map: &HashMap<usize, V, S>, mut f: impl FnMut(&V)) {
    if tiles.len() <= map.len() {
        for a in tiles.iter() {
            if let Some(v) = map.get(&a) {
                f(v);
            }
        }
    } else {
        for (&a, v) in map {
            if tiles.contains(a) {
                f(v);
            }
        }
    }
}

/// what the lists submitted since the last update touched: tiles, and tagged keys
#[derive(Default)]
pub(super) struct FrameKeys {
    pub(super) tiles: TileSet,
    pub(super) tagged: FxSet<u64>,
}

impl FrameKeys {
    pub(super) fn len(&self) -> usize {
        self.tiles.len() + self.tagged.len()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `k` in, unless `FRAME_KEYS_MAX` keys are in already (false: cut).
    pub(super) fn insert(&mut self, k: u64) -> bool {
        let full = self.len() >= FRAME_KEYS_MAX;
        if k & TAGS != 0 {
            if full && !self.tagged.contains(&k) {
                return false;
            }
            self.tagged.insert(k);
        } else {
            if full && !self.tiles.contains_index(k) {
                return false;
            }
            self.tiles.insert(k);
        }
        true
    }
}

static FRAME: Mutex<Option<FrameKeys>> = Mutex::new(None);

/// The PC lists `lists` were submitted: their logs go in the frame's set (they stay with the
/// lists: one submitted again counts again).
pub(super) fn submitted(lists: &[*mut std::ffi::c_void]) {
    if !on() || lists.is_empty() {
        return;
    }
    let t = Instant::now();
    flush_local();
    let mut cut = 0;
    {
        // (`FRAME`, then a part of `LOGS`: nothing takes them the other way round)
        let mut f = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        let set = f.get_or_insert_with(FrameKeys::default);
        for &l in lists {
            LOGS.peek(l as usize, |log| {
                for &k in &log.keys {
                    cut += !set.insert(k) as u64;
                }
            });
        }
    }
    if cut != 0 {
        CUT.fetch_add(cut, Ordering::Relaxed);
    }
    MERGE_NS.fetch_add(t.elapsed().as_nanos() as u64, Ordering::Relaxed);
}

fn take_frame() -> FrameKeys {
    FRAME.lock().unwrap_or_else(|e| e.into_inner()).take().unwrap_or_default()
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

/// what the frames since the last update touched, by kind (resources sorted)
#[derive(Default)]
pub(super) struct Touched {
    pub tiles: TileSet,
    pub res: Vec<usize>,
    pub psos: usize,
}

/// `keys` sorted out; a table's slots name resources by `resources_at` (`f` with each one found
/// at the addresses given), and a resource of a console buffer is its first `RES_TILES_MAX` tiles
/// (`console_buffers`: `f` with the address and bytes of each of the resources given that is one).
pub(super) fn sort_keys(
    keys: FrameKeys,
    resources_at: impl FnOnce(&[usize], &mut dyn FnMut(usize)),
    console_buffers: impl FnOnce(&[usize], &mut dyn FnMut(usize, usize)),
) -> Touched {
    let mut t = Touched { tiles: keys.tiles, ..Touched::default() };
    let mut res = FxSet::default();
    let mut slots = Vec::new();
    for k in keys.tagged {
        match decode(k) {
            Key::Tile(_) => {}
            Key::Res(r) => {
                res.insert(r);
            }
            Key::Pso(_) => t.psos += 1,
            Key::Table(at) => slots.extend((0..TABLE_SLOTS).map(|s| at + 32 * s)),
        }
    }
    slots.sort_unstable();
    slots.dedup();
    resources_at(&slots, &mut |r| {
        res.insert(r);
    });
    t.res = res.into_iter().collect();
    t.res.sort_unstable();
    let tiles = &mut t.tiles;
    console_buffers(&t.res, &mut |p, n| {
        let (keys, _) = va_keys(p as u64, n.min(RES_TILES_MAX * TILE as usize) as u64);
        tiles.insert_range(*keys.start(), *keys.end());
    });
    t
}

/// whether a tile of `tiles` lies in `[base, base + size)`
pub(super) fn any_in(tiles: &TileSet, base: usize, size: usize) -> bool {
    tiles.any_in(base, size)
}

/// heaps made since the last update (`made`): what was known at their address goes
static MADE: Mutex<Vec<usize>> = Mutex::new(Vec::new());

/// A heap was made at `ptr` (sparse pool, target pool, arena).
pub(super) fn made(ptr: usize) {
    if on() {
        MADE.lock().unwrap_or_else(|e| e.into_inner()).push(ptr);
    }
}

/// The worker's: what is known of the heaps, and the counts since the last log line.
struct Hot {
    tracker: Tracker,
    last_log: Instant,
    /// since the last log line: updates, their time (ns, and the longest), the longest scans of
    /// `sparse` and `vram` (their locks are held for them), keys mapped, heaps given a class and
    /// `SetResidencyPriority` calls, pipelines touched
    updates: u64,
    update_ns: u64,
    update_max_ns: u64,
    sparse_max_ns: u64,
    vram_max_ns: u64,
    mapped: u64,
    changed: u64,
    calls: u64,
    psos: u64,
    /// SetResidencyPriority refused: nothing more is set
    broken: bool,
}

// (D3D12 objects are free-threaded; none is held here)
static HOT_STATE: Mutex<Option<Hot>> = Mutex::new(None);

/// presents since the last log line, and their time in `frame` (ns, and the longest)
static FRAMES: AtomicU64 = AtomicU64::new(0);
static PRESENT_NS: AtomicU64 = AtomicU64::new(0);
static PRESENT_MAX_NS: AtomicU64 = AtomicU64::new(0);

/// Wakes the thread that updates: at a present at most every `UPDATE_EVERY`, with the present's
/// time (the classes are of then). A present while it is at one does not wait: the next update
/// is at the latest present's time.
pub(super) struct Worker {
    signal: Mutex<Signal>,
    wake: Condvar,
}

struct Signal {
    pending: Option<Instant>,
    last: Option<Instant>,
    stop: bool,
}

impl Worker {
    pub(super) const fn new() -> Worker {
        Worker { signal: Mutex::new(Signal { pending: None, last: None, stop: false }), wake: Condvar::new() }
    }

    /// A present at `now`: whether an update is due (the thread is woken for it).
    pub(super) fn frame(&self, now: Instant) -> bool {
        let mut s = self.signal.lock().unwrap_or_else(|e| e.into_inner());
        if s.last.is_some_and(|t| now.saturating_duration_since(t) < UPDATE_EVERY) {
            return false;
        }
        s.last = Some(now);
        s.pending = Some(now);
        drop(s);
        self.wake.notify_one();
        true
    }

    /// The thread's: the next update's time, when it is due (None: stop).
    pub(super) fn next(&self) -> Option<Instant> {
        let mut s = self.signal.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if s.stop {
                return None;
            }
            if let Some(t) = s.pending.take() {
                return Some(t);
            }
            s = self.wake.wait(s).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// The thread ends (tests).
    #[cfg(test)]
    pub(super) fn stop(&self) {
        self.signal.lock().unwrap_or_else(|e| e.into_inner()).stop = true;
        self.wake.notify_all();
    }

    /// A thread that runs `work` at each update, until `stop`.
    pub(super) fn spawn(&'static self, mut work: impl FnMut(Instant) + Send + 'static) -> std::io::Result<std::thread::JoinHandle<()>> {
        std::thread::Builder::new().name("xcl hot residency".into()).spawn(move || {
            while let Some(now) = self.next() {
                work(now);
            }
        })
    }
}

static WORKER: Worker = Worker::new();
static SPAWN: Once = Once::new();
/// no thread could be made: the present updates, as it did
static INLINE: AtomicBool = AtomicBool::new(false);

/// A frame ends (a present): what this thread recorded goes to the lists' logs, and at most every
/// `UPDATE_EVERY` the worker thread is woken to map the touches onto heaps and set their classes
/// (every `LOG_EVERY` a line).
pub(super) fn frame() {
    if !on() {
        return;
    }
    let now = Instant::now();
    flush_local();
    SPAWN.call_once(|| {
        if let Err(e) = WORKER.spawn(|now| unsafe { update_and_log(now) }) {
            INLINE.store(true, Ordering::Relaxed);
            crate::xlog_warn!("hot residency: no worker thread ({e}): the presenting thread updates");
        }
    });
    FRAMES.fetch_add(1, Ordering::Relaxed);
    if WORKER.frame(now) && INLINE.load(Ordering::Relaxed) {
        unsafe { update_and_log(now) };
    }
    let ns = now.elapsed().as_nanos() as u64;
    PRESENT_NS.fetch_add(ns, Ordering::Relaxed);
    PRESENT_MAX_NS.fetch_max(ns, Ordering::Relaxed);
}

/// An update for the present at `now`, and a line when one is due.
unsafe fn update_and_log(now: Instant) {
    let start = Instant::now();
    let mut g = HOT_STATE.lock().unwrap_or_else(|e| e.into_inner());
    let h = g.get_or_insert_with(|| Hot {
        tracker: Tracker::default(),
        last_log: now,
        updates: 0,
        update_ns: 0,
        update_max_ns: 0,
        sparse_max_ns: 0,
        vram_max_ns: 0,
        mapped: 0,
        changed: 0,
        calls: 0,
        psos: 0,
        broken: false,
    });
    update(h, now);
    let ns = start.elapsed().as_nanos() as u64;
    h.updates += 1;
    h.update_ns += ns;
    h.update_max_ns = h.update_max_ns.max(ns);
    if now.saturating_duration_since(h.last_log) >= LOG_EVERY {
        log(h, now);
    }
}

unsafe fn update(h: &mut Hot, now: Instant) {
    let keys = take_frame();
    h.mapped += keys.len() as u64;
    for p in std::mem::take(&mut *MADE.lock().unwrap_or_else(|e| e.into_inner())) {
        h.tracker.forget(p);
    }
    let t = sort_keys(keys, |at, f| super::console_heap::resources_at(at, f), |res, f| super::real::console_buffers_of(res, f));
    h.psos += t.psos as u64;
    h.tracker.begin();
    let mut out: Vec<(ID3D12Heap, Class)> = Vec::new();
    let tracker = &mut h.tracker;
    let mut see = |v: &HeapView| tracker.see(v, now);
    let t0 = Instant::now();
    super::sparse::hot_scan(&t.tiles, &t.res, &mut see, &mut out);
    let t1 = Instant::now();
    super::vram::hot_scan(&t.tiles, &t.res, &mut see, &mut out);
    h.sparse_max_ns = h.sparse_max_ns.max((t1 - t0).as_nanos() as u64);
    h.vram_max_ns = h.vram_max_ns.max(t1.elapsed().as_nanos() as u64);
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
    let frames = FRAMES.swap(0, Ordering::Relaxed).max(1);
    let present = PRESENT_NS.swap(0, Ordering::Relaxed);
    let present_max = PRESENT_MAX_NS.swap(0, Ordering::Relaxed);
    let merge = MERGE_NS.swap(0, Ordering::Relaxed);
    let recorded = RECORDED.swap(0, Ordering::Relaxed);
    let cut = CUT.swap(0, Ordering::Relaxed);
    crate::xlog_info!(
        "{text}; cost {:.1} µs a frame on the presenting thread (longest {} µs) + {:.1} µs a frame at the submits; worker {:.1} µs per update ({} updates, longest {} µs; sparse scan longest {} µs, vram scan longest {} µs); {} keys recorded and {} mapped a frame, {} cut, {} pipelines touched an update; {} heaps changed class in {} SetResidencyPriority calls over {} frames",
        present as f64 / 1e3 / frames as f64,
        present_max / 1000,
        merge as f64 / 1e3 / frames as f64,
        h.update_ns as f64 / 1e3 / h.updates.max(1) as f64,
        h.updates,
        h.update_max_ns / 1000,
        h.sparse_max_ns / 1000,
        h.vram_max_ns / 1000,
        recorded / frames,
        h.mapped / frames,
        cut,
        h.psos / h.updates.max(1),
        h.changed,
        h.calls,
        frames
    );
    h.last_log = now;
    h.updates = 0;
    h.update_ns = 0;
    h.update_max_ns = 0;
    h.sparse_max_ns = 0;
    h.vram_max_ns = 0;
    h.mapped = 0;
    h.changed = 0;
    h.calls = 0;
    h.psos = 0;
}
