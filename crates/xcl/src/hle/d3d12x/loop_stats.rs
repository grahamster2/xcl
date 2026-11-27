//! `XCL_LOOP_STATS` (`gpu::loop_probe`): what the console's compute programs are given and what
//! their loops do, per program, logged every `XCL_LOOP_STATS_EVERY` seconds (10 by default) as
//! `D3D12.X: XCL_LOOP_STATS: …`.
//!
//! For each program dispatched in that time: its dispatches and thread groups (a direct call's
//! counts; an ExecuteIndirect's as `indirect_guard` read them on the GPU), how many of those had
//! more than 65 535 groups along a dimension and how often the size jumped (four times the call
//! before, or a quarter of it); the distinct addresses of its root views (and 64-bit root
//! constants that lie in a sparse window), and for each whether its tile there has no entry, an
//! uncommitted page, no video memory, or video memory no CPU byte ever went to (`sparse::Held`).
//! For the programs whose HLSL has the counters (named in the switch), what their loop guards
//! cut: lanes that spent their whole loop budget, loop entries ended by their count, lanes past
//! 1 024 and 16 384 passes, split-wave exchanges and barriers that gave up.
//!
//! The counters of a program are a slot of one block of committed unified memory that its
//! dispatches bind at `res` (u1) in place of the dispatch's zeroed 256 bytes: they add up over
//! all its dispatches, and the report takes the difference since the last one.

use std::collections::{HashMap, VecDeque};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use super::console_list::{self, Layout, Param};
use super::sparse::Held;
use crate::gpu::loop_probe::{self as probe, WORDS};

/// bytes of a program's counters (`WORDS` used)
const SLOT_BYTES: usize = 256;
const BLOCK_BYTES: usize = 64 << 10;
const SLOTS: usize = BLOCK_BYTES / SLOT_BYTES;
/// distinct root addresses kept for a program between two reports
const ADDRS_KEPT: usize = 64;
/// ExecuteIndirects a program waits for the counts of (older ones are given up)
const PENDING_KEPT: usize = 4096;
/// reports an ExecuteIndirect's counts may take to be read before they are given up
const PENDING_REPORTS: u32 = 3;
/// what D3D12 dispatches along a dimension at most
const DIM_MAX: u32 = 65_535;
/// programs listed by threads (besides those with counters or with a flag)
const TOP: usize = 8;
/// programs a report lists at most
const LISTED_MOST: usize = 32;
/// unfilled root addresses named for a program
const NAMED_MOST: usize = 4;

pub(super) fn on() -> bool {
    probe::switch().on()
}

/// `XCL_LOOP_STATS_EVERY` (seconds, 1 at least): the time between two reports
fn every() -> Duration {
    static E: OnceLock<Duration> = OnceLock::new();
    *E.get_or_init(|| Duration::from_secs(std::env::var("XCL_LOOP_STATS_EVERY").ok().and_then(|v| v.trim().parse::<u64>().ok()).filter(|&s| s > 0).unwrap_or(10)))
}

/// The counters' block: its CPU address and its GPU address.
struct Block {
    at: usize,
    va: u64,
}

fn block() -> Option<&'static Block> {
    static B: OnceLock<Option<Block>> = OnceLock::new();
    B.get_or_init(|| {
        let u = crate::gpu::uma::get()?;
        // (zeroed, committed: the GPU and the CPU see it)
        let at = u.alloc(BLOCK_BYTES, BLOCK_BYTES)? as usize;
        let va = super::real::uma_va_of(at)?;
        crate::xlog_info!("D3D12.X: XCL_LOOP_STATS: loop counters of up to {SLOTS} programs at {at:#x} (GPU {va:#x})");
        Some(Block { at, va })
    })
    .as_ref()
}

/// program (dump hash) → its slot of the block
static SLOT_OF: Mutex<Option<HashMap<u64, usize>>> = Mutex::new(None);

fn slot_of(cs: u64) -> Option<usize> {
    let mut g = SLOT_OF.lock().unwrap();
    let m = g.get_or_insert_with(HashMap::new);
    if let Some(&k) = m.get(&cs) {
        return Some(k);
    }
    if m.len() >= SLOTS {
        static N: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(0);
        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed) == 0 {
            crate::xlog_warn!("D3D12.X: XCL_LOOP_STATS: more than {SLOTS} programs with counters: cs {cs:016x} and later ones bind none");
        }
        return None;
    }
    let k = m.len();
    m.insert(cs, k);
    Some(k)
}

/// The counter slot a dispatch of `pso` binds at `res`: None unless its CS was built with the
/// counters (`loop_probe::counted`).
pub(super) fn res_for(pso: usize) -> Option<u64> {
    let (cs, _) = super::real::pso_shaders(super::pso_async::resolve(pso))?;
    if !probe::counted(cs) {
        return None;
    }
    let (b, k) = (block()?, slot_of(cs)?);
    Some(b.va + (k * SLOT_BYTES) as u64)
}

/// The counters at `at` (a slot), as the GPU has added to them so far.
///
/// # Safety
/// `at` is `WORDS` readable, 4-byte aligned words.
unsafe fn read_words(at: *const u32) -> [u32; WORDS] {
    std::array::from_fn(|k| std::ptr::read_volatile(at.add(k)))
}

/// What a program was given in one report's time.
#[derive(Clone, Debug, Default, PartialEq)]
struct Window {
    calls: u64,
    direct: u64,
    indirect: u64,
    /// ExecuteIndirects whose counts were read / never will be
    read: u64,
    unread: u64,
    groups: u64,
    threads: u64,
    /// most groups in one call
    most: u64,
    last: [u32; 3],
    /// calls with more than `DIM_MAX` groups along a dimension
    over_dim: u64,
    /// calls of four times the groups of the call before or a quarter of them (64 groups apart at least)
    jumps: u64,
    /// distinct root addresses: (address, a root view – else a 64-bit root constant)
    addrs: Vec<(u64, bool)>,
    /// sightings of addresses past `ADDRS_KEPT` (not kept, so each one counts)
    addrs_left_out: u64,
}

/// A call of `n` groups after one of `before`: a jump in size.
fn jump(before: u64, n: u64) -> bool {
    let (lo, hi) = (before.min(n), before.max(n));
    hi - lo >= 64 && hi >= 4 * lo
}

#[derive(Debug, Default)]
struct Prog {
    win: Window,
    /// groups of the call before (any window)
    prev: Option<u64>,
    /// console threads of a group
    group_threads: u64,
    /// guard slots of ExecuteIndirects not read yet, with the reports they waited
    pending: VecDeque<(u32, u32)>,
    /// counters at the last report
    seen: [u32; WORDS],
}

impl Prog {
    fn groups(&mut self, g: [u32; 3]) {
        let n: u64 = g.iter().map(|&d| d as u64).product();
        let w = &mut self.win;
        w.groups += n;
        w.threads += n * self.group_threads.max(1);
        w.most = w.most.max(n);
        w.last = g;
        if g.iter().any(|&d| d > DIM_MAX) {
            w.over_dim += 1;
        }
        if self.prev.is_some_and(|p| jump(p, n)) {
            w.jumps += 1;
        }
        self.prev = Some(n);
    }

    fn addr(&mut self, va: u64, view: bool) {
        let w = &mut self.win;
        if w.addrs.iter().any(|&(a, _)| a == va) {
            return;
        }
        if w.addrs.len() >= ADDRS_KEPT {
            w.addrs_left_out += 1;
        } else {
            w.addrs.push((va, view));
        }
    }

    /// The ExecuteIndirects whose counts `counts` (a guard slot's) gives now; those waiting
    /// longer than `PENDING_REPORTS` reports are given up.
    fn read_pending(&mut self, counts: impl Fn(u32) -> Option<[u32; 3]>) {
        let mut left = VecDeque::with_capacity(self.pending.len());
        while let Some((slot, age)) = self.pending.pop_front() {
            match counts(slot) {
                Some(g) => {
                    self.win.read += 1;
                    self.groups(g);
                }
                None if age + 1 >= PENDING_REPORTS => self.win.unread += 1,
                None => left.push_back((slot, age + 1)),
            }
        }
        self.pending = left;
    }
}

/// program (dump hash; 0: not known) → what it was given
static PROGS: Mutex<Option<HashMap<u64, Prog>>> = Mutex::new(None);

fn cs_of(pso: usize) -> u64 {
    super::real::pso_shaders(super::pso_async::resolve(pso)).map_or(0, |(cs, _)| cs)
}

/// The 64-bit values at the root constants of `args` (packed by `l`) that may be addresses
/// (4-byte aligned, not 0), as `bind::store_addresses` takes them.
fn constant_addresses<'a>(l: &'a Layout, args: &'a [u8]) -> impl Iterator<Item = u64> + 'a {
    l.params
        .iter()
        .flat_map(move |p| {
            let (at, count) = match *p {
                Param::Constants { at, count } => (at as usize, count as usize),
                _ => (0, 0),
            };
            (0..count.saturating_sub(1)).filter_map(move |k| args.get(at + 4 * k..at + 4 * k + 8).map(|b| u64::from_le_bytes(b.try_into().unwrap())))
        })
        .filter(|&va| va != 0 && va % 4 == 0)
}

/// A dispatch of the translated compute pipeline `pso` with the root arguments `args` (packed
/// by `layout`): `groups` for a direct one, None for an ExecuteIndirect (`indirect` follows).
pub(super) fn note_dispatch(pso: usize, groups: Option<[u32; 3]>, layout: Option<&Layout>, args: &[u8]) {
    start();
    let cs = cs_of(pso);
    let threads = super::real::pso_threads(pso).map_or(1, |t| t.iter().map(|&d| d.max(1) as u64).product());
    let mut g = PROGS.lock().unwrap();
    let p = g.get_or_insert_with(HashMap::new).entry(cs).or_default();
    p.group_threads = threads;
    p.win.calls += 1;
    match groups {
        Some(gr) => {
            p.win.direct += 1;
            p.groups(gr);
        }
        None => p.win.indirect += 1,
    }
    if let Some(l) = layout {
        for va in console_list::root_views(l, args).filter(|&va| va != 0) {
            p.addr(va, true);
        }
        for va in constant_addresses(l, args) {
            p.addr(va, false);
        }
    }
}

/// The ExecuteIndirect `note_dispatch` just saw for `pso`: its counts are in the guard's slot
/// `slot` once the GPU has read them (None: not checked, so not known here).
pub(super) fn note_indirect(pso: usize, slot: Option<u32>) {
    let cs = cs_of(pso);
    let mut g = PROGS.lock().unwrap();
    let p = g.get_or_insert_with(HashMap::new).entry(cs).or_default();
    match slot {
        Some(s) => {
            if p.pending.len() >= PENDING_KEPT {
                p.pending.pop_front();
                p.win.unread += 1;
            }
            p.pending.push_back((s, 0));
        }
        None => p.win.unread += 1,
    }
}

fn start() {
    static T: OnceLock<()> = OnceLock::new();
    T.get_or_init(|| {
        let every = every();
        crate::xlog_info!(
            "D3D12.X: XCL_LOOP_STATS: dispatch statistics every {} s{}",
            every.as_secs(),
            match probe::switch() {
                probe::Switch::All => ", loop counters in every compute program".to_string(),
                probe::Switch::Named(l) => format!(", loop counters in {} named program(s)", l.len()),
                _ => ", no loop counters (name programs in XCL_LOOP_STATS to count their loops)".to_string(),
            }
        );
        let _ = std::thread::Builder::new().name("xcl-loop-stats".into()).spawn(move || {
            let mut since = Instant::now();
            loop {
                std::thread::sleep(every);
                let secs = since.elapsed().as_secs_f64();
                since = Instant::now();
                for line in report(secs) {
                    crate::xlog_info!("D3D12.X: XCL_LOOP_STATS: {line}");
                }
            }
        });
    });
}

/// One program's report.
#[derive(Clone, Debug, Default, PartialEq)]
struct Row {
    cs: u64,
    path: Option<&'static str>,
    win: Window,
    /// the counters' growth since the last report and their totals (None: no counters)
    loops: Option<([u32; WORDS], [u32; WORDS])>,
    /// `Held` of each of `win.addrs`
    held: Vec<Held>,
}

/// The counters' growth from `before` to `now` (they wrap at 2^32).
fn grown(now: [u32; WORDS], before: [u32; WORDS]) -> [u32; WORDS] {
    std::array::from_fn(|k| now[k].wrapping_sub(before[k]))
}

/// The rows of this report: every program's window taken (and reset), the ExecuteIndirect counts
/// read, the counters read.
fn take_rows(counts: impl Fn(u32) -> Option<[u32; 3]>, words: impl Fn(u64) -> Option<[u32; WORDS]>) -> Vec<Row> {
    let mut g = PROGS.lock().unwrap();
    let Some(m) = g.as_mut() else { return Vec::new() };
    let mut rows = Vec::with_capacity(m.len());
    for (&cs, p) in m.iter_mut() {
        p.read_pending(&counts);
        let loops = words(cs).map(|now| {
            let d = grown(now, p.seen);
            p.seen = now;
            (d, now)
        });
        let win = std::mem::take(&mut p.win);
        if win.calls == 0 && loops.map_or(true, |(d, _)| d.iter().all(|&x| x == 0)) {
            continue;
        }
        rows.push(Row { cs, path: None, win, loops, held: Vec::new() });
    }
    rows
}

fn report(secs: f64) -> Vec<String> {
    let words = |cs: u64| -> Option<[u32; WORDS]> {
        if !probe::counted(cs) {
            return None;
        }
        let k = *SLOT_OF.lock().unwrap().as_ref()?.get(&cs)?;
        let b = block()?;
        // SAFETY: the slot lies in the block, which is never freed
        Some(unsafe { read_words((b.at + k * SLOT_BYTES) as *const u32) })
    };
    let mut rows = take_rows(super::indirect_guard::counts, words);
    let chosen = choose(&rows);
    let mut out = Vec::new();
    for k in chosen {
        let r = &mut rows[k];
        r.path = crate::gpu::recomp::pso::compute_path(r.cs);
        let vas: Vec<usize> = r.win.addrs.iter().map(|&(a, _)| a as usize).collect();
        r.held = super::sparse::held(&vas);
        out.extend(lines(r, secs));
    }
    out
}

fn loop_hits(r: &Row) -> u64 {
    use probe::*;
    r.loops.map_or(0, |(d, _)| [W_BUDGET_LANES, W_CAP_LANES, W_X_TIMEOUTS, W_BAR_TIMEOUTS, W_PAST_16K].iter().map(|&w| d[w] as u64).sum())
}

/// The rows a report lists, in its order: those with counters or with something flagged first
/// (most loop hits, then most flags), then the `TOP` others with the most threads.
fn choose(rows: &[Row]) -> Vec<usize> {
    let flagged = |r: &Row| r.loops.is_some() || r.win.over_dim > 0 || r.win.unread > 0;
    let mut first: Vec<usize> = (0..rows.len()).filter(|&k| flagged(&rows[k])).collect();
    first.sort_by_key(|&k| (std::cmp::Reverse(loop_hits(&rows[k])), std::cmp::Reverse(rows[k].win.over_dim + rows[k].win.jumps), std::cmp::Reverse(rows[k].win.threads), rows[k].cs));
    let mut rest: Vec<usize> = (0..rows.len()).filter(|&k| !flagged(&rows[k])).collect();
    rest.sort_by_key(|&k| (std::cmp::Reverse(rows[k].win.threads), std::cmp::Reverse(rows[k].win.calls), rows[k].cs));
    first.into_iter().chain(rest.into_iter().take(TOP)).take(LISTED_MOST).collect()
}

/// A row's lines.
fn lines(r: &Row, secs: f64) -> Vec<String> {
    use probe::*;
    let w = &r.win;
    let name = format!("cs {:016x} (dump hash {:016x}, {})", probe::cs_hash(r.cs), r.cs, r.path.unwrap_or("path not known"));
    let mut out = vec![format!(
        "{name}: {} dispatch(es) in {secs:.1} s ({} direct, {} ExecuteIndirect: {} with counts read, {} not), {} groups, {} threads, at most {} groups a call (last {}x{}x{}); {} call(s) over {DIM_MAX} groups a dimension, {} size jump(s) (x4 from the call before)",
        w.calls, w.direct, w.indirect, w.read, w.unread, w.groups, w.threads, w.most, w.last[0], w.last[1], w.last[2], w.over_dim, w.jumps
    )];
    out.push(match r.loops {
        Some((d, t)) => format!(
            "{name}: loops: {} lane(s) spent their 65536 passes ({} loop exits for it), {} lane(s) had a loop entry cut at its count ({} entries cut), {} lane(s) past 1024 passes, {} past 16384; {} split-wave exchange timeout(s) (xcl_x_dead), {} barrier timeout(s) – since the start: {} spent, {} cut, {} past 16384, {} exchange timeouts",
            d[W_BUDGET_LANES], d[W_BUDGET_EXITS], d[W_CAP_LANES], d[W_CAP_EXITS], d[W_PAST_1K], d[W_PAST_16K], d[W_X_TIMEOUTS], d[W_BAR_TIMEOUTS], t[W_BUDGET_LANES], t[W_CAP_LANES], t[W_PAST_16K], t[W_X_TIMEOUTS]
        ),
        None => format!("{name}: loops: no counters (name the program in XCL_LOOP_STATS)"),
    });
    // root ranges: views always, 64-bit constants when they lie in a sparse window
    let mut n = [0u64; 6];
    let (mut views, mut consts) = (0u64, 0u64);
    let mut unfilled = Vec::new();
    for (&(va, view), &h) in w.addrs.iter().zip(&r.held) {
        if !view && h == Held::NotSparse {
            continue;
        }
        if view {
            views += 1;
        } else {
            consts += 1;
        }
        n[h as usize] += 1;
        if matches!(h, Held::NotMapped | Held::Hole | Held::NoVideoMemory | Held::NoCpuBytes) && unfilled.len() < NAMED_MOST {
            unfilled.push(format!("{va:#x} ({})", held_name(h)));
        }
    }
    let left_out = if w.addrs_left_out > 0 { format!(" ({} sighting(s) of further addresses not looked at)", w.addrs_left_out) } else { String::new() };
    out.push(format!(
        "{name}: root ranges: {views} view(s), {consts} 64-bit constant(s) in a sparse window{left_out}: {} outside the sparse windows, {} in no tile, {} in an uncommitted page, {} with no video memory, {} where no CPU byte went (zeros unless the GPU wrote them), {} with CPU bytes{}",
        n[Held::NotSparse as usize],
        n[Held::NotMapped as usize],
        n[Held::Hole as usize],
        n[Held::NoVideoMemory as usize],
        n[Held::NoCpuBytes as usize],
        n[Held::CpuBytes as usize],
        if unfilled.is_empty() { String::new() } else { format!("; unfilled: {}", unfilled.join(", ")) }
    ));
    out
}

fn held_name(h: Held) -> &'static str {
    match h {
        Held::NotSparse => "not sparse",
        Held::NotMapped => "no tile",
        Held::Hole => "uncommitted page",
        Held::NoVideoMemory => "no video memory",
        Held::NoCpuBytes => "no CPU byte",
        Held::CpuBytes => "CPU bytes",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use probe::*;

    /// (the tests share `PROGS`: one at a time)
    static ONE: Mutex<()> = Mutex::new(());

    #[test]
    fn a_size_jump_is_four_times_and_64_groups_apart() {
        assert!(jump(15, 600) && jump(600, 15) && jump(0, 64));
        assert!(!jump(15, 28) && !jump(16, 64) && !jump(0, 63) && !jump(100, 399));
        assert!(jump(100, 400));
    }

    #[test]
    fn calls_add_up_with_their_flags() {
        let mut p = Prog { group_threads: 64, ..Default::default() };
        p.groups([160, 90, 1]);
        p.groups([160, 90, 1]);
        p.groups([70_000, 1, 1]);
        let w = &p.win;
        assert_eq!((w.groups, w.threads, w.most, w.last), (2 * 14_400 + 70_000, 64 * (2 * 14_400 + 70_000), 70_000, [70_000, 1, 1]));
        assert_eq!((w.over_dim, w.jumps), (1, 1));
        // the call before counts across windows
        p.win = Window::default();
        p.groups([1, 1, 1]);
        assert_eq!(p.win.jumps, 1);
    }

    #[test]
    fn root_addresses_are_kept_once_and_up_to_a_limit() {
        let mut p = Prog::default();
        for k in 0..(ADDRS_KEPT as u64 + 3) {
            p.addr(0x1000 * k, true);
            p.addr(0x1000 * k, false);
        }
        // (each sighting of one not kept counts: it is not remembered)
        assert_eq!((p.win.addrs.len(), p.win.addrs_left_out), (ADDRS_KEPT, 6));
        assert!(p.win.addrs.iter().all(|&(_, view)| view), "the first sighting decides");
    }

    #[test]
    fn indirect_counts_are_read_once_the_gpu_has_them_or_given_up() {
        let mut p = Prog { group_threads: 32, ..Default::default() };
        p.pending.extend([(1, 0), (2, 0), (3, 0)]);
        let ready = |s: u32| (s != 2).then_some([s * 10, 1, 1]);
        p.read_pending(ready);
        assert_eq!((p.win.read, p.win.unread, p.win.groups), (2, 0, 40));
        assert_eq!(p.pending, VecDeque::from([(2, 1)]));
        for _ in 0..PENDING_REPORTS {
            p.read_pending(ready);
        }
        assert_eq!((p.win.read, p.win.unread), (2, 1));
        assert!(p.pending.is_empty());
    }

    #[test]
    fn root_constants_that_may_be_addresses() {
        let l = Layout { params: vec![Param::Constants { at: 0, count: 4 }, Param::Descriptor { at: 16 }], size: 24, inline: 0, ..Default::default() };
        let mut args = vec![0u8; 24];
        args[0..8].copy_from_slice(&0x2_0000_1000u64.to_le_bytes());
        args[16..24].copy_from_slice(&0x3_0000_0000u64.to_le_bytes());
        let consts: Vec<u64> = constant_addresses(&l, &args).collect();
        assert_eq!(consts, vec![0x2_0000_1000]);
        assert_eq!(console_list::root_views(&l, &args).collect::<Vec<_>>(), vec![0x3_0000_0000]);
    }

    /// The counters a report gives: their growth since the last one, also across a wrap, and the
    /// totals read from the slot.
    #[test]
    fn counters_are_read_back_as_their_growth() {
        let mut slot = [0u32; WORDS];
        slot[W_BUDGET_LANES] = 64;
        slot[W_PAST_16K] = 64;
        // SAFETY: a local array of `WORDS` words
        let now = unsafe { read_words(slot.as_ptr()) };
        assert_eq!(now, slot);
        let mut before = [0u32; WORDS];
        before[W_PAST_16K] = u32::MAX;
        let g = grown(now, before);
        assert_eq!((g[W_BUDGET_LANES], g[W_PAST_16K], g[W_CAP_LANES]), (64, 65, 0));
    }

    #[test]
    fn a_report_takes_each_window_and_the_counters_once() {
        let _one = ONE.lock().unwrap();
        *PROGS.lock().unwrap() = None;
        {
            let mut g = PROGS.lock().unwrap();
            let m = g.get_or_insert_with(HashMap::new);
            let p = m.entry(0xAA).or_default();
            p.group_threads = 64;
            p.win.calls = 1;
            p.win.direct = 1;
            p.groups([2, 1, 1]);
            m.entry(0xBB).or_default();
        }
        let mut c = [0u32; WORDS];
        c[W_CAP_LANES] = 5;
        let rows = take_rows(|_| None, |cs| (cs == 0xAA).then_some(c));
        assert_eq!(rows.len(), 1, "a program with no calls and no counts is left out");
        assert_eq!(rows[0].cs, 0xAA);
        assert_eq!(rows[0].loops.unwrap().0[W_CAP_LANES], 5);
        // the next report: nothing new
        let rows = take_rows(|_| None, |cs| (cs == 0xAA).then_some(c));
        assert!(rows.is_empty(), "{rows:?}");
        c[W_CAP_LANES] = 7;
        let rows = take_rows(|_| None, |cs| (cs == 0xAA).then_some(c));
        assert_eq!(rows[0].loops.unwrap(), ({ let mut d = [0; WORDS]; d[W_CAP_LANES] = 2; d }, c));
        *PROGS.lock().unwrap() = None;
    }

    fn row(cs: u64, threads: u64) -> Row {
        Row { cs, win: Window { calls: 1, threads, ..Default::default() }, ..Default::default() }
    }

    #[test]
    fn counted_and_flagged_programs_come_first_then_the_busiest() {
        let mut rows: Vec<Row> = (0..20).map(|k| row(k, 1000 + k)).collect();
        rows[3].loops = Some(([0; WORDS], [0; WORDS]));
        let mut hit = [0; WORDS];
        hit[W_BUDGET_LANES] = 9;
        rows[5].loops = Some((hit, hit));
        rows[7].win.over_dim = 1;
        let c = choose(&rows);
        assert_eq!(&c[..3], &[5, 7, 3], "most loop hits first, then flags");
        assert_eq!(&c[3..], &[19, 18, 17, 16, 15, 14, 13, 12]);
    }

    #[test]
    fn the_lines_say_what_the_counters_and_tiles_found() {
        let mut r = row(0xb78f_a10b_425b_8e54, 64 * 14_400);
        r.path = Some("HLSL, with XCL_LOOP_STATS counters");
        r.win.direct = 1;
        r.win.groups = 14_400;
        r.win.most = 14_400;
        r.win.last = [160, 90, 1];
        let mut d = [0; WORDS];
        d[W_CAP_LANES] = 3;
        d[W_CAP_EXITS] = 4;
        d[W_X_TIMEOUTS] = 2;
        r.loops = Some((d, d));
        r.win.addrs = vec![(0x10_0000, true), (0x20_0000, true), (0x30_0000, false), (0x40_0000, false)];
        r.held = vec![Held::NoCpuBytes, Held::CpuBytes, Held::NotSparse, Held::NoVideoMemory];
        let l = lines(&r, 10.0);
        assert_eq!(l.len(), 3);
        let name = format!("cs {:016x} (dump hash b78fa10b425b8e54, HLSL, with XCL_LOOP_STATS counters)", probe::cs_hash(0xb78f_a10b_425b_8e54));
        assert!(l.iter().all(|x| x.starts_with(&name)), "{l:?}");
        assert!(l[0].contains(": 1 dispatch(es) in 10.0 s (1 direct, 0 ExecuteIndirect: 0 with counts read, 0 not), 14400 groups, 921600 threads, at most 14400 groups a call (last 160x90x1)"), "{}", l[0]);
        assert!(l[1].contains("0 lane(s) spent their 65536 passes (0 loop exits for it), 3 lane(s) had a loop entry cut at its count (4 entries cut)"), "{}", l[1]);
        assert!(l[1].contains("2 split-wave exchange timeout(s) (xcl_x_dead)"), "{}", l[1]);
        // (the constant outside the sparse windows is no address)
        assert!(l[2].contains("root ranges: 2 view(s), 1 64-bit constant(s) in a sparse window: 0 outside the sparse windows, 0 in no tile, 0 in an uncommitted page, 1 with no video memory, 1 where no CPU byte went"), "{}", l[2]);
        assert!(l[2].ends_with("1 with CPU bytes; unfilled: 0x100000 (no CPU byte), 0x400000 (no video memory)"), "{}", l[2]);
        r.loops = None;
        assert!(lines(&r, 10.0)[1].ends_with("loops: no counters (name the program in XCL_LOOP_STATS)"));
    }
}
