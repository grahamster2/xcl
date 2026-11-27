//! Pipelines XCL made that have not been used for a while are released, and made again from the
//! driver's blob (in memory or on disk, `real::driver_cache`) or compiled again when one is
//! bound after that.
//!
//! A retail title's largest use of memory on an NVIDIA GPU is the driver's write-combined
//! private memory, ~13 – 15 GB in ~50 MB allocations, growing with the pipelines alive by about
//! 1 – 2.5 MB each – far more than the title's allocators (releasing 1 751 spare ones gave back
//! 0.7 GB). Two switches, both off by default:
//! * `XCL_PSO_IDLE_S=<seconds>`: a pipeline the background builds made for one of the title's
//!   stand-ins (`pso_async`: translated compute and graphics pipelines, split programs with their
//!   pieces) that no bind has asked for in that long is released. The title holds the stand-in,
//!   never the driver pipeline, and every bind asks `pso_async` for the driver pipeline of the
//!   stand-in it has: the next one makes it again.
//! * `XCL_SPLIT_DROP_WHOLE=1`: the whole form of a split compute program (`dispatch_chain`),
//!   built for an `ExecuteIndirect` or a dispatch without scratch, is released when it has not
//!   been used for `XCL_PSO_IDLE_S` (30 s when that is unset). Its driver blob is kept on disk.
//!
//! What may still use a pipeline when it is taken out of use (`bury`):
//! * a list recording, or recorded and not yet submitted, or submitted again without a new
//!   recording (as a bundle is): every list whose recording began before the pipeline was taken
//!   out of use holds it until the list is reset or released. Lists are known from their
//!   creation (`real::adopt`), each Reset (`real::cl_reset`, `alloc_pool::reset_list`) and their
//!   final Release; one first seen at a submit, its beginning unknown, holds them all;
//! * the GPU: once no such list is left, the pipeline goes to `late_release`, which releases it
//!   after every queue has passed a fence value signalled then.
//!
//! A bind that resolved the pipeline just before it was taken out of use records it on a list
//! that began before: that list holds it as above.
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

/// how often the sweep looks for pipelines not used for long enough
const SWEEP: Duration = Duration::from_secs(5);
/// `XCL_SPLIT_DROP_WHOLE` without `XCL_PSO_IDLE_S`
const WHOLE_IDLE: Duration = Duration::from_secs(30);

fn parse_idle(v: Option<&str>) -> Option<Duration> {
    v.and_then(|v| v.trim().parse::<f64>().ok()).filter(|s| s.is_finite() && *s > 0.0).map(Duration::from_secs_f64)
}

/// `XCL_PSO_IDLE_S`: how long a pipeline goes unused before it is released (None: never).
pub fn idle_after() -> Option<Duration> {
    static D: OnceLock<Option<Duration>> = OnceLock::new();
    *D.get_or_init(|| parse_idle(std::env::var("XCL_PSO_IDLE_S").ok().as_deref()))
}

/// `XCL_SPLIT_DROP_WHOLE=1`
pub fn drop_whole() -> bool {
    static D: OnceLock<bool> = OnceLock::new();
    *D.get_or_init(|| std::env::var("XCL_SPLIT_DROP_WHOLE").is_ok_and(|v| v.trim() == "1"))
}

/// How long a split program's whole form goes unused before it is released.
pub fn whole_after() -> Option<Duration> {
    drop_whole().then(|| idle_after().unwrap_or(WHOLE_IDLE))
}

/// Either switch is on: uses are timed and lists followed.
pub fn active() -> bool {
    static A: OnceLock<bool> = OnceLock::new();
    *A.get_or_init(|| idle_after().is_some() || drop_whole())
}

/// Milliseconds since the first call: the time of a use.
pub fn now_ms() -> u64 {
    static T0: OnceLock<Instant> = OnceLock::new();
    T0.get_or_init(Instant::now).elapsed().as_millis() as u64 + 1
}

/// Whether a pipeline last used at `last` (ms, 0 = never) is unused for `after` at `now`.
pub fn idle(last: u64, now: u64, after: Duration) -> bool {
    now.saturating_sub(last) >= after.as_millis() as u64
}

/// What a pipeline taken out of use is.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum What {
    Compute,
    Graphics,
    Whole,
}

struct Grave {
    pso: usize,
    /// the ledger's count when it was taken out of use: lists that began before hold it
    epoch: u64,
}

/// The lists that may hold a pipeline taken out of use, and those pipelines.
#[derive(Default)]
pub struct Ledger {
    epoch: u64,
    /// live list → the count when its recording began (0: not known – holds every pipeline)
    lists: HashMap<usize, u64>,
    graves: Vec<Grave>,
}

impl Ledger {
    fn tick(&mut self) -> u64 {
        self.epoch += 1;
        self.epoch
    }

    /// `list` was made, or reset: a new recording begins.
    pub fn began(&mut self, list: usize) {
        let e = self.tick();
        self.lists.insert(list, e);
    }

    /// `list` was adopted: made now, or (a list already followed) given another vtable, which
    /// begins nothing.
    pub fn adopted(&mut self, list: usize) {
        if !self.lists.contains_key(&list) {
            self.began(list);
        }
    }

    /// `list` is submitted: one not seen beginning holds every pipeline until its next Reset.
    pub fn seen(&mut self, list: usize) {
        self.lists.entry(list).or_insert(0);
    }

    /// When the recording of `list` began, as far as what it may hold goes (0: not known).
    pub fn began_at(&self, list: usize) -> u64 {
        self.lists.get(&list).copied().unwrap_or(0)
    }

    /// The recording `list` begins now goes on with state of one that began at `from`
    /// (`began_at`): the pipeline bound there, a split's next segment, a put-off dispatch. It
    /// holds what that one holds.
    pub fn carries(&mut self, list: usize, from: u64) {
        let e = self.lists.entry(list).or_insert(from);
        *e = (*e).min(from);
    }

    /// `list` is released for good.
    pub fn gone(&mut self, list: usize) {
        self.lists.remove(&list);
    }

    /// `pso` (one reference, now the ledger's) is out of use from now.
    pub fn bury(&mut self, pso: usize) {
        let epoch = self.tick();
        self.graves.push(Grave { pso, epoch });
    }

    /// The pipelines no list can hold any more (their references, to release after the GPU is
    /// past them), and how many are still held, by how many lists.
    pub fn take_free(&mut self) -> (Vec<usize>, usize, usize) {
        let mut free = Vec::new();
        let lists = &self.lists;
        self.graves.retain(|g| {
            let held = lists.values().any(|&began| began < g.epoch);
            if !held {
                free.push(g.pso);
            }
            held
        });
        let oldest = self.graves.iter().map(|g| g.epoch).max().unwrap_or(0);
        let holding = self.lists.values().filter(|&&b| b < oldest).count();
        (free, self.graves.len(), holding)
    }

    /// lists alive whose beginning is not known
    pub fn unknown(&self) -> usize {
        self.lists.values().filter(|&&b| b == 0).count()
    }
}

static LEDGER: Mutex<Option<Ledger>> = Mutex::new(None);

fn with<R>(f: impl FnOnce(&mut Ledger) -> R) -> R {
    f(LEDGER.lock().unwrap().get_or_insert_with(Ledger::default))
}

/// The PC list `list` was adopted (`real::adopt`): when it is new, it was just made.
pub fn list_adopted(list: usize) {
    if active() && list != 0 {
        with(|l| l.adopted(list));
    }
}

/// The PC list `list` was reset (its recording begins).
pub fn list_began(list: usize) {
    if active() && list != 0 {
        with(|l| l.began(list));
    }
}

/// When the recording of the PC list `list` began (`Ledger::began_at`; 0 when off).
pub fn list_recording(list: usize) -> u64 {
    if active() {
        with(|l| l.began_at(list))
    } else {
        0
    }
}

/// The recording `list` began goes on with state of one that began at `from` (`list_recording`
/// of that one, read before its own Reset): a KickoffX's Reset that keeps the bound pipeline,
/// a mid-list split's next segment, a put-off dispatch's segment.
pub fn list_carries(list: usize, from: u64) {
    if active() && list != 0 {
        with(|l| l.carries(list, from));
    }
}

/// The PC lists `lists` are being submitted.
pub fn lists_submitted(lists: &[*mut std::ffi::c_void]) {
    if active() {
        with(|l| lists.iter().for_each(|&p| l.seen(p as usize)));
    }
}

/// The PC list `list` is released for good.
pub fn list_gone(list: usize) {
    if active() {
        with(|l| l.gone(list));
    }
}

static EVICTED: [AtomicU64; 3] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
static MADE_AGAIN: AtomicU64 = AtomicU64::new(0);
static MADE_AGAIN_FAILED: AtomicU64 = AtomicU64::new(0);
static RELEASED: AtomicU64 = AtomicU64::new(0);

fn index(w: What) -> usize {
    match w {
        What::Compute => 0,
        What::Graphics => 1,
        What::Whole => 2,
    }
}

/// A pipeline taken out of use was made again at a bind (`ok`), or could not be.
pub fn note_made_again(ok: bool) {
    if ok {
        MADE_AGAIN.fetch_add(1, Ordering::Relaxed);
    } else {
        MADE_AGAIN_FAILED.fetch_add(1, Ordering::Relaxed);
    }
}

/// `pso` (a reference that is now this module's) is out of use: released once no list holds it
/// and the GPU is past it.
pub fn retire(pso: usize, what: What) {
    if pso == 0 {
        return;
    }
    EVICTED[index(what)].fetch_add(1, Ordering::Relaxed);
    with(|l| l.bury(pso));
}

/// Start the sweep (once; nothing when both switches are off).
pub fn start() {
    if !active() {
        return;
    }
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let after = |d: Option<Duration>| d.map_or("off".to_string(), |d| format!("after {:.1} s unused", d.as_secs_f64()));
        crate::xlog_info!(
            "D3D12.X: pipelines XCL made are released when unused and made again at their next bind: background-built pipelines (XCL_PSO_IDLE_S) {}, split programs' whole forms (XCL_SPLIT_DROP_WHOLE) {}",
            after(idle_after()),
            after(whole_after())
        );
        let _ = std::thread::Builder::new().name("xcl-pso-idle".into()).spawn(|| loop {
            std::thread::sleep(SWEEP);
            sweep();
        });
    });
}

/// counts of the last sweep's line: (evicted ×3, made again, failed, released)
static LAST: Mutex<[u64; 6]> = Mutex::new([0; 6]);

fn sweep() {
    let now = now_ms();
    if let Some(after) = idle_after() {
        for (pso, compute) in super::pso_async::evict_idle(now, after) {
            retire(pso, if compute { What::Compute } else { What::Graphics });
        }
    }
    if let Some(after) = whole_after() {
        for pso in super::real::dispatch_chain_drop_wholes(now, after) {
            retire(pso, What::Whole);
        }
    }
    let (free, waiting, holding) = with(|l| l.take_free());
    let unknown = with(|l| l.unknown());
    for &pso in &free {
        // SAFETY: the reference `retire` was given; the PSO's own Release (XCL's override)
        // forgets what XCL keeps by its address
        unsafe { super::late_release::hold_object(pso) };
    }
    RELEASED.fetch_add(free.len() as u64, Ordering::Relaxed);
    let now_counts = [
        EVICTED[0].load(Ordering::Relaxed),
        EVICTED[1].load(Ordering::Relaxed),
        EVICTED[2].load(Ordering::Relaxed),
        MADE_AGAIN.load(Ordering::Relaxed),
        MADE_AGAIN_FAILED.load(Ordering::Relaxed),
        RELEASED.load(Ordering::Relaxed),
    ];
    let mut last = LAST.lock().unwrap();
    if *last == now_counts {
        return;
    }
    let d: Vec<u64> = now_counts.iter().zip(last.iter()).map(|(a, b)| a - b).collect();
    *last = now_counts;
    crate::xlog_info!(
        "D3D12.X: idle pipelines: taken out of use {} compute, {} graphics, {} split whole forms; made again at a bind {} ({} failed); {} released after the GPU (so far: {} out of use, {} made again, {} released); {waiting} held by {holding} lists recorded before ({unknown} lists of unknown beginning)",
        d[0], d[1], d[2], d[3], d[4], free.len(),
        now_counts[0] + now_counts[1] + now_counts[2], now_counts[3], now_counts[5]
    );
}

/// For the memory map: taken out of use so far, made again, released, still held.
pub fn summary() -> String {
    if !active() {
        return String::new();
    }
    let out: u64 = EVICTED.iter().map(|e| e.load(Ordering::Relaxed)).sum();
    let held = LEDGER.lock().unwrap().as_ref().map_or(0, |l| l.graves.len());
    format!(
        "; idle pipelines so far: {out} out of use, {} made again, {} released, {held} still held by lists",
        MADE_AGAIN.load(Ordering::Relaxed),
        RELEASED.load(Ordering::Relaxed)
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_pipeline_is_idle_once_unused_for_the_time_given() {
        let s = Duration::from_secs(10);
        assert!(!idle(5_000, 14_999, s));
        assert!(idle(5_000, 15_000, s));
        assert!(idle(0, 10_000, s), "never used since it was built");
        assert!(!idle(20_000, 15_000, s), "a use after the sweep's clock read is not idle");
        assert_eq!(parse_idle(Some(" 30 ")), Some(Duration::from_secs(30)));
        assert_eq!(parse_idle(Some("0.5")), Some(Duration::from_millis(500)));
        assert_eq!(parse_idle(Some("0")), None);
        assert_eq!(parse_idle(Some("-3")), None);
        assert_eq!(parse_idle(Some("x")), None);
        assert_eq!(parse_idle(None), None);
    }

    #[test]
    fn a_list_that_began_before_holds_the_pipeline_until_it_is_reset() {
        let mut l = Ledger::default();
        l.began(0xA);
        l.bury(0x100);
        l.began(0xB); // began after: cannot have it
        assert_eq!(l.take_free(), (vec![], 1, 1), "list A may have it recorded");
        l.began(0xA); // A reset: its recording is a new one
        assert_eq!(l.take_free(), (vec![0x100], 0, 0));
        assert_eq!(l.take_free(), (vec![], 0, 0), "released once");
    }

    #[test]
    fn a_list_adopted_again_keeps_its_recording() {
        let mut l = Ledger::default();
        l.adopted(0xA); // made
        l.bury(0x100);
        l.adopted(0xA); // its vtable swapped at Close: the same recording
        assert!(l.take_free().0.is_empty());
        l.gone(0xA);
        l.adopted(0xA); // a new list at the same address, made before 0x200 was out of use
        l.bury(0x200);
        assert_eq!(l.take_free(), (vec![0x100], 1, 1));
    }

    #[test]
    fn a_recording_that_goes_on_with_anothers_state_holds_what_that_one_holds() {
        let mut l = Ledger::default();
        l.began(0xA);
        l.bury(0x100);
        let from = l.began_at(0xA);
        // a split: segment B, made now, goes on with A's bound pipeline; then A is reset
        l.adopted(0xB);
        l.carries(0xB, from);
        l.began(0xA);
        assert_eq!(l.take_free(), (vec![], 1, 1), "B holds it");
        // B's own Reset after it ran, with nothing carried over, ends that
        l.began(0xB);
        assert_eq!(l.take_free().0, vec![0x100]);
        // carried from a list not known: holds everything
        l.bury(0x200);
        l.began(0xC);
        l.carries(0xC, l.began_at(0xD));
        assert!(l.take_free().0.is_empty());
    }

    #[test]
    fn a_released_list_holds_nothing() {
        let mut l = Ledger::default();
        l.began(0xA);
        l.bury(0x100);
        l.gone(0xA);
        assert_eq!(l.take_free().0, vec![0x100]);
    }

    #[test]
    fn a_list_first_seen_at_a_submit_holds_every_pipeline() {
        let mut l = Ledger::default();
        l.bury(0x100);
        l.seen(0xC); // its recording may be from any time
        l.bury(0x200);
        assert_eq!(l.take_free(), (vec![], 2, 1));
        assert_eq!(l.unknown(), 1);
        l.seen(0xC); // seen again: still not known
        assert!(l.take_free().0.is_empty());
        l.began(0xC);
        assert_eq!(l.take_free().0, vec![0x100, 0x200]);
        assert_eq!(l.unknown(), 0);
    }

    #[test]
    fn a_known_list_seen_at_its_submit_keeps_its_beginning() {
        let mut l = Ledger::default();
        l.bury(0x100);
        l.began(0xA);
        l.seen(0xA);
        assert_eq!(l.take_free().0, vec![0x100], "A began after the pipeline was out of use");
    }

    #[test]
    fn each_pipeline_waits_only_for_the_lists_that_began_before_it() {
        let mut l = Ledger::default();
        l.began(0xA);
        l.bury(0x100);
        l.began(0xB);
        l.bury(0x200);
        l.began(0xA);
        // B began between the two: it holds 0x200, not 0x100
        assert_eq!(l.take_free(), (vec![0x100], 1, 1));
        l.gone(0xB);
        assert_eq!(l.take_free(), (vec![0x200], 0, 0));
    }
}
