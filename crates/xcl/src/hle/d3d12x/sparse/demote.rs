//! Window tiles under the title's textures, moved to system memory (`XCL_DEMOTE_UNDER_TARGETS`).
//!
//! A render target, depth buffer, UAV texture or sampled texture the title places inside a sparse
//! window gets memory of its own (`vram::place` / `vram::place_pages`), and the texture uploads
//! read the title's CPU memory, not the window: the window's video-memory tile under it is a dead
//! copy. A retail title's GPU heaps in window 0 held 3.85 GB of such tiles on a 10 GB card, past
//! its budget. Those tiles go to system-memory pool heaps (`Demotion`): same bytes, same addresses
//! (what still reaches them by a raw address or a V# reads and writes them, over the bus), and
//! their video tile is freed behind the title queues as any freed tile is (`quarantine_tiles`).
//!
//! The move waits for the title queues to be past the work submitted before the texture was placed
//! (a mark on each, polled on the CPU: no GPU wait on a title queue, which may itself wait for
//! work the title submits later), and a tile a recorded GPU write names meanwhile is left where it
//! is. Then, on the sparse queue: the tile's 64 KiB copied into its system-memory tile, the window
//! remapped onto it. A placed buffer over a moved tile moves it back (`promote_range`); GPU writes
//! there stay in system memory and are counted.

use super::*;

/// `TileEntry::pool` heap numbers of system-memory pool heaps (`Demotion::heaps`) have this bit:
/// never a video pool's (`State::pools`), nor `SHADOW_POOL` or `LAZY`
pub(super) const SYS_POOL: u32 = 0x4000_0000;

/// Whether pool tile `tile` lies in a system-memory pool heap
#[inline]
pub(super) fn is_sys(tile: (u32, u32)) -> bool {
    tile.0 != SHADOW_POOL && tile.0 & SYS_POOL != 0
}

/// so far: tiles moved to system memory, moved back, recorded GPU writes that named a moved tile
static DEMOTED: AtomicU64 = AtomicU64::new(0);
static PROMOTED: AtomicU64 = AtomicU64::new(0);
static WRITES: AtomicU64 = AtomicU64::new(0);
/// system-memory tiles in use now (all states)
static SYS_TILES: AtomicU64 = AtomicU64::new(0);
/// a tile was asked to move: recorded GPU writes are asked for (`gpu_write`) from here on
static WATCHING: AtomicBool = AtomicBool::new(false);

/// how long after a system-memory heap could not be made no other is tried
const SYS_RETRY: std::time::Duration = std::time::Duration::from_secs(5);

/// Tiles asked to move (`State::demote_range`), and the title queues' marks they wait for
pub(super) struct Batch {
    tiles: Vec<(usize, (u32, u32))>,
    marks: Vec<(usize, ID3D12Fence, u64)>,
}

/// `State::demote`: the system-memory pool and the moves waiting
#[derive(Default)]
pub(super) struct Demotion {
    /// system-memory pool heaps (`POOL_TILES` tiles each) and a buffer placed over each, the copy
    /// destination of a move (None: given back, its index free)
    heaps: Vec<Option<(ID3D12Heap, ID3D12Resource)>>,
    /// tiles of `heaps` to use, and how many of each heap's
    free: Vec<(u32, u32)>,
    free_of: Vec<u32>,
    /// moves waiting for their marks, oldest first, and their tiles
    waiting: VecDeque<Batch>,
    queued: FxHashSet<usize>,
    /// buffers over video tiles a move back copied into, held until the sparse fence passes the value
    held: Vec<(u64, ID3D12Resource)>,
    /// when no system-memory heap could be made last
    failed: Option<std::time::Instant>,
    /// recorded GPU writes that named a moved tile (this state's; `WRITES` is every state's)
    writes: u64,
}

thread_local! {
    /// `XCL_DEMOTE_UNDER_TARGETS` for a test on this thread (None: as the environment says)
    pub(super) static DEMOTE_FOR_TEST: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

/// `XCL_DEMOTE_UNDER_TARGETS` (`v`): on unless `0`, `off` or `false`
fn switch(v: Option<&str>) -> bool {
    !v.map(str::trim).is_some_and(|v| v == "0" || v.eq_ignore_ascii_case("off") || v.eq_ignore_ascii_case("false"))
}

fn demote_asked() -> bool {
    if let Some(on) = DEMOTE_FOR_TEST.with(|c| c.get()) {
        return on;
    }
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = switch(std::env::var("XCL_DEMOTE_UNDER_TARGETS").ok().as_deref());
        if !on {
            crate::xlog_info!("sparse windows: tiles under textures stay in video memory (XCL_DEMOTE_UNDER_TARGETS=0)");
        }
        on
    })
}

/// Whether recorded GPU writes are to be asked for (`gpu_write`): a tile was asked to move
#[inline]
pub(super) fn watching() -> bool {
    WATCHING.load(Ordering::Relaxed)
}

/// A texture of the title now lies at `[va, va + len)` with memory of its own: the windows' tiles
/// wholly inside it go to system memory once the title queues are past what was submitted before
/// (`State::demote_range`). Returns the tiles asked to move. Nothing outside a sparse window, or
/// with `XCL_DEMOTE_UNDER_TARGETS=0`.
pub unsafe fn demote(va: usize, len: usize) -> usize {
    if va == 0 || len < TILE || !demote_asked() || sparse_window_of(va).is_none() {
        return 0;
    }
    let mut g = lock_state();
    let _held = holding(Holder::Buffers);
    g.as_mut().map_or(0, |s| s.demote_range(va, len))
}

/// The local video memory the moved tiles hold no more (bytes)
pub fn saved_bytes() -> u64 {
    SYS_TILES.load(Ordering::Relaxed) * TILE as u64
}

/// The demotion line (host's memory lines, every ~10 s): None while nothing moved
pub fn line() -> Option<String> {
    let (d, p, w) = (DEMOTED.load(Ordering::Relaxed), PROMOTED.load(Ordering::Relaxed), WRITES.load(Ordering::Relaxed));
    (d > 0).then(|| format!("sparse windows: {d} tiles demoted to system memory ({} MB there now), {p} promoted back, {w} GPU writes to demoted tiles", saved_bytes() >> 20))
}

fn sys_heap_properties() -> D3D12_HEAP_PROPERTIES {
    D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_CUSTOM, CPUPageProperty: D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE, MemoryPoolPreference: D3D12_MEMORY_POOL_L0, ..Default::default() }
}

impl State {
    /// Whether window tile `t` (pool tile `pool`) may move: the window's own (not over a CPU view),
    /// with video memory, held by no placed buffer
    fn demotable(&self, t: usize, pool: (u32, u32)) -> bool {
        pool != LAZY && pool.0 != SHADOW_POOL && !is_sys(pool) && !self.pinned.contains_key(&pool) && self.window_index(t).is_some() && self.mapped.get(&t).is_some_and(|e| e.src == 0 && e.pool == pool)
    }

    /// The tiles wholly inside `[va, va + len)` that may move (`demotable`) wait for the title
    /// queues' marks made now; those ready move (`demote_ready`). Returns the tiles asked.
    pub(super) unsafe fn demote_range(&mut self, va: usize, len: usize) -> usize {
        if !demote_asked() {
            return 0;
        }
        let first = va.saturating_add(TILE - 1) & !(TILE - 1);
        let end = va.saturating_add(len) & !(TILE - 1);
        let mut tiles = Vec::new();
        let mut t = first;
        while t < end {
            if let Some(e) = self.mapped.get(&t) {
                if !self.demote.queued.contains(&t) && self.demotable(t, e.pool) {
                    tiles.push((t, e.pool));
                }
            }
            t += TILE;
        }
        let n = tiles.len();
        if n > 0 {
            if !WATCHING.swap(true, Ordering::Relaxed) {
                crate::xlog_info!("sparse windows: window tiles under the title's textures go to system memory (XCL_DEMOTE_UNDER_TARGETS=0: they stay)");
            }
            // (a thread that asked for a GPU write lately asks again: `write_asked_lately`)
            LAZY_GEN.fetch_add(1, Ordering::Relaxed);
            self.demote.queued.extend(tiles.iter().map(|x| x.0));
            let marks = self.stamp_title_queues();
            self.demote.waiting.push_back(Batch { tiles, marks });
        }
        self.demote_ready();
        n
    }

    /// The moves whose marks are reached go, together (`demote_now`), but tiles that changed
    /// meanwhile (decommitted, given to a buffer, written: `demote_written`).
    pub(super) unsafe fn demote_ready(&mut self) {
        if !self.demote.held.is_empty() {
            let done = super::super::real::fence_completed(&self.fence);
            self.demote.held.retain(|(v, _)| *v > done);
        }
        let mut ready = Vec::new();
        while self.demote.waiting.front().is_some_and(|b| marks_done(&b.marks, |f| super::super::real::fence_completed(f))) {
            ready.extend(self.demote.waiting.pop_front().expect("a batch").tiles);
        }
        if ready.is_empty() {
            return;
        }
        for &(t, _) in &ready {
            self.demote.queued.remove(&t);
        }
        ready.retain(|&(t, pool)| self.demotable(t, pool));
        if !ready.is_empty() {
            self.demote_now(ready);
        }
    }

    /// `tiles` (window tile, its video tile) moved to system-memory tiles: copied there and mapped
    /// on the sparse queue, their video tiles freed behind the title queues. As many as there is
    /// system memory for; returns those.
    unsafe fn demote_now(&mut self, tiles: Vec<(usize, (u32, u32))>) -> usize {
        let mut moves = Vec::with_capacity(tiles.len());
        for (t, old) in tiles {
            let Some(sys) = self.take_sys_tile() else { break };
            moves.push((t, old, sys));
        }
        if moves.is_empty() {
            return 0;
        }
        // (every mapping sent before the copies read the window)
        self.flush_pending();
        let idx = self.begin_slot();
        {
            let list = &self.slots[idx].list;
            for &(t, _, (h, i)) in &moves {
                let wi = self.window_index(t).expect("a window tile");
                let (b, r) = (self.windows[wi].0, &self.windows[wi].1);
                let buf = &self.demote.heaps[(h & !SYS_POOL) as usize].as_ref().expect("a live system-memory pool").1;
                list.CopyBufferRegion(buf, i as u64 * TILE as u64, r, (t - b) as u64, TILE as u64);
            }
        }
        self.submit_slot(idx);
        self.remap(moves.iter().map(|&(t, _, sys)| (t, sys)));
        self.quarantine_tiles(moves.iter().map(|m| m.1).collect());
        DEMOTED.fetch_add(moves.len() as u64, Ordering::Relaxed);
        moves.len()
    }

    /// Window tiles `moves` (tile, its new pool tile) mapped onto their new tiles (after the
    /// copies on the same queue), and listed with them
    unsafe fn remap(&mut self, moves: impl Iterator<Item = (usize, (u32, u32))>) {
        let mut by_window: FxHashMap<usize, Vec<(usize, (u32, u32))>> = FxHashMap::default();
        for (t, tile) in moves {
            if let Some(e) = self.mapped.get_mut(&t) {
                e.pool = tile;
            }
            if let Some(wi) = self.window_index(t) {
                by_window.entry(wi).or_default().push((t, tile));
            }
        }
        for (wi, ts) in by_window {
            let (b, r) = (self.windows[wi].0, self.windows[wi].1.clone());
            self.map_tiles(&r, b, &ts);
        }
        self.signal();
    }

    /// A tile of a system-memory pool heap (one made when none is free). None: none could be made
    /// (warned once; tried again `SYS_RETRY` later)
    unsafe fn take_sys_tile(&mut self) -> Option<(u32, u32)> {
        if let Some(t) = self.demote.free.pop() {
            self.demote.free_of[(t.0 & !SYS_POOL) as usize] -= 1;
            SYS_TILES.fetch_add(1, Ordering::Relaxed);
            return Some(t);
        }
        if self.demote.failed.is_some_and(|at| at.elapsed() < SYS_RETRY) {
            return None;
        }
        let bytes = POOL_TILES as u64 * TILE as u64;
        let desc = D3D12_HEAP_DESC {
            SizeInBytes: bytes,
            Properties: sys_heap_properties(),
            Alignment: 0,
            Flags: D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES | D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES,
        };
        let made = (|| -> windows::core::Result<(ID3D12Heap, ID3D12Resource)> {
            let mut heap: Option<ID3D12Heap> = None;
            super::super::title_textures::own_work(|| self.dev.CreateHeap(&desc, &mut heap))?;
            let heap = heap.ok_or_else(|| windows::core::Error::from(windows::Win32::Foundation::E_FAIL))?;
            let mut d = buffer_desc(bytes);
            d.Flags = D3D12_RESOURCE_FLAG_NONE;
            let mut buf: Option<ID3D12Resource> = None;
            super::super::title_textures::own_work(|| self.dev.CreatePlacedResource(&heap, 0, &d, D3D12_RESOURCE_STATE_COMMON, None, &mut buf))?;
            let buf = buf.ok_or_else(|| windows::core::Error::from(windows::Win32::Foundation::E_FAIL))?;
            Ok((heap, buf))
        })();
        let (heap, buf) = match made {
            Ok(m) => m,
            Err(e) => {
                static WARNED: AtomicBool = AtomicBool::new(false);
                if !WARNED.swap(true, Ordering::Relaxed) {
                    crate::xlog_warn!("sparse windows: no system memory for tiles under textures ({e}): they stay in video memory (warned once)");
                }
                self.demote.failed = Some(std::time::Instant::now());
                return None;
            }
        };
        let _ = heap.SetName(windows::core::w!("xcl sparse tile pool (system memory)"));
        let _ = buf.SetName(windows::core::w!("xcl sparse tile pool (system memory) copy buffer"));
        let h = match self.demote.heaps.iter().position(|p| p.is_none()) {
            Some(i) => i,
            None => {
                self.demote.heaps.push(None);
                self.demote.free_of.push(0);
                self.demote.heaps.len() - 1
            }
        };
        self.demote.heaps[h] = Some((heap, buf));
        let hk = SYS_POOL | h as u32;
        self.demote.free.extend((1..POOL_TILES).rev().map(|i| (hk, i)));
        self.demote.free_of[h] = POOL_TILES - 1;
        SYS_TILES.fetch_add(1, Ordering::Relaxed);
        Some((hk, 0))
    }

    /// A system-memory tile no longer used (past the queues: `drain_quarantined`) is free; a heap
    /// all of whose tiles are goes.
    pub(super) fn sys_tile_free(&mut self, tile: (u32, u32)) {
        let h = (tile.0 & !SYS_POOL) as usize;
        if self.demote.heaps.get(h).is_none_or(|p| p.is_none()) {
            return;
        }
        self.demote.free.push(tile);
        self.demote.free_of[h] += 1;
        let _ = SYS_TILES.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |n| Some(n.saturating_sub(1)));
        if self.demote.free_of[h] == POOL_TILES {
            self.demote.free.retain(|t| (t.0 & !SYS_POOL) as usize != h);
            self.demote.free_of[h] = 0;
            self.demote.heaps[h] = None;
        }
    }

    /// The heap of a system-memory pool tile's heap number (`map_tiles`)
    pub(super) fn sys_heap(&self, h: u32) -> &ID3D12Heap {
        &self.demote.heaps[(h & !SYS_POOL) as usize].as_ref().expect("a tile of a live system-memory pool").0
    }

    /// System-memory tiles in use now (this state's)
    pub(super) fn sys_in_use(&self) -> usize {
        self.demote.heaps.iter().filter(|p| p.is_some()).count() * POOL_TILES as usize - self.demote.free.len()
    }

    /// A buffer is placed over `[a, a + n)`: its tiles waiting to move stay, and those moved come
    /// back to video memory (copied into a video tile, mapped there on the sparse queue, the
    /// system-memory tile freed behind the title queues), so that the buffer maps video memory.
    /// One a buffer already holds stays (both have to show the same bytes), as do all once no
    /// video tile is to be had. Returns the tiles moved back.
    pub(super) unsafe fn promote_range(&mut self, a: usize, n: usize) -> usize {
        let first = a & !(TILE - 1);
        let end = a.saturating_add(n);
        if !self.demote.queued.is_empty() {
            let queued = &mut self.demote.queued;
            for b in self.demote.waiting.iter_mut() {
                b.tiles.retain(|&(t, _)| {
                    let inside = t >= first && t < end;
                    if inside {
                        queued.remove(&t);
                    }
                    !inside
                });
            }
        }
        if self.sys_in_use() == 0 {
            return 0;
        }
        let mut moves: Vec<(usize, (u32, u32), (u32, u32), ID3D12Resource)> = Vec::new();
        let mut t = first;
        while t < end {
            let sys = self.mapped.get(&t).filter(|e| e.src == 0 && is_sys(e.pool) && !self.pinned.contains_key(&e.pool)).map(|e| e.pool);
            if let (Some(sys), Some(_)) = (sys, self.window_index(t)) {
                let Some((v, _)) = self.take_tile() else { break };
                // (a buffer over the video tile to copy into: the copy lands before the window shows it)
                let Some(heap) = self.pools.get(v.0 as usize).and_then(|p| p.clone()) else {
                    self.free.push(v);
                    break;
                };
                let mut d = buffer_desc(TILE as u64);
                d.Flags = D3D12_RESOURCE_FLAG_NONE;
                let mut buf: Option<ID3D12Resource> = None;
                let made = super::super::title_textures::own_work(|| self.dev.CreatePlacedResource(&heap, v.1 as u64 * TILE as u64, &d, D3D12_RESOURCE_STATE_COMMON, None, &mut buf));
                match (made, buf) {
                    (Ok(()), Some(buf)) => moves.push((t, sys, v, buf)),
                    _ => {
                        self.free.push(v);
                        break;
                    }
                }
            }
            t += TILE;
        }
        if moves.is_empty() {
            return 0;
        }
        self.flush_pending();
        let idx = self.begin_slot();
        {
            let list = &self.slots[idx].list;
            for (t, _, _, buf) in &moves {
                let wi = self.window_index(*t).expect("a window tile");
                let (b, r) = (self.windows[wi].0, &self.windows[wi].1);
                list.CopyBufferRegion(buf, 0, r, (*t - b) as u64, TILE as u64);
            }
        }
        self.submit_slot(idx);
        self.remap(moves.iter().map(|m| (m.0, m.2)));
        let v = self.value;
        let mut freed = Vec::with_capacity(moves.len());
        for (_, sys, _, buf) in moves {
            freed.push(sys);
            self.demote.held.push((v, buf));
        }
        let n = freed.len();
        self.quarantine_tiles(freed);
        PROMOTED.fetch_add(n as u64, Ordering::Relaxed);
        n
    }

    /// A recorded GPU write reaches `[a, a + n)` (`give_memory`, before the submit that runs it):
    /// its tiles waiting to move stay; those moved are written in system memory (counted).
    pub(super) fn demote_written(&mut self, a: usize, n: usize) {
        if self.demote.queued.is_empty() && self.sys_in_use() == 0 {
            return;
        }
        let first = a & !(TILE - 1);
        let end = a.saturating_add(n);
        let queued = &mut self.demote.queued;
        if !queued.is_empty() {
            for b in self.demote.waiting.iter_mut() {
                b.tiles.retain(|&(t, _)| {
                    let inside = t >= first && t < end;
                    if inside {
                        queued.remove(&t);
                    }
                    !inside
                });
            }
        }
        let mut written = 0u64;
        let mut t = first;
        while t < end {
            if self.mapped.get(&t).is_some_and(|e| is_sys(e.pool)) {
                written += 1;
            }
            match t.checked_add(TILE) {
                Some(next) => t = next,
                None => break,
            }
        }
        if written > 0 {
            self.demote.writes += written;
            WRITES.fetch_add(written, Ordering::Relaxed);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{settle_quarantined, test_state_with_reserved, window_bytes, write_tile};
    use super::*;
    use windows_sys::Win32::System::Memory::MEM_RELEASE;

    /// `XCL_DEMOTE_UNDER_TARGETS` for this thread while it lives
    struct Demote;
    impl Demote {
        fn on(on: bool) -> Demote {
            DEMOTE_FOR_TEST.with(|c| c.set(Some(on)));
            Demote
        }
    }
    impl Drop for Demote {
        fn drop(&mut self) {
            DEMOTE_FOR_TEST.with(|c| c.set(None));
        }
    }

    /// A state of `tiles` tiles all committed, each written with its own bytes and sent up on a
    /// title queue (noted: moves wait for it); the bytes of each tile.
    unsafe fn written_state(tiles: usize) -> (usize, State, ID3D12CommandQueue, Vec<Vec<u8>>) {
        let (base, mut s) = test_state_with_reserved(tiles);
        assert!(!VirtualAlloc(base as *const c_void, tiles * TILE, MEM_COMMIT, PAGE_READWRITE).is_null());
        s.commit_range(base, tiles * TILE);
        let queue: ID3D12CommandQueue = s.dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC::default()).expect("title queue");
        let mut dirty = Vec::new();
        for k in 0..tiles {
            dirty.extend(write_tile(base + k * TILE, 0xD0 + k as u64));
        }
        s.before_execute_on(&queue, &dirty, PAGE);
        s.wait_for(s.value);
        let bytes = (0..tiles).map(|k| std::slice::from_raw_parts((base + k * TILE) as *const u8, TILE).to_vec()).collect();
        (base, s, queue, bytes)
    }

    /// Until every move waiting went (5 s at most)
    unsafe fn settle_moves(s: &mut State) {
        let start = std::time::Instant::now();
        while !s.demote.waiting.is_empty() && start.elapsed() < std::time::Duration::from_secs(5) {
            s.demote_ready();
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
        assert!(s.demote.waiting.is_empty(), "the title queue got past its mark");
    }

    /// Video pool tile `tile` overwritten with zeros (through a buffer placed over it): a window
    /// tile still mapped there would read them
    unsafe fn clobber(s: &mut State, tile: (u32, u32)) {
        assert!(s.ensure_zero_buf());
        let heap = s.pools[tile.0 as usize].clone().expect("a live pool");
        let mut d = buffer_desc(TILE as u64);
        d.Flags = D3D12_RESOURCE_FLAG_NONE;
        let mut buf: Option<ID3D12Resource> = None;
        s.dev.CreatePlacedResource(&heap, tile.1 as u64 * TILE as u64, &d, D3D12_RESOURCE_STATE_COMMON, None, &mut buf).expect("a buffer over the tile");
        let buf = buf.expect("a buffer over the tile");
        let idx = s.begin_slot();
        s.slots[idx].list.CopyBufferRegion(&buf, 0, s.zero_buf.as_ref().expect("zeros"), 0, TILE as u64);
        s.submit_slot(idx);
        s.wait_for(s.value);
    }

    unsafe fn release(s: State, base: usize) {
        s.wait_for(s.value);
        drop(s);
        assert_ne!(VirtualFree(base as *mut c_void, 0, MEM_RELEASE), 0);
    }

    #[test]
    fn the_switch_is_on_unless_0() {
        assert!(switch(None));
        assert!(switch(Some("1")));
        assert!(!switch(Some("0")));
        assert!(!switch(Some(" off ")));
        assert!(!switch(Some("FALSE")));
        assert!(!is_sys(LAZY) && !is_sys((SHADOW_POOL, 3)) && !is_sys((7, 1)) && is_sys((SYS_POOL | 2, 5)));
    }

    /// Only the tiles wholly inside the range that no placed buffer holds move; their bytes
    /// stay what the window shows, and their video tiles are freed behind the title queues.
    #[test]
    fn demote_moves_only_the_free_tiles_under_the_range_and_keeps_their_bytes() {
        let _on = Demote::on(true);
        unsafe {
            let (base, mut s, _queue, bytes) = written_state(4);
            let pools: Vec<(u32, u32)> = (0..4).map(|k| s.mapped[&(base + k * TILE)].pool).collect();
            // (a placed buffer over tile 2)
            *s.pinned.entry(pools[2]).or_default() += 1;
            // tiles 0 and 3 only in part
            let asked = s.demote_range(base + TILE / 2, 3 * TILE);
            assert_eq!(asked, 1, "tile 1 alone: whole, held by no buffer");
            settle_moves(&mut s);
            assert!(is_sys(s.mapped[&(base + TILE)].pool), "tile 1 is in system memory");
            for k in [0, 2, 3] {
                assert_eq!(s.mapped[&(base + k * TILE)].pool, pools[k], "tile {k} stays");
            }
            assert_eq!(s.sys_in_use(), 1);
            assert!(s.quarantined.iter().any(|b| b.tiles.contains(&pools[1])), "the video tile is freed behind the queues");
            // (the old video tile holds other bytes now: the window shows the system tile)
            clobber(&mut s, pools[1]);
            for k in 0..4 {
                assert!(window_bytes(&mut s, base + k * TILE, TILE) == bytes[k], "tile {k} reads its bytes");
            }
            // asked again: the tiles under it in part do not move, the held and moved ones neither
            assert_eq!(s.demote_range(base, 4 * TILE), 2, "tiles 0 and 3 now");
            settle_moves(&mut s);
            assert_eq!(s.sys_in_use(), 3);
            assert!(window_bytes(&mut s, base, 4 * TILE) == bytes.concat(), "all of it reads its bytes");
            settle_quarantined(&mut s);
            assert!(s.cooling.iter().any(|c| c.0 == pools[1]) && !s.cooling.iter().any(|c| is_sys(c.0)), "a freed video tile cools; no system tile does");
            release(s, base);
        }
    }

    /// A move waits for the title queue's work submitted before it; a recorded GPU write to the
    /// tile meanwhile keeps it where it is.
    #[test]
    fn a_move_waits_for_the_title_queue_and_a_written_tile_stays() {
        let _on = Demote::on(true);
        unsafe {
            let (base, mut s, queue, bytes) = written_state(2);
            let gate: ID3D12Fence = s.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("gate");
            queue.Wait(&gate, 1).expect("gate the title queue");
            assert_eq!(s.demote_range(base, 2 * TILE), 2);
            std::thread::sleep(std::time::Duration::from_millis(20));
            s.demote_ready();
            assert_eq!(s.sys_in_use(), 0, "nothing moves while the title queue has work before it");
            s.demote_written(base + TILE + 16, 4);
            gate.Signal(1).expect("open the gate");
            settle_moves(&mut s);
            assert!(is_sys(s.mapped[&base].pool), "tile 0 moved");
            assert!(!is_sys(s.mapped[&(base + TILE)].pool), "tile 1 was written meanwhile: it stays");
            s.demote_written(base, 8);
            assert_eq!(s.demote.writes, 1, "a write to a moved tile is counted");
            assert!(window_bytes(&mut s, base, 2 * TILE) == bytes.concat());
            release(s, base);
        }
    }

    /// A buffer placed over a moved tile moves it back to video memory with its bytes; one
    /// waiting to move stays.
    #[test]
    fn a_placed_buffer_brings_its_tiles_back() {
        let _on = Demote::on(true);
        unsafe {
            let (base, mut s, _queue, bytes) = written_state(3);
            s.demote_range(base, 2 * TILE);
            settle_moves(&mut s);
            let gate: ID3D12Fence = s.dev.CreateFence(0, D3D12_FENCE_FLAG_NONE).expect("gate");
            _queue.Wait(&gate, 1).expect("gate the title queue");
            assert_eq!(s.demote_range(base + 2 * TILE, TILE), 1, "tile 2 waits");
            let sys = s.mapped[&base].pool;
            assert!(is_sys(sys) && is_sys(s.mapped[&(base + TILE)].pool));
            // the buffer over tiles 0 and 2 (`place_buffer`)
            assert_eq!(s.promote_range(base, TILE), 1);
            assert_eq!(s.promote_range(base + 2 * TILE, TILE), 0);
            assert!(!is_sys(s.mapped[&base].pool), "tile 0 is in video memory again");
            assert!(is_sys(s.mapped[&(base + TILE)].pool), "tile 1 stays in system memory");
            assert!(s.demote.waiting.iter().all(|b| b.tiles.is_empty()), "tile 2 waits no more");
            gate.Signal(1).expect("open the gate");
            settle_moves(&mut s);
            assert!(!is_sys(s.mapped[&(base + 2 * TILE)].pool), "tile 2 stayed");
            assert!(s.quarantined.iter().any(|b| b.tiles.contains(&sys)), "its system tile is freed behind the queues");
            assert!(window_bytes(&mut s, base, 3 * TILE) == bytes.concat(), "the bytes came back with it");
            settle_quarantined(&mut s);
            assert_eq!(s.sys_in_use(), 1);
            release(s, base);
        }
    }

    /// `XCL_DEMOTE_UNDER_TARGETS=0`: nothing moves.
    #[test]
    fn with_the_switch_off_nothing_moves() {
        let _off = Demote::on(false);
        unsafe {
            let (base, mut s, _queue, _bytes) = written_state(2);
            let pools: Vec<(u32, u32)> = (0..2).map(|k| s.mapped[&(base + k * TILE)].pool).collect();
            assert_eq!(s.demote_range(base, 2 * TILE), 0);
            s.demote_ready();
            assert!(s.demote.waiting.is_empty() && s.demote.heaps.is_empty());
            assert_eq!((0..2).map(|k| s.mapped[&(base + k * TILE)].pool).collect::<Vec<_>>(), pools);
            assert_eq!(demote(base, 2 * TILE), 0);
            release(s, base);
        }
    }
}
