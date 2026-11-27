use super::*;

thread_local! {
    /// `on()` for this thread (the tests record without `XCL_HOT_RESIDENCY`)
    pub(super) static ON_FOR_TEST: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct On;
impl On {
    fn new() -> On {
        ON_FOR_TEST.with(|c| c.set(true));
        On
    }
}
impl Drop for On {
    fn drop(&mut self) {
        ON_FOR_TEST.with(|c| c.set(false));
    }
}

/// (the frame's set is the process's: one test at a time takes it)
static FRAME_TEST: Mutex<()> = Mutex::new(());

const T: usize = TILE as usize;

fn tiles_of(keys: &FrameKeys) -> Vec<usize> {
    sorted(&keys.tiles)
}

fn sorted(t: &TileSet) -> Vec<usize> {
    let mut v: Vec<usize> = t.iter().collect();
    v.sort_unstable();
    v
}

#[test]
fn ranges_count_each_tile_they_touch_once() {
    let (k, cut) = va_keys(0x1_0000, 1);
    assert_eq!((k.collect::<Vec<_>>(), cut), (vec![1], false));
    // across a tile boundary: both
    let (k, cut) = va_keys(0x1_FFF0, 0x20);
    assert_eq!((k.collect::<Vec<_>>(), cut), (vec![1, 2], false));
    // an empty range counts its first tile
    assert_eq!(va_keys(0x3_0004, 0).0.collect::<Vec<_>>(), vec![3]);
    // a range past 256 MiB is cut there
    let (k, cut) = va_keys(0, 1 << 30);
    assert_eq!((k.count() as u64, cut), (RANGE_TILES_MAX, true));
    // (to the end of the address space: no overflow)
    assert!(va_keys(u64::MAX - 5, u64::MAX).1 || va_keys(u64::MAX - 5, u64::MAX).0.count() >= 1);
}

#[test]
fn keys_say_what_they_stand_for() {
    assert_eq!(decode(5), Key::Tile(5 * T));
    assert_eq!(decode(RES | 0x7FF6_1234_5670), Key::Res(0x7FF6_1234_5670));
    assert_eq!(decode(PSO | 0x1230), Key::Pso(0x1230));
    assert_eq!(decode(TABLE | 0x2_0000_0040), Key::Table(0x2_0000_0040));
}

/// A list's touches reach the frame's set when it is submitted, again at each submit of it, and
/// no more after its Reset; another list's do not come with them.
#[test]
fn a_submitted_list_s_touches_reach_the_frame_and_count_again() {
    let _g = FRAME_TEST.lock().unwrap_or_else(|e| e.into_inner());
    let _on = On::new();
    let (a, b) = (0x7E57_1000usize, 0x7E57_2000usize);
    let _ = take_frame();
    touch_va(a, 0x10_0000, 3 * TILE);
    touch_va(a, 0x10_0000, 1); // (the same tile again: once)
    touch_res(a, 0xABC0);
    touch_va(b, 0x90_0000, 1);
    // `a` again after `b` on this thread: `a`'s log is one log
    touch_va(a, 0x20_0000, 1);
    closed(a);
    submitted(&[a as *mut std::ffi::c_void]);
    let f = take_frame();
    assert_eq!(tiles_of(&f), vec![0x10_0000, 0x11_0000, 0x12_0000, 0x20_0000]);
    assert!(f.tagged.contains(&(RES | 0xABC0)));
    // submitted again without a Reset: the same again
    submitted(&[a as *mut std::ffi::c_void]);
    assert_eq!(tiles_of(&take_frame()), vec![0x10_0000, 0x11_0000, 0x12_0000, 0x20_0000]);
    // after its Reset: nothing; `b` still has its own
    reset(a);
    submitted(&[a as *mut std::ffi::c_void, b as *mut std::ffi::c_void]);
    assert_eq!(tiles_of(&take_frame()), vec![0x90_0000]);
    reset(b);
    // with the switch off nothing is recorded
    drop(_on);
    touch_va(a, 0x10_0000, 1);
    let _on = On::new();
    submitted(&[a as *mut std::ffi::c_void]);
    assert!(take_frame().is_empty());
}

/// Descriptor tables name the resources of their first slots; a console buffer is its tiles.
#[test]
fn tables_and_buffers_come_down_to_resources_and_tiles() {
    let table = 0x5_0000_0000usize;
    let keys = [TABLE | table as u64, RES | 0x900, 7, PSO | 0x40, PSO | 0x80];
    let resource_at = |at: usize| match at - table {
        0 => Some(0x100),
        64 => Some(0x200),
        x if x >= 32 * TABLE_SLOTS => panic!("slot {} looked at", x / 32),
        _ => None,
    };
    let console_buffer = |r: usize| (r == 0x900).then_some((0x40_0000usize, 2 * T + 1));
    let t = sort_keys(frame_keys(keys), batch(resource_at), batch_buffers(console_buffer));
    assert_eq!(t.res, vec![0x100, 0x200, 0x900]);
    assert_eq!(sorted(&t.tiles), vec![7 * T, 0x40_0000, 0x41_0000, 0x42_0000]);
    assert_eq!(t.psos, 2);
}

fn frame_keys(keys: impl IntoIterator<Item = u64>) -> FrameKeys {
    let mut f = FrameKeys::default();
    for k in keys {
        assert!(f.insert(k));
    }
    f
}

/// `resource_at` one address at a time, as `sort_keys` asks for them
fn batch(resource_at: impl Fn(usize) -> Option<usize>) -> impl FnOnce(&[usize], &mut dyn FnMut(usize)) {
    move |ats, f| ats.iter().filter_map(|&a| resource_at(a)).for_each(f)
}

fn batch_buffers(console_buffer: impl Fn(usize) -> Option<(usize, usize)>) -> impl FnOnce(&[usize], &mut dyn FnMut(usize, usize)) {
    move |res, f| res.iter().filter_map(|&r| console_buffer(r)).for_each(|(p, n)| f(p, n))
}

#[test]
fn an_address_is_in_a_heap_s_range_by_its_tile() {
    let tiles = TileSet::of(&[0x10_0000, 0x30_0000, 0x50_0000]);
    assert!(any_in(&tiles, 0x30_0000, T));
    // an arena from the middle of a touched tile
    assert!(any_in(&tiles, 0x30_8000, T));
    assert!(!any_in(&tiles, 0x31_0000, 0x1F_0000));
    assert!(any_in(&tiles, 0x31_0000, 0x20_0000));
    assert!(!any_in(&TileSet::default(), 0, usize::MAX / 2));
    // across a GiB's bitmap, and at its edges
    let edge = TileSet::of(&[(1 << 30) - T, 1 << 30, 5 << 30]);
    assert!(any_in(&edge, (1 << 30) - T, T) && any_in(&edge, 1 << 30, 1) && !any_in(&edge, (1 << 30) + T, (4 << 30) - T));
    assert!(any_in(&edge, (1 << 30) + T, 4 << 30) && any_in(&edge, 0, usize::MAX));
}

fn view(ptr: usize, touched: bool) -> HeapView {
    HeapView { ptr, bytes: 4 << 20, touched, pinned: false, empty: false }
}

/// HIGH while touched in the last 2 s, NORMAL then, LOW only after 4 s untouched, HIGH again at
/// a touch; a class is given once (no call for a heap whose class stays).
#[test]
fn a_heap_cools_down_slowly_and_heats_up_at_once() {
    let t0 = Instant::now();
    let at = |ms: u64| t0 + Duration::from_millis(ms);
    let mut tr = Tracker::default();
    let step = |tr: &mut Tracker, v: HeapView, now| {
        tr.begin();
        let c = tr.see(&v, now);
        tr.end();
        c
    };
    assert_eq!(step(&mut tr, view(1, true), at(0)), Some(Class::High));
    assert_eq!(step(&mut tr, view(1, false), at(1500)), None, "still HIGH: given once");
    assert_eq!(tr.tally.hot_bytes, 4 << 20, "in the hot set");
    assert_eq!(step(&mut tr, view(1, false), at(2500)), Some(Class::Normal));
    assert_eq!(tr.tally.hot_bytes, 0);
    assert_eq!(step(&mut tr, view(1, false), at(3900)), None, "not LOW before 4 s");
    assert_eq!(step(&mut tr, view(1, false), at(4000)), Some(Class::Low));
    assert_eq!(step(&mut tr, view(1, false), at(60_000)), None);
    assert_eq!(tr.tally.bytes[Class::Low as usize], 4 << 20);
    assert_eq!(step(&mut tr, view(1, true), at(60_100)), Some(Class::High), "a touch: HIGH at once");
    // a heap first seen untouched counts from then (made, or there before the switch)
    assert_eq!(step(&mut tr, view(2, false), at(70_000)), Some(Class::High));
    assert_eq!(step(&mut tr, view(2, false), at(74_000)), Some(Class::Low));
}

#[test]
fn empty_and_pinned_heaps_have_their_own_classes() {
    let t0 = Instant::now();
    let mut tr = Tracker::default();
    tr.begin();
    let pinned = HeapView { pinned: true, ..view(1, false) };
    let empty = HeapView { empty: true, ..view(2, false) };
    assert_eq!(tr.see(&pinned, t0), Some(Class::High));
    assert_eq!(tr.see(&empty, t0), Some(Class::Minimum));
    tr.end();
    let later = t0 + Duration::from_secs(30);
    tr.begin();
    assert_eq!(tr.see(&pinned, later), None, "a GPU-owned target's heap stays HIGH untouched");
    assert_eq!(tr.see(&empty, later), None);
    tr.end();
    tr.begin();
    assert_eq!(tr.see(&pinned, later), None);
    // an empty heap in use again is new work: HIGH, not LOW for its idle time
    assert_eq!(tr.see(&view(2, false), later), Some(Class::High));
    tr.end();
    assert_eq!(tr.tally.heaps, [0, 0, 0, 2]);
    assert_eq!(classify(Duration::ZERO, true, true), Class::Minimum, "nothing in it: nothing to keep");
}

/// A heap not seen in an update is gone; one made at a known address starts anew.
#[test]
fn heaps_gone_or_made_again_are_forgotten() {
    let t0 = Instant::now();
    let mut tr = Tracker::default();
    tr.begin();
    assert_eq!(tr.see(&view(1, true), t0), Some(Class::High));
    assert_eq!(tr.see(&view(2, true), t0), Some(Class::High));
    tr.end();
    tr.begin();
    assert_eq!(tr.see(&view(1, false), t0), None);
    tr.end();
    assert!(!tr.heaps.contains_key(&2), "not seen: gone");
    tr.forget(1);
    tr.begin();
    assert_eq!(tr.see(&view(1, false), t0), Some(Class::High), "made again: given its class again");
    tr.end();
}

#[test]
fn the_line_says_what_fits() {
    let mut t = Tally::default();
    t.bytes[Class::High as usize] = 3000 << 20;
    t.heaps[Class::High as usize] = 700;
    t.bytes[Class::Low as usize] = 5000 << 20;
    t.heaps[Class::Low as usize] = 1200;
    t.bytes[Class::Minimum as usize] = 64 << 20;
    t.heaps[Class::Minimum as usize] = 1;
    t.hot_bytes = 2500 << 20;
    let s = line(&t, 32 << 20, 9000 << 20, 1024 << 20, 14000 << 20);
    assert!(s.starts_with("hot residency: HIGH 3000 MB (700 heaps), LOW 5000 MB (1200 heaps), MIN 64 MB; visible hot set 2500 MB vs target 7976 MB (budget 9000 − headroom 1024); local 14000 MB"), "{s}");
}

#[test]
fn the_switch_is_off_unless_asked() {
    assert!(!parse_on(None));
    assert!(!parse_on(Some("0")));
    assert!(parse_on(Some(" 1 ")));
    assert!(parse_on(Some("on")));
}

/// What recording costs on the recording thread (`cargo test -- --nocapture` prints it): a
/// draw's worth of touches (vertex and index buffers, root views – one of 4 MiB –, a texture, a
/// pipeline, a table) on lists recorded in turn.
#[test]
fn recording_is_cheap() {
    let _g = FRAME_TEST.lock().unwrap_or_else(|e| e.into_inner());
    let _on = On::new();
    let lists = [0x7E58_0000usize, 0x7E58_1000, 0x7E58_2000, 0x7E58_3000];
    let draws = 50_000u64;
    let t = Instant::now();
    for d in 0..draws {
        let list = lists[(d / 1000) as usize % lists.len()];
        let at = 0x1_0000_0000 + (d % 512) * TILE;
        touch_va(list, at, 3 * TILE);
        touch_va(list, at + 0x100_0000, 1);
        // (root views a pipeline may store through: 4 MiB each, of a few buffers)
        touch_va(list, 0x2_0000_0000 + (d % 8) * 0x40_0000, ROOT_VIEW_WRITE_BYTES);
        touch_va(list, at + 0x300_0000, ROOT_VIEW_BYTES);
        touch_res(list, 0x5000 + (d % 64) as usize * 16);
        touch_pso(list, 0x9000 + (d % 32) as usize * 16);
        touch_table(list, 0x2_0000_0000 + (d % 128) as usize * 32);
    }
    let per_draw = t.elapsed().as_nanos() as u64 / draws;
    let t = Instant::now();
    submitted(&lists.map(|l| l as *mut std::ffi::c_void));
    let merge = t.elapsed();
    let f = take_frame();
    for l in lists {
        reset(l);
    }
    println!("hot residency recording: {per_draw} ns a draw (7 touches, up to 70 keys); a submit of {} lists: {} µs for {} keys", lists.len(), merge.as_micros(), f.len());
    assert!(f.len() >= 512 && f.len() < 4 * 512 + 512 + 8 * 64, "{} keys", f.len());
}

// ---------------------------------------------------------------------------------------
// the update as it was (a sorted vector of tiles, each looked up): what the bitmap and the
// walks must decide the same as
// ---------------------------------------------------------------------------------------

/// (before) `keys` sorted out: tiles sorted, resources sorted, pipelines
fn old_sort_keys(keys: &FxSet<u64>, resource_at: impl Fn(usize) -> Option<usize>, console_buffer: impl Fn(usize) -> Option<(usize, usize)>) -> (Vec<usize>, Vec<usize>, usize) {
    let (mut tiles, mut res, mut psos) = (OldFxSet::default(), OldFxSet::default(), 0);
    for &k in keys {
        match decode(k) {
            Key::Tile(a) => {
                tiles.insert(a);
            }
            Key::Res(r) => {
                res.insert(r);
            }
            Key::Pso(_) => psos += 1,
            Key::Table(at) => res.extend((0..TABLE_SLOTS).filter_map(|s| resource_at(at + 32 * s))),
        }
    }
    for &r in &res {
        if let Some((p, n)) = console_buffer(r) {
            let (keys, _) = va_keys(p as u64, n.min(RES_TILES_MAX * TILE as usize) as u64);
            tiles.extend(keys.map(|k| k as usize * T));
        }
    }
    let mut tiles: Vec<usize> = tiles.into_iter().collect();
    tiles.sort_unstable();
    let mut res: Vec<usize> = res.into_iter().collect();
    res.sort_unstable();
    (tiles, res, psos)
}

/// (before) `Fx` without its last rotation
#[derive(Default)]
struct OldFx(u64);
impl Hasher for OldFx {
    fn write(&mut self, _: &[u8]) {
        unreachable!()
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
type OldFxSet<K> = HashSet<K, BuildHasherDefault<OldFx>>;

fn old_any_in(tiles: &[usize], base: usize, size: usize) -> bool {
    let i = tiles.partition_point(|&a| a < base & !(T - 1));
    tiles.get(i).is_some_and(|&a| a < base + size)
}

/// a small deterministic generator
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
}

/// Heaps as the owners have them: window tiles on sparse pool heaps, pages on target pool
/// heaps, arenas over address ranges; resources of console buffers and table slots.
struct Model {
    sparse: FxMap<usize, u32>,
    pages: HashMap<usize, u32>,
    sparse_pools: u32,
    page_pools: u32,
    arenas: Vec<(usize, usize)>,
    /// (resource to console buffer) and (slot to resource)
    buffers: HashMap<usize, (usize, usize)>,
    slots: HashMap<usize, usize>,
}

const WIN: usize = 0x2_0000_0000;
/// (64 GiB of tiles: room for the mappings Model::new is asked for)
const WIN_TILES: u64 = 1 << 20;

impl Model {
    fn new(r: &mut Lcg, mapped: usize, pages: usize) -> Model {
        let mut m = Model {
            sparse: FxMap::default(),
            pages: HashMap::new(),
            sparse_pools: (mapped / 64 + 1) as u32,
            page_pools: (pages / 1024 + 1) as u32,
            arenas: Vec::new(),
            buffers: HashMap::new(),
            slots: HashMap::new(),
        };
        while m.sparse.len() < mapped {
            m.sparse.insert(WIN + r.below(WIN_TILES) as usize * T, r.below(m.sparse_pools as u64) as u32);
        }
        // (target pages at their own addresses, some in the windows)
        while m.pages.len() < pages {
            let base = if r.below(4) == 0 { WIN } else { 0x10_0000_0000 };
            m.pages.insert(base + r.below(WIN_TILES) as usize * T, r.below(m.page_pools as u64) as u32);
        }
        for _ in 0..200 {
            let base = WIN + r.below(WIN_TILES) as usize * T + if r.below(2) == 0 { 0x8000 } else { 0 };
            m.arenas.push((base, (1 + r.below(64)) as usize * T));
        }
        for i in 0..400usize {
            let bytes = if i % 20 == 0 { 256 << 20 } else { r.below(8 << 20) as usize };
            m.buffers.insert(0xB000_0000 + i * 16, (WIN + r.below(WIN_TILES - 8192) as usize * T + r.below(TILE) as usize, bytes));
        }
        for i in 0..4000usize {
            if r.below(3) != 0 {
                let res = if r.below(2) == 0 { 0xB000_0000 + r.below(400) as usize * 16 } else { 0xC000_0000 + r.below(1000) as usize * 16 };
                m.slots.insert(0x3_0000_0000 + i * 32, res);
            }
        }
        m
    }

    /// some tiles mapped elsewhere, unmapped or mapped anew (between updates)
    fn churn(&mut self, r: &mut Lcg) {
        let keys: Vec<usize> = self.sparse.keys().copied().take(64).collect();
        for k in keys {
            match r.below(3) {
                0 => {
                    self.sparse.remove(&k);
                }
                1 => {
                    self.sparse.insert(k, r.below(self.sparse_pools as u64) as u32);
                }
                _ => {
                    self.sparse.insert(WIN + r.below(WIN_TILES) as usize * T, r.below(self.sparse_pools as u64) as u32);
                }
            }
        }
    }

    /// a frame's keys: tiles of draws and copies, textures, buffers, pipelines, tables
    fn keys(&self, r: &mut Lcg, n: usize) -> Vec<u64> {
        let mut k = Vec::new();
        for _ in 0..n {
            match r.below(10) {
                0..=5 => k.push((WIN / T) as u64 + r.below(WIN_TILES)),
                6 => k.push(RES | (0xB000_0000 + r.below(400) * 16)),
                7 => k.push(RES | (0xC000_0000 + r.below(1000) * 16)),
                8 => k.push(PSO | (0x9000 + r.below(512) * 16)),
                _ => k.push(TABLE | (0x3_0000_0000 + r.below(4000) * 32)),
            }
        }
        k
    }

    /// The views of the heaps at an update as the owners make them: `mark_sparse` and
    /// `mark_pages` give the pool of each touched tile of the map, `arena_touched` an arena's.
    fn views(&self, mark_sparse: impl FnOnce(&mut dyn FnMut(u32)), mark_pages: impl FnOnce(&mut dyn FnMut(u32)), arena_touched: impl Fn(usize, usize) -> bool) -> Vec<HeapView> {
        let mut st = vec![false; self.sparse_pools as usize];
        let mut used = vec![false; self.sparse_pools as usize];
        self.sparse.values().for_each(|&p| used[p as usize] = true);
        mark_sparse(&mut |p| st[p as usize] = true);
        let mut pt = vec![false; self.page_pools as usize];
        mark_pages(&mut |p| pt[p as usize] = true);
        let mut v: Vec<HeapView> = (0..self.sparse_pools as usize).map(|h| HeapView { ptr: 0x1000 + h, bytes: 4 << 20, touched: st[h], pinned: false, empty: !used[h] }).collect();
        v.extend((0..self.page_pools as usize).map(|h| HeapView { ptr: 0x10_0000 + h, bytes: 64 << 20, touched: pt[h], pinned: h % 7 == 0, empty: false }));
        v.extend(self.arenas.iter().enumerate().map(|(i, &(b, n))| HeapView { ptr: 0x20_0000 + i, bytes: n as u64, touched: arena_touched(b, n), pinned: false, empty: i % 13 == 0 }));
        v
    }
}

/// For a recorded run of updates (the frames' keys, the mappings changing between them) the
/// bitmap, the batched lookups and the walks give every heap the class the sorted tiles did,
/// at the same updates.
#[test]
fn the_classes_are_the_ones_the_sorted_tiles_gave() {
    let mut r = Lcg(7);
    // (maps larger and maps smaller than the tiles touched: both ways of `for_each_touched`)
    for (mapped, pages, keys_a_frame) in [(30_000, 20_000, 3000), (2000, 500, 20_000)] {
        let mut m = Model::new(&mut r, mapped, pages);
        let (mut old, mut new) = (Tracker::default(), Tracker::default());
        let t0 = Instant::now();
        let mut changes = 0;
        for u in 0..40u64 {
            // (some updates with little or nothing touched: heaps cool down)
            let n = if (12..24).contains(&u) { 0 } else if u % 5 == 4 { keys_a_frame / 50 } else { keys_a_frame };
            let keys = m.keys(&mut r, n);
            let now = t0 + Duration::from_millis(500 * u + r.below(40));
            let set: FxSet<u64> = keys.iter().copied().collect();
            let (old_tiles, old_res, old_psos) = old_sort_keys(&set, |a| m.slots.get(&a).copied(), |res| m.buffers.get(&res).copied());
            let mut f = FrameKeys::default();
            keys.iter().for_each(|&k| assert!(f.insert(k)));
            let t = sort_keys(f, batch(|a| m.slots.get(&a).copied()), batch_buffers(|res| m.buffers.get(&res).copied()));
            assert_eq!(sorted(&t.tiles), old_tiles);
            assert_eq!((&t.res, t.psos), (&old_res, old_psos));
            let old_views = m.views(
                |mark| old_tiles.iter().filter_map(|a| m.sparse.get(a)).for_each(|&p| mark(p)),
                |mark| old_tiles.iter().filter_map(|a| m.pages.get(a)).for_each(|&p| mark(p)),
                |b, n| old_any_in(&old_tiles, b, n),
            );
            let new_views = m.views(|mark| for_each_touched(&t.tiles, &m.sparse, |&p| mark(p)), |mark| for_each_touched(&t.tiles, &m.pages, |&p| mark(p)), |b, n| any_in(&t.tiles, b, n));
            for (a, b) in old_views.iter().zip(&new_views) {
                assert_eq!((a.ptr, a.touched, a.pinned, a.empty), (b.ptr, b.touched, b.pinned, b.empty), "update {u}");
            }
            old.begin();
            new.begin();
            let a: Vec<Option<Class>> = old_views.iter().map(|v| old.see(v, now)).collect();
            let b: Vec<Option<Class>> = new_views.iter().map(|v| new.see(v, now)).collect();
            old.end();
            new.end();
            assert_eq!(a, b, "update {u}: the classes given");
            assert_eq!((old.tally.bytes, old.tally.heaps, old.tally.hot_bytes), (new.tally.bytes, new.tally.heaps, new.tally.hot_bytes));
            changes += a.iter().flatten().count();
            m.churn(&mut r);
        }
        assert!(changes > 100, "the run changes classes ({changes})");
    }
}

#[test]
fn a_tile_set_is_its_tiles() {
    let mut r = Lcg(3);
    let mut t = TileSet::default();
    let mut s = std::collections::BTreeSet::new();
    for _ in 0..5000 {
        let i = r.below(1 << 20) + if r.below(2) == 0 { 0 } else { 1 << 40 };
        if r.below(10) == 0 {
            let n = r.below(5000);
            t.insert_range(i, i + n);
            s.extend(i..=i + n);
        } else {
            assert_eq!(t.insert(i), s.insert(i));
        }
        assert_eq!(t.len(), s.len());
    }
    let mut v: Vec<u64> = t.iter().map(|a| (a / T) as u64).collect();
    v.sort_unstable();
    assert_eq!(v, s.iter().copied().collect::<Vec<_>>());
    assert!(s.iter().all(|&i| t.contains_index(i)));
    assert!(!t.contains(T + 1), "an address inside a tile is no tile");
    // the address space's last tiles
    let mut end = TileSet::default();
    end.insert_range(u64::MAX - 3, u64::MAX);
    assert_eq!(end.len(), 4);
}

/// The present wakes the worker and does not wait for it; updates are at most every
/// `UPDATE_EVERY`, at the present's time, and they run on the worker's thread.
#[test]
fn the_update_runs_off_the_presenting_thread() {
    static W: Worker = Worker::new();
    let (tx, rx) = std::sync::mpsc::channel();
    let join = W
        .spawn(move |now| {
            // (an update as long as the longest seen at a present)
            std::thread::sleep(Duration::from_millis(60));
            tx.send((std::thread::current().id(), now)).unwrap();
        })
        .expect("a thread");
    let t0 = Instant::now();
    let at = |ms: u64| t0 + Duration::from_millis(ms);
    let took = Instant::now();
    assert!(W.frame(at(0)), "the first present updates");
    assert!(took.elapsed() < Duration::from_millis(30), "the present does not wait for the update ({:?})", took.elapsed());
    assert!(!W.frame(at(100)) && !W.frame(at(499)), "not again before {UPDATE_EVERY:?}");
    let (id, now) = rx.recv_timeout(Duration::from_secs(10)).expect("an update");
    assert_ne!(id, std::thread::current().id());
    assert_eq!(now, at(0), "the classes are of the present's time");
    // presents due while the worker is busy: the last one is updated for
    assert!(W.frame(at(500)));
    assert!(W.frame(at(1000)));
    let mut got = vec![rx.recv_timeout(Duration::from_secs(10)).unwrap().1];
    if got[0] != at(1000) {
        got.push(rx.recv_timeout(Duration::from_secs(10)).unwrap().1);
    }
    assert_eq!(*got.last().unwrap(), at(1000));
    W.stop();
    join.join().unwrap();
}

/// What a present costs with the worker (`--nocapture` prints it): every other one wakes the
/// worker, which is busy for 20 ms at each update.
#[test]
fn a_present_costs_little() {
    static W: Worker = Worker::new();
    let join = W.spawn(|_| std::thread::sleep(Duration::from_millis(20))).expect("a thread");
    let t0 = Instant::now();
    let (mut sum, mut max) = (Duration::ZERO, Duration::ZERO);
    let n = 400u64;
    for i in 0..n {
        let t = Instant::now();
        W.frame(t0 + Duration::from_millis(250 * i));
        let d = t.elapsed();
        sum += d;
        max = max.max(d);
        if i % 8 == 0 {
            std::thread::sleep(Duration::from_millis(2));
        }
    }
    W.stop();
    join.join().unwrap();
    println!("hot residency present: {:.2} µs a present on average, longest {} µs ({n} presents, {} updates asked for)", sum.as_secs_f64() * 1e6 / n as f64, max.as_micros(), n / 2);
    assert!(sum / (n as u32) < Duration::from_micros(200));
}

/// What mapping an update's keys costs, before and now (`--nocapture` prints it), at about the
/// size run 76 saw: 50 000 keys an update, 160 000 window tiles mapped (10 GiB), 100 000 target
/// pages, console buffers of up to 256 MiB in tables.
#[test]
fn an_update_maps_keys_fast() {
    let mut r = Lcg(11);
    let m = Model::new(&mut r, 160_000, 100_000);
    let keys = m.keys(&mut r, 50_000);
    let set: FxSet<u64> = keys.iter().copied().collect();
    let runs = 5;
    let (mut old_ns, mut new_ns, mut merge_ns, mut old_merge_ns) = (0u128, 0u128, 0u128, 0u128);
    let (mut old_hits, mut new_hits) = (0usize, 0usize);
    for _ in 0..runs {
        let t = Instant::now();
        let mut old_set: FxSet<u64> = FxSet::default();
        old_set.extend(set.iter().copied());
        old_merge_ns += t.elapsed().as_nanos();
        let t = Instant::now();
        let (tiles, _, _) = old_sort_keys(&old_set, |a| m.slots.get(&a).copied(), |res| m.buffers.get(&res).copied());
        old_hits = tiles.iter().filter(|&a| m.sparse.contains_key(a)).count()
            + tiles.iter().filter(|&a| m.pages.contains_key(a)).count()
            + m.arenas.iter().filter(|&&(b, n)| old_any_in(&tiles, b, n)).count();
        old_ns += t.elapsed().as_nanos();
        let t = Instant::now();
        let mut f = FrameKeys::default();
        set.iter().for_each(|&k| {
            f.insert(k);
        });
        merge_ns += t.elapsed().as_nanos();
        let t = Instant::now();
        let touched = sort_keys(f, batch(|a| m.slots.get(&a).copied()), batch_buffers(|res| m.buffers.get(&res).copied()));
        let mut hits = 0;
        for_each_touched(&touched.tiles, &m.sparse, |_| hits += 1);
        for_each_touched(&touched.tiles, &m.pages, |_| hits += 1);
        new_hits = hits + m.arenas.iter().filter(|&&(b, n)| any_in(&touched.tiles, b, n)).count();
        new_ns += t.elapsed().as_nanos();
    }
    assert_eq!(old_hits, new_hits);
    println!(
        "hot residency update (50 000 keys, 160 000 tiles mapped, 100 000 pages): before {} µs, now {} µs; the keys into the frame's set: before {} µs, now {} µs",
        old_ns / runs / 1000,
        new_ns / runs / 1000,
        old_merge_ns / runs / 1000,
        merge_ns / runs / 1000
    );
}
