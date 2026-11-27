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

fn tiles_of(keys: &FxSet<u64>) -> Vec<usize> {
    let mut v: Vec<usize> = keys.iter().filter_map(|&k| match decode(k) {
        Key::Tile(a) => Some(a),
        _ => None,
    }).collect();
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
    assert!(f.contains(&(RES | 0xABC0)));
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
    let (t, cut) = sort_keys(keys, resource_at, console_buffer);
    assert_eq!(cut, 0);
    assert_eq!(t.res, vec![0x100, 0x200, 0x900]);
    assert_eq!(t.tiles, vec![7 * T, 0x40_0000, 0x41_0000, 0x42_0000]);
    assert_eq!(t.psos, 2);
}

#[test]
fn an_address_is_in_a_heap_s_range_by_its_tile() {
    let tiles = [0x10_0000, 0x30_0000, 0x50_0000];
    assert!(any_in(&tiles, 0x30_0000, T));
    // an arena from the middle of a touched tile
    assert!(any_in(&tiles, 0x30_8000, T));
    assert!(!any_in(&tiles, 0x31_0000, 0x1F_0000));
    assert!(any_in(&tiles, 0x31_0000, 0x20_0000));
    assert!(!any_in(&[], 0, usize::MAX / 2));
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
