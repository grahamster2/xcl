//! `XCL_MEMORY_MAP=1`: every `XCL_MEMORY_MAP_EVERY` seconds (default 10), on a low-priority
//! thread of its own, what fills the process's memory: a walk of its address space
//! (`VirtualQuery`) that sorts committed memory into XCL's own kinds (unified-memory windows,
//! the title's reservations, the arena pool, the D3D12 driver's images, mappings and
//! write-combined memory) and by allocation size for the rest, how much of each is resident
//! (one page per 64 KiB asked, `QueryWorkingSetEx`), and the PC's memory. Off by default.
//! The walk takes no XCL lock: XCL's region lists are copied first, each under its lock briefly.
//! Windows holds other threads' memory calls while it sizes a region of gigabytes (89 ms for a
//! 6.2 GB pool, measured in VIctory), so a short stall every period is expected with it on.
//!
//! Ported from trustno1-coder/VIctory-NVIDIA crates/victory-gpu/src/copystat.rs @d12eb1b2
//! (`log_memory_map`, `process_memory`, `system_memory`; written by fxceless)
//! (GPL-3.0-or-later), modified for XCL 2026-11-27.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use windows_sys::Win32::System::Memory::{
    VirtualQuery, MEMORY_BASIC_INFORMATION, MEM_COMMIT, MEM_FREE, MEM_IMAGE, MEM_MAPPED, PAGE_NOCACHE, PAGE_WRITECOMBINE,
};

/// One page asked about for residency stands for this much.
const STEP: usize = 64 << 10;
const MB: u64 = 1 << 20;
/// groups logged by committed size
const TOP: usize = 15;

static ON: AtomicBool = AtomicBool::new(false);
/// The title's plain `VirtualAlloc` reservations (base → bytes), recorded only while the map is on.
static TITLE: Mutex<BTreeMap<usize, usize>> = Mutex::new(BTreeMap::new());

/// Starts the logging thread when `XCL_MEMORY_MAP=1` (before the title runs). With the map off
/// and `XCL_NV_CACHE_TRIM_GB` set, a thread only records the NVIDIA cache files mapped
/// (`nv_cache`), every minute.
pub fn start() {
    if std::env::var_os("XCL_MEMORY_MAP").is_none_or(|v| v == "0") {
        if crate::nv_cache::enabled() {
            let _ = std::thread::Builder::new().name("xcl-nv-cache-files".into()).spawn(|| loop {
                std::thread::sleep(std::time::Duration::from_secs(60));
                note_nv_cache_files(mapped_files());
            });
        }
        return;
    }
    ON.store(true, Ordering::Release);
    let every = std::env::var("XCL_MEMORY_MAP_EVERY").ok().and_then(|v| v.parse::<u64>().ok()).unwrap_or(10).max(1);
    let _ = std::thread::Builder::new().name("xcl-memory-map".into()).spawn(move || {
        use windows_sys::Win32::System::Threading::{GetCurrentThread, SetThreadPriority, THREAD_PRIORITY_LOWEST};
        // SAFETY: this thread's own pseudo-handle.
        unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST) };
        let mut last = (std::time::Instant::now(), process_memory().2);
        loop {
            std::thread::sleep(std::time::Duration::from_secs(every));
            for line in log_memory_map() {
                crate::xlog_info!("{line}");
            }
            let (private, resident, faults) = process_memory();
            let rate = faults.saturating_sub(last.1) as f64 / last.0.elapsed().as_secs_f64().max(0.001);
            last = (std::time::Instant::now(), faults);
            crate::xlog_info!("{}", format_system(system_memory(), private, resident, rate));
        }
    });
}

/// A title reservation of `size` bytes at `base` (kernelx `VirtualAlloc`'s plain path).
pub fn note_title(base: usize, size: usize) {
    if ON.load(Ordering::Relaxed) {
        TITLE.lock().unwrap().insert(base, size);
    }
}

/// The title released the reservation at `base`.
pub fn forget_title(base: usize) {
    if ON.load(Ordering::Relaxed) {
        TITLE.lock().unwrap().remove(&base);
    }
}

/// Named ranges, in layers: the first layer that holds an address names it. Within a layer the
/// ranges do not overlap. Entries are (start, end, group).
#[derive(Default)]
struct Layers(Vec<Vec<(usize, usize, usize)>>);

impl Layers {
    fn push(&mut self, mut l: Vec<(usize, usize, usize)>) {
        l.retain(|e| e.1 > e.0);
        l.sort_unstable();
        self.0.push(l);
    }

    fn find(&self, at: usize) -> Option<usize> {
        self.0.iter().find_map(|l| {
            let i = l.partition_point(|e| e.0 <= at);
            (i > 0 && at < l[i - 1].1).then(|| l[i - 1].2)
        })
    }
}

/// Group names to indices.
#[derive(Default)]
struct Groups {
    ids: HashMap<String, usize>,
    names: Vec<String>,
}

impl Groups {
    fn id(&mut self, name: &str) -> usize {
        if let Some(&i) = self.ids.get(name) {
            return i;
        }
        self.names.push(name.to_string());
        self.ids.insert(name.to_string(), self.names.len() - 1);
        self.names.len() - 1
    }
}

/// XCL's own regions, copied from its lists (each lock held only for the copy).
fn snapshot(g: &mut Groups) -> Layers {
    let mut layers = Layers::default();
    let span = |v: Vec<(usize, usize)>, id: usize| v.into_iter().map(move |(a, n)| (a, a.saturating_add(n), id));
    if let Some(u) = crate::gpu::uma::made() {
        let (start, end) = u.pool_range();
        layers.push(vec![(start, end, g.id("arenas: XCL's UMA arena pool"))]);
        let id = g.id("title: reservations in the sparse windows");
        layers.push(span(u.sparse_reservations(), id).collect());
        let mut windows = Vec::new();
        for k in 0..crate::gpu::uma::MAX_HEAP_WINDOWS {
            let (a, n) = u.window(k);
            let name = if u.is_sparse_window(k) {
                "UMA: sparse windows outside the title's reservations".to_string()
            } else {
                format!("UMA: coherent window {k}{}", if u.is_driver_window(k) { " (driver heap)" } else { "" })
            };
            windows.push((a, a + n, g.id(&name)));
        }
        layers.push(windows);
    }
    let id = g.id("title: graphics reservations (write-watched)");
    layers.push(span(crate::hle::kernelx::graphics_reservations(), id).collect());
    let id = g.id("title: VirtualAlloc reservations");
    let title: Vec<(usize, usize)> = TITLE.lock().unwrap().iter().map(|(&a, &n)| (a, n)).collect();
    layers.push(span(title, id).collect());
    let modules = crate::modules::mapped_modules();
    layers.push(modules.iter().map(|m| (m.base, m.base + m.size, g.id(&format!("title: image {}", m.name)))).collect());
    layers
}

/// The group of committed memory no XCL list names, from what Windows says of it: its type,
/// protection, the whole size of its allocation, and for images and mappings the file.
fn region_group(kind: u32, protect: u32, whole: usize, path: Option<&str>) -> String {
    let lower = path.map(|p| p.to_ascii_lowercase());
    let lower = lower.as_deref().unwrap_or("");
    let file = path.map_or("", |p| p.rsplit('\\').next().unwrap_or(p));
    let combined = protect & (PAGE_WRITECOMBINE | PAGE_NOCACHE) != 0;
    if kind == MEM_IMAGE {
        // GPU user-mode drivers are installed through the driver store
        if lower.contains("\\driverstore\\") {
            return format!("D3D12 driver: image {file}");
        }
        return format!("image: {file}");
    }
    if kind == MEM_MAPPED {
        const CACHES: [&str; 5] = ["\\nvidia\\dxcache\\", "\\nvidia\\glcache\\", "\\d3dscache\\", "\\amd\\dxcache\\", "\\amd\\dxccache\\"];
        if CACHES.iter().any(|c| lower.contains(c)) {
            return format!("D3D12 driver: shader cache file {file}");
        }
        if combined {
            return "D3D12 driver: write-combined or uncached mappings".into();
        }
        return if file.is_empty() { "mapped: pagefile-backed sections".into() } else { format!("mapped: {file}") };
    }
    if combined {
        return "D3D12 driver: write-combined or uncached private memory".into();
    }
    match whole {
        // what an NT heap grows by
        0xfd_0000 => "rest: process heap segments (15.81 MB)".into(),
        n if n >= 1 << 20 => format!("rest: private allocations of {:.2} MB", n as f64 / MB as f64),
        _ => "rest: private allocations under 1 MB".into(),
    }
}

/// The file mapped at `base`.
fn mapped_file(base: usize) -> Option<String> {
    use windows_sys::Win32::System::ProcessStatus::K32GetMappedFileNameW;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    let mut buf = [0u16; 520];
    // SAFETY: our own process; the buffer's length in characters.
    let n = unsafe { K32GetMappedFileNameW(GetCurrentProcess(), base as *const _, buf.as_mut_ptr(), buf.len() as u32) };
    (n > 0).then(|| String::from_utf16_lossy(&buf[..n as usize]))
}

/// The files mapped into the process (one entry per mapping).
fn mapped_files() -> Vec<String> {
    let mut out = Vec::new();
    let mut at = 0usize;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { VirtualQuery(at as *const _, &mut m, size_of_val(&m)) } == 0 {
            break;
        }
        if m.Type == MEM_MAPPED && m.BaseAddress == m.AllocationBase {
            out.extend(mapped_file(m.BaseAddress as usize));
        }
        let next = (m.BaseAddress as usize).saturating_add(m.RegionSize);
        if next <= at {
            break;
        }
        at = next;
    }
    out
}

/// The NVIDIA shader cache files among the mapped `paths` go to the title's list
/// (`nv_cache_files.txt` next to xcl.log), which `XCL_NV_CACHE_TRIM_GB` trims at the next start.
fn note_nv_cache_files(paths: impl IntoIterator<Item = String>) {
    let Some(boot) = crate::state::boot() else { return };
    let paths: Vec<String> = paths.into_iter().collect();
    for n in crate::nv_cache::note_mapped(&boot.logs_dir(), paths.iter().map(String::as_str)) {
        crate::xlog_info!("memory map: the NVIDIA driver maps its shader cache file {n} (listed in {})", crate::nv_cache::LIST);
    }
}

/// Which of the pages in `asked` are resident; adds the bytes each stands for (group, type,
/// bytes) to its group and its type.
fn in_memory(asked: &mut Vec<[usize; 2]>, stands: &mut Vec<(usize, usize, u64)>, groups: &mut [(u64, u64)], types: &mut [(u64, u64); 3]) {
    use windows_sys::Win32::System::ProcessStatus::K32QueryWorkingSetEx;
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // SAFETY: our own process; `asked` holds that many entries of two words each.
    let ok = unsafe { K32QueryWorkingSetEx(GetCurrentProcess(), asked.as_mut_ptr().cast(), (asked.len() * 16) as u32) } != 0;
    if ok {
        for (a, &(g, t, bytes)) in asked.iter().zip(stands.iter()) {
            // bit 0: the page is in the working set
            if a[1] & 1 != 0 {
                groups[g].1 += bytes;
                types[t].1 += bytes;
            }
        }
    }
    asked.clear();
    stands.clear();
}

/// Walks the address space and returns the log lines: totals by type, then the biggest groups.
fn log_memory_map() -> Vec<String> {
    let started = std::time::Instant::now();
    let mut g = Groups::default();
    let layers = snapshot(&mut g);
    // allocation base, start, bytes, committed, type, protection
    let mut regions: Vec<(usize, usize, usize, bool, u32, u32)> = Vec::new();
    let mut at = 0usize;
    loop {
        // SAFETY: plain data filled in by the query.
        let mut m: MEMORY_BASIC_INFORMATION = unsafe { std::mem::zeroed() };
        // SAFETY: a valid out pointer of the right size.
        if unsafe { VirtualQuery(at as *const _, &mut m, size_of_val(&m)) } == 0 {
            break;
        }
        if m.State != MEM_FREE {
            regions.push((m.AllocationBase as usize, m.BaseAddress as usize, m.RegionSize, m.State == MEM_COMMIT, m.Type, m.Protect));
        }
        let next = (m.BaseAddress as usize).saturating_add(m.RegionSize);
        if next <= at {
            break;
        }
        at = next;
    }
    let mut whole: HashMap<usize, usize> = HashMap::new();
    for r in &regions {
        *whole.entry(r.0).or_default() += r.2;
    }
    let mut files: HashMap<usize, Option<String>> = HashMap::new();
    let mut groups: Vec<(u64, u64)> = Vec::new();
    // private, mapped, image: (committed, resident)
    let mut types = [(0u64, 0u64); 3];
    let (mut asked, mut stands) = (Vec::with_capacity(4096), Vec::with_capacity(4096));
    for r in regions.iter().filter(|r| r.3) {
        let t = match r.4 {
            MEM_MAPPED => 1,
            MEM_IMAGE => 2,
            _ => 0,
        };
        let path = (t != 0).then(|| files.entry(r.0).or_insert_with(|| mapped_file(r.0)).clone()).flatten();
        let fallback = g.id(&region_group(r.4, r.5, whole[&r.0], path.as_deref()));
        let mut offset = 0;
        while offset < r.2 {
            let at = r.1 + offset;
            let n = STEP.min(r.2 - offset) as u64;
            let id = layers.find(at).unwrap_or(fallback);
            groups.resize(g.names.len(), (0, 0));
            groups[id].0 += n;
            types[t].0 += n;
            asked.push([at, 0]);
            stands.push((id, t, n));
            if asked.len() == 4096 {
                in_memory(&mut asked, &mut stands, &mut groups, &mut types);
            }
            offset += STEP;
        }
    }
    if !asked.is_empty() {
        in_memory(&mut asked, &mut stands, &mut groups, &mut types);
    }
    note_nv_cache_files(files.into_values().flatten());
    let named: Vec<(String, u64, u64)> = g.names.into_iter().zip(groups).map(|(n, (c, r))| (n, c, r)).collect();
    let mut lines = vec![format_totals(regions.len(), started.elapsed().as_millis(), &types)];
    lines.extend(format_top(named, TOP));
    let (compute, graphics) = crate::hle::d3d12x::real::blob_cache_bytes();
    let shaders = crate::gpu::recomp::pso::shader_cache_bytes();
    lines.push(format!(
        "memory map: XCL's caches in the process heap: pipeline blobs {} MB compute + {} MB graphics, translated shaders {} MB",
        compute as u64 / MB,
        graphics as u64 / MB,
        shaders as u64 / MB
    ));
    lines
}

fn format_totals(regions: usize, ms: u128, t: &[(u64, u64); 3]) -> String {
    format!(
        "memory map ({regions} regions, {ms} ms): committed private {} MB ({} MB resident), mapped {} MB ({} MB resident), image {} MB ({} MB resident)",
        t[0].0 / MB,
        t[0].1 / MB,
        t[1].0 / MB,
        t[1].1 / MB,
        t[2].0 / MB,
        t[2].1 / MB
    )
}

/// The `top` groups by committed bytes (name, committed, resident), and a line for the rest.
fn format_top(mut groups: Vec<(String, u64, u64)>, top: usize) -> Vec<String> {
    groups.retain(|g| g.1 > 0);
    groups.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let mut lines: Vec<String> = groups
        .iter()
        .take(top)
        .enumerate()
        .map(|(i, (name, c, r))| format!("memory map {:>2}. {name}: {} MB committed, {} MB resident", i + 1, c / MB, r / MB))
        .collect();
    if groups.len() > top {
        let (c, r) = groups[top..].iter().fold((0, 0), |s, g| (s.0 + g.1, s.1 + g.2));
        lines.push(format!("memory map     {} more groups: {} MB committed, {} MB resident", groups.len() - top, c / MB, r / MB));
    }
    lines
}

/// The PC's memory in bytes: physical (available, total), commit (charge, limit), page files
/// (in use, size).
struct System {
    available: u64,
    physical: u64,
    commit: u64,
    limit: u64,
    paged: u64,
    page_files: u64,
}

fn format_system(s: System, private: u64, resident: u64, faults: f64) -> String {
    format!(
        "memory map: system: {} MB of {} MB RAM available; commit {} MB of {} MB; page files {} MB of {} MB in use; process {} MB private, {} MB working set, {:.0} page faults/s",
        s.available / MB,
        s.physical / MB,
        s.commit / MB,
        s.limit / MB,
        s.paged / MB,
        s.page_files / MB,
        private / MB,
        resident / MB,
        faults
    )
}

/// The process's memory: (committed private bytes, working set bytes, page faults so far).
fn process_memory() -> (u64, u64, u64) {
    use windows_sys::Win32::System::ProcessStatus::{K32GetProcessMemoryInfo, PROCESS_MEMORY_COUNTERS_EX};
    use windows_sys::Win32::System::Threading::GetCurrentProcess;
    // SAFETY: plain data the call fills in.
    let mut c: PROCESS_MEMORY_COUNTERS_EX = unsafe { std::mem::zeroed() };
    c.cb = size_of_val(&c) as u32;
    // SAFETY: our own process and a sized structure.
    unsafe { K32GetProcessMemoryInfo(GetCurrentProcess(), (&mut c as *mut PROCESS_MEMORY_COUNTERS_EX).cast(), c.cb) };
    (c.PrivateUsage as u64, c.WorkingSetSize as u64, u64::from(c.PageFaultCount))
}

#[link(name = "ntdll", kind = "raw-dylib")]
extern "system" {
    fn NtQuerySystemInformation(class: u32, info: *mut std::ffi::c_void, len: u32, returned: *mut u32) -> i32;
}

fn system_memory() -> System {
    use windows_sys::Win32::System::ProcessStatus::{K32GetPerformanceInfo, PERFORMANCE_INFORMATION};
    use windows_sys::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
    // SAFETY: plain data the calls fill in, with their sizes.
    let mut m: MEMORYSTATUSEX = unsafe { std::mem::zeroed() };
    m.dwLength = size_of_val(&m) as u32;
    unsafe { GlobalMemoryStatusEx(&mut m) };
    let mut p: PERFORMANCE_INFORMATION = unsafe { std::mem::zeroed() };
    p.cb = size_of_val(&p) as u32;
    unsafe { K32GetPerformanceInfo(&mut p, p.cb) };
    let page = p.PageSize.max(1) as u64;
    // SystemPageFileInformation (18): entries of (next offset, total, in use, peak pages, name)
    let mut buf = [0u64; 512];
    let (mut paged, mut size) = (0u64, 0u64);
    // SAFETY: a buffer of its length; entries lie inside it at the offsets the call gives.
    if unsafe { NtQuerySystemInformation(18, buf.as_mut_ptr().cast(), 4096, std::ptr::null_mut()) } >= 0 {
        let words: &[u32] = unsafe { std::slice::from_raw_parts(buf.as_ptr().cast(), 1024) };
        let mut at = 0usize;
        while at + 4 <= words.len() {
            size += u64::from(words[at + 1]) * page;
            paged += u64::from(words[at + 2]) * page;
            let next = words[at] as usize / 4;
            if next == 0 {
                break;
            }
            at += next;
        }
    }
    System {
        available: m.ullAvailPhys,
        physical: m.ullTotalPhys,
        commit: p.CommitTotal as u64 * page,
        limit: p.CommitLimit as u64 * page,
        paged,
        page_files: size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows_sys::Win32::System::Memory::{MEM_PRIVATE, PAGE_READWRITE};

    #[test]
    fn layers_name_an_address_by_the_first_layer_holding_it() {
        let mut l = Layers::default();
        l.push(vec![(0x2000, 0x3000, 7)]);
        l.push(vec![(0x5000, 0x6000, 2), (0x1000, 0x4000, 1), (0x9000, 0x9000, 3)]);
        assert_eq!(l.find(0x1000), Some(1));
        assert_eq!(l.find(0x2fff), Some(7), "the arena pool inside a window is the pool");
        assert_eq!(l.find(0x3000), Some(1));
        assert_eq!(l.find(0x4000), None, "ends are exclusive");
        assert_eq!(l.find(0x5800), Some(2));
        assert_eq!(l.find(0x9000), None, "empty ranges are dropped");
        assert_eq!(l.find(0), None);
    }

    #[test]
    fn regions_no_list_names_are_grouped_by_owner_and_size() {
        let nv = r"\Device\HarddiskVolume3\Windows\System32\DriverStore\FileRepository\nv_dispi.inf_amd64_1\nvwgf2umx.dll";
        assert_eq!(region_group(MEM_IMAGE, 0, 0, Some(nv)), "D3D12 driver: image nvwgf2umx.dll");
        assert_eq!(region_group(MEM_IMAGE, 0, 0, Some(r"\Device\X\Windows\System32\d3d12.dll")), "image: d3d12.dll");
        let cache = r"\Device\X\Users\u\AppData\Local\NVIDIA\DXCache\7f1c_0.toc";
        assert_eq!(region_group(MEM_MAPPED, PAGE_READWRITE, 1 << 30, Some(cache)), "D3D12 driver: shader cache file 7f1c_0.toc");
        assert_eq!(region_group(MEM_MAPPED, PAGE_READWRITE, 1 << 20, None), "mapped: pagefile-backed sections");
        assert_eq!(region_group(MEM_MAPPED, PAGE_READWRITE | PAGE_WRITECOMBINE, 1 << 20, None), "D3D12 driver: write-combined or uncached mappings");
        assert_eq!(region_group(MEM_PRIVATE, PAGE_READWRITE | PAGE_WRITECOMBINE, 4 << 20, None), "D3D12 driver: write-combined or uncached private memory");
        assert_eq!(region_group(MEM_PRIVATE, PAGE_READWRITE, 0xfd_0000, None), "rest: process heap segments (15.81 MB)");
        assert_eq!(region_group(MEM_PRIVATE, PAGE_READWRITE, 4 << 20, None), "rest: private allocations of 4.00 MB");
        assert_eq!(region_group(MEM_PRIVATE, PAGE_READWRITE, 0x8000, None), "rest: private allocations under 1 MB");
    }

    #[test]
    fn the_biggest_groups_come_first_and_the_rest_is_summed() {
        let g = |n: &str, c: u64, r: u64| (n.to_string(), c * MB, r * MB);
        let lines = format_top(vec![g("a", 10, 1), g("b", 300, 200), g("empty", 0, 0), g("c", 20, 20), g("d", 5, 0)], 2);
        assert_eq!(
            lines,
            [
                "memory map  1. b: 300 MB committed, 200 MB resident",
                "memory map  2. c: 20 MB committed, 20 MB resident",
                "memory map     2 more groups: 15 MB committed, 1 MB resident",
            ]
        );
        assert_eq!(format_top(vec![g("a", 1, 1)], 15).len(), 1, "no summary line when all groups fit");
        let t = format_totals(12, 615, &[(21_600 * MB, 18_000 * MB), (2 * MB, MB), (400 * MB, 300 * MB)]);
        assert_eq!(t, "memory map (12 regions, 615 ms): committed private 21600 MB (18000 MB resident), mapped 2 MB (1 MB resident), image 400 MB (300 MB resident)");
        let s = System { available: 300 * MB, physical: 31_900 * MB, commit: 52_000 * MB, limit: 56_000 * MB, paged: 9_000 * MB, page_files: 24_000 * MB };
        assert_eq!(
            format_system(s, 21_600 * MB, 17_000 * MB, 1234.4),
            "memory map: system: 300 MB of 31900 MB RAM available; commit 52000 MB of 56000 MB; page files 9000 MB of 24000 MB in use; process 21600 MB private, 17000 MB working set, 1234 page faults/s"
        );
    }

    #[test]
    fn the_walk_of_this_process_adds_up() {
        // 64 MB committed and written: private, resident, one allocation of its own
        let held = vec![1u8; 64 << 20];
        let lines = log_memory_map();
        lines.iter().for_each(|l| println!("{l}"));
        let mb_after = |l: &str, key: &str| l.split(key).nth(1).and_then(|r| r.split(' ').next()?.parse::<u64>().ok()).unwrap();
        assert!(mb_after(&lines[0], "committed private ") >= 64, "{}", lines[0]);
        assert!(mb_after(&lines[0], "MB (") >= 60, "most of it resident: {}", lines[0]);
        let big = lines.iter().find(|l| l.contains("rest: private allocations of 64.")).expect("its own group");
        assert!(mb_after(big, "MB: ") >= 64, "{big}");
        drop(held);
        assert!(lines.iter().any(|l| l.contains("image: ")), "the test binary is an image");
        let s = system_memory();
        println!("{}", format_system(system_memory(), process_memory().0, process_memory().1, 0.0));
        assert!(s.physical > 0 && s.available <= s.physical && s.commit <= s.limit);
        assert!(s.paged <= s.page_files);
    }

    #[test]
    fn groups_keep_one_index_per_name() {
        let mut g = Groups::default();
        assert_eq!((g.id("a"), g.id("b"), g.id("a")), (0, 1, 0));
        assert_eq!(g.names, ["a", "b"]);
    }
}
