//! PSOs for a split compute entry. The first piece is the real PSO returned
//! to the caller. Other pieces and the lazily built whole form belong to it.
use super::*;
use crate::gpu::{dxil::kernel::DispatchChain, recomp::pso};
use std::sync::{Arc, LazyLock, OnceLock};

static PROGRAMS: AtomicU64 = AtomicU64::new(0);
static PIECES: AtomicU64 = AtomicU64::new(0);
static DISPATCHES: AtomicU64 = AtomicU64::new(0);
/// slabs dispatched (`pso::dispatch_slabs`)
static SLABS: AtomicU64 = AtomicU64::new(0);
/// scratch alive now, the most there was, and the most a dispatch would have needed at once (bytes)
static LIVE: AtomicU64 = AtomicU64::new(0);
static PEAK: AtomicU64 = AtomicU64::new(0);
static WANTED: AtomicU64 = AtomicU64::new(0);

/// (only when a count changed since the last line: called every 10 s in every run)
pub(crate) fn log_counts() {
    log_residency("every 10 s");
    static LAST: Mutex<[u64; 7]> = Mutex::new([0; 7]);
    let now = [&PROGRAMS, &PIECES, &DISPATCHES, &SLABS, &LIVE, &PEAK, &WANTED].map(|c| c.load(Ordering::Relaxed));
    if std::mem::replace(&mut *LAST.lock().unwrap(), now) == now {
        return;
    }
    let mib = |a: &AtomicU64| a.load(Ordering::Relaxed) >> 20;
    crate::xlog_info!("compute split: {} programs split, {} pieces built, {} chained dispatches issued in {} slabs; scratch {} MiB (most {}, largest dispatch {} in one slab)",
        PROGRAMS.load(Ordering::Relaxed), PIECES.load(Ordering::Relaxed), DISPATCHES.load(Ordering::Relaxed), SLABS.load(Ordering::Relaxed),
        mib(&LIVE), mib(&PEAK), mib(&WANTED));
}

/// The scratch a dispatch uses at once, `XCL_SPLIT_SCRATCH_MB` (256): more groups than fit run in
/// slabs. A retail title's split programs need 190 KB for each group (median), 3 GB for 960x540
/// pixels in groups of 8x4.
fn scratch_budget() -> u64 {
    static B: OnceLock<u64> = OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_SPLIT_SCRATCH_MB").ok().and_then(|v| v.trim().parse().ok()).filter(|&m| m > 0).unwrap_or(256u64).min(4096) << 20)
}

struct PipelineBuilder {
    device: ID3D12Device,
    create: PsoFn,
    signature: ID3D12RootSignature,
    flags: D3D12_PIPELINE_STATE_FLAGS,
    node: u32,
    record: u64,
}

impl PipelineBuilder {
    unsafe fn pipeline(&self, dxil: &[u8], iid: &GUID) -> Option<ID3D12PipelineState> {
        let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC { NodeMask: self.node, Flags: self.flags, ..Default::default() };
        let mut raw = std::ptr::null_mut();
        let (hr, in_driver, _) = compute_container(self.device.as_raw(), self.create, &desc, self.signature.as_raw(),
            dxil, self.record, iid, &mut raw, ComputeBuild::Chain);
        // (as `translated_compute_pso_once` logs a huge whole program the driver refuses)
        if hr < 0 && dxil.len() >= huge_pso_bytes() {
            crate::xlog_warn!(
                "D3D12.X: the driver refused a dispatch chain's compute PSO of record {:016x} ({} B DXIL, driver cache {}) with {hr:#x} after {} ms in the driver",
                self.record,
                dxil.len(),
                driver_cache::name(dxil, self.flags.0),
                in_driver.as_millis()
            );
        }
        (hr >= 0 && !raw.is_null()).then(|| ID3D12PipelineState::from_raw(raw))
    }
}

struct Chain {
    builder: Arc<PipelineBuilder>,
    /// the pieces after the first, until the whole form is built (`release_tail`); a recording
    /// holds its own reference, so the pieces stay alive until the GPU is done with them
    tail: Mutex<Option<Arc<[ID3D12PipelineState]>>>,
    words: u32,
    whole_code: Vec<u8>,
    whole: OnceLock<ID3D12PipelineState>,
    build_whole: Mutex<()>,
    /// driver blob bytes (`GetCachedBlob`) of the first piece, the tail and the whole form
    first_bytes: u64,
    tail_bytes: u64,
    whole_bytes: AtomicU64,
}

impl Chain {
    unsafe fn whole(&self) -> Result<ID3D12PipelineState, String> {
        if let Some(p) = self.whole.get() { return Ok(p.clone()); }
        // (a driver compile on the title's thread, of the largest programs: the title is held
        // while it lasts, as in any other long PC call)
        let _call = super::super::pso_async::pc_call();
        let _guard = self.build_whole.lock().unwrap();
        if let Some(p) = self.whole.get() { return Ok(p.clone()); }
        let p = self.builder.pipeline(&self.whole_code, &pc_iid(Kind::PipelineState)).ok_or("whole compute pipeline")?;
        self.whole_bytes.store(blob_bytes(&p), Ordering::Relaxed);
        let _ = self.whole.set(p.clone());
        drop(_guard);
        // (dropped here, with no lock held: a piece's Release goes through `forget_pso`)
        drop(self.release_tail());
        Ok(p)
    }

    /// The tail pieces while the chain keeps them.
    fn tail(&self) -> Option<Arc<[ID3D12PipelineState]>> {
        self.tail.lock().unwrap().clone()
    }

    /// Once the whole form is built, the chain stops holding its tail pieces: every later
    /// dispatch takes the whole form, and the driver keeps one copy of the program's code (#2403,
    /// NVIDIA's 512 MB shader-code heap). Recordings still on the GPU keep the pieces they use.
    /// `XCL_CHAIN_KEEP_PIECES=1` keeps both forms, as before. The caller drops what this returns
    /// with no lock held.
    fn release_tail(&self) -> Option<Arc<[ID3D12PipelineState]>> {
        if self.whole.get().is_none() || keep_pieces() {
            return None;
        }
        let gone = self.tail.lock().unwrap().take();
        if gone.is_some() {
            TAILS_RELEASED.fetch_add(1, Ordering::Relaxed);
        }
        gone
    }
}

fn keep_pieces() -> bool {
    static K: OnceLock<bool> = OnceLock::new();
    *K.get_or_init(|| std::env::var("XCL_CHAIN_KEEP_PIECES").is_ok_and(|v| v.trim() == "1"))
}

/// The size of the driver's blob of `p` (`GetCachedBlob`), 0 when it gives none: what the driver
/// keeps of its code, near enough.
unsafe fn blob_bytes(p: &ID3D12PipelineState) -> u64 {
    read_pso_cached_blob(p.as_raw()).map_or(0, |b| b.len() as u64)
}

/// chains whose tail pieces were released (`Chain::release_tail`)
static TAILS_RELEASED: AtomicU64 = AtomicU64::new(0);

/// What the chains hold: (pieces only, whole form only, both, driver blob bytes of all that).
fn residency() -> (usize, usize, usize, u64) {
    let chains: Vec<Arc<Chain>> = CHAINS.lock().unwrap().values().cloned().collect();
    let (mut pieces, mut whole, mut both, mut bytes) = (0, 0, 0, 0u64);
    for c in &chains {
        let has_tail = c.tail.lock().unwrap().is_some();
        let has_whole = c.whole.get().is_some();
        bytes += c.first_bytes;
        if has_tail {
            bytes += c.tail_bytes;
        }
        if has_whole {
            bytes += c.whole_bytes.load(Ordering::Relaxed);
        }
        match (has_tail, has_whole) {
            (true, false) => pieces += 1,
            (false, true) => whole += 1,
            (true, true) => both += 1,
            (false, false) => {}
        }
    }
    (pieces, whole, both, bytes)
}

fn log_residency(why: &str) {
    let (pieces, whole, both, bytes) = residency();
    if pieces + whole + both == 0 {
        return;
    }
    crate::xlog_info!("compute split residency ({why}): {} chains: {pieces} hold pieces only, {whole} the whole form only, {both} both; {} tails released; driver blobs of what the chains hold {} MiB",
        pieces + whole + both, TAILS_RELEASED.load(Ordering::Relaxed), bytes >> 20);
}

static CHAINS: LazyLock<Mutex<HashMap<usize, Arc<Chain>>>> = LazyLock::new(Default::default);

pub(super) unsafe fn build(this: *mut c_void, create: PsoFn, desc: &D3D12_COMPUTE_PIPELINE_STATE_DESC,
    iid: *const GUID, result: *mut *mut c_void, record: u64, whole: &[u8], samplers: &[pso::StaticSampler],
    plan: Arc<DispatchChain>) -> Option<usize> {
    if result.is_null() || iid.is_null() || plan.pieces.len() < 2 { return None; }
    let dev = ID3D12Device::from_raw_borrowed(&this)?;
    let signature = pso::dispatch_chain_root_signature(dev, samplers).ok()?;
    let rs = signature.as_raw() as usize;
    let builder = Arc::new(PipelineBuilder { device: dev.clone(), create, signature,
        flags: desc.Flags, node: desc.NodeMask, record });
    let jobs = (0..plan.pieces.len()).map(|i| {
        let plan = plan.clone();
        let builder = builder.clone();
        let iid = if i == 0 { *iid } else { pc_iid(Kind::PipelineState) };
        Box::new(move || builder.pipeline(&plan.pieces[i], &iid).map_or(0, |p| p.into_raw() as usize)) as super::super::pso_async::Job
    }).collect();
    // Take ownership of every success, including when a different piece failed.
    let built: Vec<_> = super::super::pso_async::build_pieces(jobs).into_iter()
        .map(|p| (p != 0).then(|| ID3D12PipelineState::from_raw(p as *mut c_void))).collect();
    let mut pieces = built.into_iter().collect::<Option<Vec<_>>>()?.into_iter();
    let first = pieces.next()?;
    let tail: Arc<[ID3D12PipelineState]> = pieces.collect();
    let (first_bytes, tail_bytes) = (blob_bytes(&first), tail.iter().map(|p| blob_bytes(p)).sum());
    let chain = Arc::new(Chain { builder, tail: Mutex::new(Some(tail)), words: plan.dwords_per_group,
        whole_code: whole.to_vec(), whole: OnceLock::new(), build_whole: Mutex::new(()),
        first_bytes, tail_bytes, whole_bytes: AtomicU64::new(0) });
    let programs = PROGRAMS.fetch_add(1, Ordering::Relaxed) + 1;
    PIECES.fetch_add(plan.pieces.len() as u64, Ordering::Relaxed);
    *result = first.into_raw();
    CHAINS.lock().unwrap().insert(*result as usize, chain);
    if programs % 2048 == 0 {
        log_residency("every 2048 chains");
    }
    Some(rs)
}

pub(super) fn forget(pso: usize) {
    let gone = CHAINS.lock().unwrap().remove(&pso);
    drop(gone); // releasing a tail PSO also goes through forget_pso
}

pub(in crate::hle::d3d12x) unsafe fn whole(pso: usize) -> Result<Option<ID3D12PipelineState>, String> {
    let chain = CHAINS.lock().unwrap().get(&pso).cloned();
    chain.map(|c| c.whole()).transpose()
}

struct Scratch {
    buffer: ID3D12Resource,
    bytes: u64,
}

impl Drop for Scratch {
    fn drop(&mut self) {
        LIVE.fetch_sub(self.bytes, Ordering::Relaxed);
    }
}

struct Recording {
    scratch: Arc<Scratch>,
    first: ID3D12PipelineState,
    /// the tail pieces this recording runs (the chain may stop holding them before the GPU is done)
    tail: Arc<[ID3D12PipelineState]>,
    /// (only held: the chain lives as long as what was recorded with it, as before)
    #[allow(dead_code)]
    chain: Arc<Chain>,
}

static RECORDINGS: LazyLock<Mutex<HashMap<usize, Vec<Arc<Recording>>>>> = LazyLock::new(Default::default);
/// list → the allocator it records on and the scratch its chains share: they are dispatched one
/// after another, with a barrier between them (the recordings keep it alive)
static SHARED: LazyLock<Mutex<HashMap<usize, (usize, std::sync::Weak<Scratch>)>>> = LazyLock::new(Default::default);

/// `list` begins another recording, which the GPU may run beside the last one.
pub(in crate::hle::d3d12x) fn list_reset(list: usize) {
    SHARED.lock().unwrap().remove(&list);
}

/// `to` records right after `from` on the same allocator: its chains go on in `from`'s scratch.
pub(in crate::hle::d3d12x) fn list_follows(from: usize, to: usize) {
    let mut shared = SHARED.lock().unwrap();
    if let Some(e) = shared.get(&from).cloned() {
        shared.insert(to, e);
    }
}

/// The scratch of all recordings alive together, `XCL_SPLIT_SCRATCH_TOTAL_MB` (384). A recording
/// keeps its scratch until its allocator is reset, and a title that records ahead of the GPU has
/// many: with a bound on one dispatch alone a retail title's world took 1.2 GB and went on growing.
fn scratch_total() -> u64 {
    static B: OnceLock<u64> = OnceLock::new();
    *B.get_or_init(|| std::env::var("XCL_SPLIT_SCRATCH_TOTAL_MB").ok().and_then(|v| v.trim().parse().ok()).filter(|&m| m > 0).unwrap_or(384u64) << 20)
}

/// A recording gets at least this much (or one group's worth) however much the others hold.
const LEAST: u64 = 2 << 20;

/// How much scratch a dispatch of `needed` bytes (`group` for one group) runs with at once: the
/// list's own buffer (`had` bytes) when that is enough or no more may be made, else a new one of
/// the bytes given (second: make one).
fn scratch_bytes(needed: u64, group: u64, budget: u64, had: Option<u64>, live: u64, total: u64) -> (u64, bool) {
    let want = needed.min(budget).max(group);
    let room = total.saturating_sub(live);
    match had {
        Some(h) if h >= want || (room < want && h >= group) => (h, false),
        _ => (want.min(room.max(LEAST)).max(group), true),
    }
}

/// A scratch for a chain of `needed` bytes recorded on `list` with `allocator`, and whether a
/// chain before it on the list used it.
unsafe fn scratch_for(dev: &ID3D12Device, list: usize, allocator: usize, needed: u64, group: u64) -> Option<(Arc<Scratch>, bool)> {
    let had = SHARED.lock().unwrap().get(&list).filter(|s| s.0 == allocator).and_then(|s| s.1.upgrade());
    let (bytes, make) = scratch_bytes(needed, group, scratch_budget(), had.as_ref().map(|s| s.bytes), LIVE.load(Ordering::Relaxed), scratch_total());
    if !make {
        return had.map(|s| (s, true));
    }
    let s = Arc::new(Scratch { buffer: scratch_buffer(dev, bytes)?, bytes });
    let live = LIVE.fetch_add(bytes, Ordering::Relaxed) + bytes;
    PEAK.fetch_max(live, Ordering::Relaxed);
    SHARED.lock().unwrap().insert(list, (allocator, Arc::downgrade(&s)));
    Some((s, false))
}

pub(in crate::hle::d3d12x) fn allocator_reset(allocator: usize) {
    let recordings = RECORDINGS.lock().unwrap().remove(&allocator);
    drop(recordings);
}

unsafe fn scratch_buffer(dev: &ID3D12Device, bytes: u64) -> Option<ID3D12Resource> {
    let desc = D3D12_RESOURCE_DESC {
        Dimension: D3D12_RESOURCE_DIMENSION_BUFFER, Width: bytes, Height: 1, DepthOrArraySize: 1, MipLevels: 1,
        SampleDesc: windows::Win32::Graphics::Dxgi::Common::DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Layout: D3D12_TEXTURE_LAYOUT_ROW_MAJOR, Flags: D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, ..Default::default()
    };
    let mut resource = None;
    dev.CreateCommittedResource(&D3D12_HEAP_PROPERTIES { Type: D3D12_HEAP_TYPE_DEFAULT, CreationNodeMask: 1, VisibleNodeMask: 1, ..Default::default() },
        D3D12_HEAP_FLAG_NONE, &desc, D3D12_RESOURCE_STATE_UNORDERED_ACCESS, None, &mut resource).ok()?;
    resource
}

/// A piece or the whole form on `list`: on a list XCL adopted (a title's) past the hook of a
/// title's SetPipelineState, as `bind::bind_abi` binds a compute pipeline. Through the hook it
/// became the list's bound pipeline (the first one of a list split after it) and got a DRED
/// marker without a label.
unsafe fn bind(list: &ID3D12GraphicsCommandList, pso: &ID3D12PipelineState) {
    if is_adopted(list.as_raw()) {
        super::super::list_x::set_compute_pipeline(list.as_raw(), pso.as_raw());
    } else {
        list.SetPipelineState(pso);
    }
}

/// Called with the original six ABI views bound. The extended signature is
/// common to every piece and the whole fallback. Choose before recording any
/// part of the dispatch, including when scratch allocation fails.
pub(in crate::hle::d3d12x) unsafe fn dispatch(list: &ID3D12GraphicsCommandList, pso: usize, allocator: usize, grid: [u32; 3]) -> Result<(), String> {
    let [x, y, z] = grid;
    let chain = CHAINS.lock().unwrap().get(&pso).cloned();
    let Some(chain) = chain else { list.Dispatch(x, y, z); return Ok(()) };
    if grid.contains(&0) { return Ok(()); }
    // A title Reset can switch to a different PC allocator while this
    // recording is still submitted on the old one. Retain by backing.
    let allocator = if super::super::alloc_pool::enabled() {
        super::super::alloc_pool::recording_backing(list.as_raw() as usize).unwrap_or(0)
    } else { allocator };
    let group = u64::from(chain.words) * 4;
    let needed = grid.into_iter().fold(group, |s, n| s.saturating_mul(n as u64));
    WANTED.fetch_max(needed, Ordering::Relaxed);
    // (no tail: the chain has its whole form, `Chain::release_tail`)
    let tail = chain.tail();
    let scratch = (tail.is_some() && allocator != 0 && list.GetType() != D3D12_COMMAND_LIST_TYPE_BUNDLE)
        .then(|| scratch_for(&chain.builder.device, list.as_raw() as usize, allocator, needed, group)).flatten();
    let most = scratch.as_ref().map_or(1, |(s, _)| (s.bytes / group).max(1));
    let slabs = pso::dispatch_slabs(grid, most);
    let (Some((scratch, used)), Some(tail)) = (scratch, tail) else {
        bind(list, &chain.whole()?);
        list.Dispatch(x, y, z);
        return Ok(());
    };
    let first = ID3D12PipelineState::from_raw_borrowed(&(pso as *mut c_void)).ok_or("first compute piece")?.clone();
    let held = Arc::new(Recording { scratch, first, tail, chain });
    RECORDINGS.lock().unwrap().entry(allocator).or_default().push(held.clone());
    DISPATCHES.fetch_add(1, Ordering::Relaxed);
    SLABS.fetch_add(slabs.len() as u64, Ordering::Relaxed);
    pso::record_dispatch_chain(list, grid, most, &held.scratch.buffer, used, std::iter::once(&held.first).chain(held.tail.iter()), |p| bind(list, p));
    Ok(())
}

#[cfg(test)]
mod tail_tests {
    use super::*;

    unsafe extern "system" fn no_create(_: *mut c_void, _: *const c_void, _: *const GUID, _: *mut *mut c_void) -> HRESULT {
        E_INVALIDARG
    }

    /// A chain gives up its tail pieces only once its whole form is built, and a recording that
    /// took them keeps them alive after that.
    #[test]
    fn a_chain_releases_its_tail_once_the_whole_form_is_built() {
        unsafe {
            if crate::gpu::recomp::dxc::location().is_err() {
                return crate::gpu::recomp::runner::skip_test("no DXC");
            }
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                if let Ok(warp) = CreateDXGIFactory1::<IDXGIFactory4>().and_then(|f| f.EnumWarpAdapter::<IDXGIAdapter>()) {
                    let _ = D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev);
                }
            }
            let Some(dev) = dev else { return crate::gpu::recomp::runner::skip_test("no D3D12 device") };
            let dxil = crate::gpu::recomp::dxc::compile(
                "RWByteAddressBuffer mem : register(u2); [numthreads(1,1,1)] void main() { mem.Store(0, 1); }",
                "main", "cs_6_6", &[],
            ).expect("DXC compile");
            let rs = pso::dispatch_chain_root_signature(&dev, &[]).expect("dispatch chain RS");
            let make = || {
                let mut desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                    pRootSignature: std::mem::ManuallyDrop::new(Some(rs.clone())),
                    CS: D3D12_SHADER_BYTECODE { pShaderBytecode: dxil.as_ptr() as *const c_void, BytecodeLength: dxil.len() },
                    ..Default::default()
                };
                let p: ID3D12PipelineState = dev.CreateComputePipelineState(&desc).expect("CreateComputePipelineState");
                std::mem::ManuallyDrop::drop(&mut desc.pRootSignature);
                p
            };
            let builder = Arc::new(PipelineBuilder { device: dev.clone(), create: no_create, signature: rs.clone(), flags: Default::default(), node: 0, record: 0 });
            let tail: Arc<[ID3D12PipelineState]> = vec![make(), make()].into();
            let chain = Chain {
                builder, tail: Mutex::new(Some(tail)), words: 4, whole_code: Vec::new(), whole: OnceLock::new(),
                build_whole: Mutex::new(()), first_bytes: 0, tail_bytes: 0, whole_bytes: AtomicU64::new(0),
            };
            // no whole form yet: the pieces are all there is
            assert!(chain.release_tail().is_none());
            let held = chain.tail().expect("the tail is kept without a whole form");
            let _ = chain.whole.set(make());
            let released = TAILS_RELEASED.load(Ordering::Relaxed);
            let gone = chain.release_tail().expect("the tail goes once the whole form is built");
            assert!(chain.tail().is_none(), "later dispatches take the whole form");
            assert_eq!(TAILS_RELEASED.load(Ordering::Relaxed), released + 1);
            drop(gone);
            // a recording made before the release still has its pieces
            assert_eq!(held.len(), 2);
            assert_eq!(Arc::strong_count(&held), 1);
            assert!(chain.release_tail().is_none(), "released once");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{scratch_bytes, LEAST};

    const MB: u64 = 1 << 20;

    #[test]
    fn a_list_recorded_after_another_goes_on_in_its_scratch() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Direct3D12::{D3D12CreateDevice, ID3D12Device};
        use windows::Win32::Graphics::Dxgi::*;
        let mut dev: Option<ID3D12Device> = None;
        if unsafe { D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev) }.is_err() {
            let warp = unsafe { CreateDXGIFactory1::<IDXGIFactory4>().and_then(|f| f.EnumWarpAdapter::<IDXGIAdapter>()) };
            if warp.and_then(|w| unsafe { D3D12CreateDevice(&w, D3D_FEATURE_LEVEL_11_0, &mut dev) }).is_err() {
                crate::gpu::recomp::runner::skip_test("no GPU");
                return;
            }
        }
        let dev = dev.unwrap();
        let (a, b, c, alloc) = (0x7E57_5C01usize, 0x7E57_5C02usize, 0x7E57_5C03usize, 0x7E57_5CA0usize);
        let (first, used) = unsafe { super::scratch_for(&dev, a, alloc, MB, 4096) }.unwrap();
        assert!(!used);
        super::list_follows(a, b);
        let (second, used) = unsafe { super::scratch_for(&dev, b, alloc, MB, 4096) }.unwrap();
        assert!(used && std::sync::Arc::ptr_eq(&first, &second), "the list after it made a scratch of its own");
        let (third, _) = unsafe { super::scratch_for(&dev, c, alloc, MB, 4096) }.unwrap();
        assert!(!std::sync::Arc::ptr_eq(&first, &third));
        for list in [a, b, c] {
            super::list_reset(list);
        }
    }

    #[test]
    fn a_chains_scratch_is_bounded_by_what_all_recordings_hold() {
        let group = 187 << 10;
        // the first of a recording: as much as it needs, up to the budget of one dispatch
        assert_eq!(scratch_bytes(15 * MB, group, 64 * MB, None, 0, 384 * MB), (15 * MB, true));
        assert_eq!(scratch_bytes(900 * MB, group, 64 * MB, None, 0, 384 * MB), (64 * MB, true));
        // the others hold nearly all: what is left, and at least LEAST
        assert_eq!(scratch_bytes(900 * MB, group, 64 * MB, None, 364 * MB, 384 * MB), (20 * MB, true));
        assert_eq!(scratch_bytes(900 * MB, group, 64 * MB, None, 1200 * MB, 384 * MB), (LEAST, true));
        // the list's own buffer: kept when big enough, or when no more may be made
        assert_eq!(scratch_bytes(8 * MB, group, 64 * MB, Some(16 * MB), 100 * MB, 384 * MB), (16 * MB, false));
        assert_eq!(scratch_bytes(60 * MB, group, 64 * MB, Some(16 * MB), 380 * MB, 384 * MB), (16 * MB, false));
        // … but made again, larger, while there is room
        assert_eq!(scratch_bytes(60 * MB, group, 64 * MB, Some(16 * MB), 100 * MB, 384 * MB), (60 * MB, true));
        // one group always fits, whatever the bounds say
        assert_eq!(scratch_bytes(9 * MB, 9 * MB, 4 * MB, None, 400 * MB, 384 * MB), (9 * MB, true));
        assert_eq!(scratch_bytes(9 * MB, 9 * MB, 4 * MB, Some(MB), 400 * MB, 384 * MB), (9 * MB, true));
    }
}
