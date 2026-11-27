//! The descriptors a draw or dispatch reads, and the PC resources and title memory behind them
//! (`XCL_HOT_RESIDENCY`): what `hot_residency` cannot see from the root arguments alone – a
//! descriptor table is only an address, and a bindless title indexes one shared heap.
//!
//! A pipeline's console shaders are analysed once each (`gpu::effects::descriptors`, cached by
//! the shader's bytes) when the pipeline is noted (`note_pipeline`). At a draw or dispatch
//! `read_descriptors` evaluates what the analysis found with the root values in use – reading the
//! descriptor tables, heaps and constant buffers in unified memory – and `resources_of` turns the
//! descriptors into PC resources (a T#'s word 0 is the index of its view in the shared heap,
//! `console_heap::named_by_index`) and title memory ranges (a V#'s address and size). The result
//! of a (pipeline, root values) is kept until the next present: a frame's draws repeat both.
//!
//! Nothing here changes a translated shader or the shader cache key.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::gpu::effects::descriptors::{self, DescRead, Kind, Shader, UserData};

#[cfg(test)]
mod tests;

/// The root values of a draw or dispatch, by the stages that take them (None: the pipeline has
/// no such stage, or they are not known). Each is `UserData`: dword d = user-data dword d.
#[derive(Clone, Copy, Debug, Default)]
pub struct Roots<'a> {
    /// a CS (`console_list::user_data_for` + `with_heaps`)
    pub cs: Option<UserData<'a>>,
    /// VS and PS (`console_list::user_data_gfx` + `with_heaps`)
    pub gfx: Option<UserData<'a>>,
    /// an NGG stage: a mesh shader's (`console_list::user_data_ms` from s8, where its dword 0 is)
    pub ms: Option<UserData<'a>>,
}

impl<'a> Roots<'a> {
    fn of_hw(&self, hw: u8) -> Option<&UserData<'a>> {
        match hw {
            6 => self.cs.as_ref(),
            4 | 5 => self.gfx.as_ref(),
            7 => self.ms.as_ref(),
            _ => None,
        }
    }

    /// Identity of the values (a frame's draws with the same ones read the same descriptors,
    /// unless the title rewrote a table or constant buffer in between: then the next frame sees it).
    fn key(&self) -> u64 {
        let mut h = 0xcbf2_9ce4_8422_2325u64;
        let mut eat = |b: &[u8]| {
            for &x in b {
                h ^= x as u64;
                h = h.wrapping_mul(0x100_0000_01b3);
            }
        };
        for u in [&self.cs, &self.gfx, &self.ms] {
            match u {
                None => eat(&[0]),
                Some(u) => {
                    eat(&[1]);
                    u.dwords.iter().for_each(|w| eat(&w.to_le_bytes()));
                    eat(&(u.dwords.len() as u32).to_le_bytes());
                    eat(u.table);
                    eat(&(u.table.len() as u32).to_le_bytes());
                }
            }
        }
        h
    }
}

/// pipeline → the analyses of its console shaders
static PIPELINES: super::sharded::Sharded<Arc<Vec<Arc<Shader>>>> = super::sharded::Sharded::new();

/// The pipeline `pso` was made from the console shaders `blobs` (compute: one; graphics: VS / MS
/// and PS). Each is analysed once (the analysis is cached by the shader's bytes). Meant for where
/// the translated pipeline is made (`real.rs`, next to `note_stores`).
pub fn note_pipeline(pso: usize, blobs: &[&[u8]]) {
    if pso == 0 {
        return;
    }
    let shaders: Vec<Arc<Shader>> = blobs.iter().filter_map(|b| descriptors::cached(b)).collect();
    if !shaders.is_empty() {
        PIPELINES.insert(pso, Arc::new(shaders));
    }
}

/// The pipeline is gone.
pub fn forget_pipeline(pso: usize) {
    PIPELINES.remove(pso);
}

/// (frame, (pipeline, roots) → reads)
type Cache = (u64, HashMap<(usize, u64), Arc<[DescRead]>>);
static CACHE: Mutex<Option<Cache>> = Mutex::new(None);
/// entries a frame keeps at most (past it: evaluated again, not kept)
const CACHE_MAX: usize = 1 << 14;

/// The descriptors the console shaders of `pso` (a stand-in resolved) read with the root values
/// `roots`, each once. Empty for a pipeline not noted, or nothing known. Kept for the frame.
pub fn read_descriptors(pso: usize, roots: &Roots) -> Arc<[DescRead]> {
    let pso = super::pso_async::resolve(pso);
    let Some(shaders) = PIPELINES.get(pso) else { return Arc::from([]) };
    let frame = super::present::presents();
    let key = (pso, roots.key());
    if let Some((f, m)) = CACHE.lock().unwrap().as_ref() {
        if *f == frame {
            if let Some(r) = m.get(&key) {
                return r.clone();
            }
        }
    }
    let r: Arc<[DescRead]> = reads_of(&shaders, roots, &mut read_console).into();
    let mut g = CACHE.lock().unwrap();
    let c = g.get_or_insert_with(|| (frame, HashMap::new()));
    if c.0 != frame {
        *c = (frame, HashMap::new());
    }
    if c.1.len() < CACHE_MAX {
        c.1.insert(key, r.clone());
    }
    r
}

/// `read_descriptors` without the registry and the cache: the reads of `shaders`, the memory
/// read by `read`.
pub fn reads_of(shaders: &[Arc<Shader>], roots: &Roots, read: &mut dyn FnMut(u64, &mut [u32]) -> bool) -> Vec<DescRead> {
    let mut out: Vec<DescRead> = Vec::new();
    for st in shaders.iter().flat_map(|s| &s.stages) {
        let Some(u) = roots.of_hw(st.hw) else { continue };
        for p in &st.programs {
            for r in p.reads(u, read) {
                if !out.contains(&r) {
                    out.push(r);
                }
            }
        }
    }
    out
}

/// The host address of console address `a`: one with the high half 0 is in window 0 (descriptor
/// tables and heaps: 32-bit addresses), the others are the address itself (`gpu::uma`; the
/// shaders map them so: `recomp::emit` `window_read`).
pub fn host_of(base: usize, a: u64) -> usize {
    if a >> 32 == 0 {
        base.wrapping_add(a as usize)
    } else {
        a as usize
    }
}

/// Read `out.len()` dwords at console address `a`: from a console descriptor heap's slots or a
/// committed window of unified memory (no other memory: a sparse window's tile may be unmapped).
fn read_console(a: u64, out: &mut [u32]) -> bool {
    let Some(u) = crate::gpu::uma::get() else { return false };
    let p = host_of(u.base(), a);
    let bytes = 4 * out.len();
    if p % 4 != 0 || bytes == 0 {
        return false;
    }
    let readable = u.in_committed_window(p, bytes) || {
        // heap slots (the heaps may be in sparse memory XCL keeps mapped): both ends in one
        a >> 32 == 0 && super::console_heap::slot(p as u64).is_some() && super::console_heap::slot((p + bytes - 4) as u64).is_some()
    };
    if !readable {
        return false;
    }
    for (i, o) in out.iter_mut().enumerate() {
        *o = unsafe { std::ptr::read_volatile((p + 4 * i) as *const u32) };
    }
    true
}

/// What a descriptor read touches.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Touch {
    /// a PC resource (a T#'s view in the shared heap)
    Res(usize),
    /// title memory at host address `va` (a V#'s range; a T# whose view is not known: its
    /// address)
    Va { va: u64, bytes: u64 },
}

/// most bytes a V# is taken to cover (a structured buffer of the whole heap says little)
pub const BUFFER_MAX: u64 = 64 << 20;

/// The range of the buffer V# `w` (host address, bytes; None: empty or none).
pub fn buffer_range(base: usize, w: &[u32; 8]) -> Option<(u64, u64)> {
    let va = (w[1] as u64 & 0xFFFF) << 32 | w[0] as u64;
    let stride = (w[1] >> 16) & 0x3FFF;
    let bytes = (w[2] as u64).saturating_mul(stride.max(1) as u64).min(BUFFER_MAX);
    (va != 0 && bytes != 0).then(|| (host_of(base, va) as u64, bytes))
}

/// Whether the T# `w` is an image (its type, word 3 [31:28], 8 or more; else a buffer type)
pub fn is_image(w: &[u32; 8]) -> bool {
    w[3] >> 28 >= 8
}

/// The PC resources and title memory behind `reads`, each once. `named`: the resource the
/// shared-heap index names (`console_heap::named_by_index`).
pub fn touches_of(reads: &[DescRead], base: usize, named: &dyn Fn(u32) -> usize) -> Vec<Touch> {
    let mut out = Vec::new();
    let push = |t: Touch, out: &mut Vec<Touch>| {
        if !out.contains(&t) {
            out.push(t);
        }
    };
    for r in reads {
        match r.kind {
            Kind::Sampler => {}
            Kind::Image if r.dwords == 8 && is_image(&r.words) => {
                // the SRV's index (even) or the UAV's next to it: both views of one resource
                let i = r.words[0];
                let res = [i, i ^ 1].into_iter().map(named).find(|&res| res != 0 && res != super::console_heap::GONE);
                if let Some(res) = res {
                    push(Touch::Res(res), &mut out);
                }
            }
            Kind::Image | Kind::Buffer => {
                if let Some((va, bytes)) = buffer_range(base, &r.words) {
                    push(Touch::Va { va, bytes }, &mut out);
                }
            }
        }
    }
    out
}

/// `touches_of` with XCL's shared heap and unified memory.
pub fn resources_of(reads: &[DescRead]) -> Vec<Touch> {
    let base = crate::gpu::uma::get().map_or(0, |u| u.base());
    touches_of(reads, base, &super::console_heap::named_by_index)
}

/// `XCL_DESC_CENSUS` style counts of what the noted pipelines' shaders read (for a log line).
pub fn census_of(pso: usize) -> Option<descriptors::Census> {
    let shaders = PIPELINES.get(super::pso_async::resolve(pso))?;
    let mut c = descriptors::Census::default();
    shaders.iter().for_each(|s| c.add(s.census()));
    Some(c)
}
