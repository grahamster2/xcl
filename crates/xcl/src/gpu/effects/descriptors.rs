//! The descriptors a console program reads, from its code alone (`XCL_HOT_RESIDENCY`).
//!
//! On the console a descriptor is memory: a T# (image, 8 dwords), S# (sampler, 4) or V# (buffer,
//! 4) is loaded into SGPRs with `s_load_dword*` from a table or heap whose address the program
//! got in a user SGPR (a root argument), or with `s_buffer_load_dword*` from a constant buffer,
//! and then named as the SRSRC / SSAMP of an image or buffer access or the SBASE of another
//! `s_buffer_load`. XCL's residency tracking cannot see which descriptors a draw reads: a
//! descriptor table is only an address and bindless titles index one shared heap. So the code is
//! read here once per shader: a forward analysis of the scalar registers (each holds an
//! expression of the user SGPRs at entry, constants and dwords of earlier scalar loads, or is
//! unknown – a value from a vector register, a loop, the hardware) gives, for each register range
//! used as a descriptor, the expression of each of its dwords. At a draw or dispatch the
//! expressions are evaluated with the root values in use (`Program::reads`), reading the memory
//! the loads read: the descriptor words the program will read, each T# word 0 the index in the
//! shared PC heap of its view (`hle::d3d12x::console_heap`).
//!
//! Nothing here changes a translated shader or its cache key (`build.rs` does not hash this file).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use crate::gpu::{recomp, xbox};

mod build;
#[cfg(test)]
mod tests;

pub use build::analyze;

/// an expression (index into `Program::nodes`); 0 is `Node::Unknown`
pub type Id = u32;
const UNKNOWN: Id = 0;
/// scalar registers followed: s0 – s105 (VCC, M0, EXEC and the rest are never descriptor sources)
const SGPRS: usize = 106;
/// expressions a program may make (a bigger one: what is past it is unknown)
const MAX_NODES: usize = 1 << 16;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Node {
    Unknown,
    Const(u32),
    /// user-data dword `d` at entry (a root argument, the spilled-argument table address in
    /// dwords 0 – 1, the heap addresses after the arguments)
    Entry(u8),
    /// dword `k` of the scalar load `load` (`Program::loads`)
    Load { load: u32, k: u8 },
    Add(Id, Id),
    Mul(Id, u32),
    Shl(Id, u8),
    Shr(Id, u8),
    And(Id, u32),
    Or(Id, Id),
}

/// Where a scalar load reads.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Base {
    /// `s_load_dword*`: the 64-bit address in SBASE
    Pointer { lo: Id, hi: Id },
    /// `s_buffer_load_dword*`: the V# in SBASE (its words 0, 1 and 2: address, num_records)
    Buffer { w0: Id, w1: Id, records: Id },
}

/// One `s_load_dword*` / `s_buffer_load_dword*` (its address met over every path to it).
#[derive(Clone, Copy, Debug)]
pub struct Load {
    pub pc: u32,
    pub base: Base,
    /// OFFSET (signed) plus SOFFSET
    pub imm: i32,
    pub soff: Id,
    pub dwords: u8,
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub enum Kind {
    /// T#: SRSRC of an image access (8 dwords; 4 with R128 and for BVH descriptors)
    Image,
    /// S#: SSAMP of a sampling image access
    Sampler,
    /// V#: SRSRC of a buffer access, SBASE of an `s_buffer_load`
    Buffer,
}

/// A register range the program uses as a descriptor: the expression of each of its dwords.
#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct Use {
    pub kind: Kind,
    pub words: Vec<Id>,
}

/// How a descriptor's words are reached.
#[derive(Clone, Copy, PartialEq, Eq, Debug, PartialOrd, Ord)]
pub enum Class {
    /// built in registers from root values and constants (no descriptor load)
    Inline,
    /// loaded at a constant offset from a root pointer (or a pointer loaded at constant offsets)
    Constant,
    /// at an offset read from memory (an index in a constant buffer): known at the draw
    Indexed,
    /// from a value the scalar code cannot know (a vector register, a loop): unknown
    Dynamic,
}

/// The analysis of one program.
#[derive(Debug, Default)]
pub struct Program {
    pub nodes: Vec<Node>,
    pub loads: Vec<Load>,
    pub uses: Vec<Use>,
}

/// Counts of a program's (or shader's) descriptor uses by `Class`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Census {
    pub inline: usize,
    pub constant: usize,
    pub indexed: usize,
    pub dynamic: usize,
}

impl Census {
    pub fn add(&mut self, o: Census) {
        self.inline += o.inline;
        self.constant += o.constant;
        self.indexed += o.indexed;
        self.dynamic += o.dynamic;
    }
}

/// The root values of one stage at a draw or dispatch: the user-data dwords (index d = dword d,
/// as `xbox::user_sgpr` numbers them: for compute, VS and PS XCL's user SGPR vector itself, for an
/// NGG stage that vector from s8) and the bytes of the spilled-argument table their dwords 0 – 1
/// point at (read from here, not from memory: it may not be written yet).
#[derive(Clone, Copy, Debug, Default)]
pub struct UserData<'a> {
    pub dwords: &'a [u32],
    pub table: &'a [u8],
}

/// A descriptor a program reads: its kind and words (`dwords` of `words` are valid).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct DescRead {
    pub kind: Kind,
    pub dwords: u8,
    pub words: [u32; 8],
    /// console address it was loaded from (0: built in registers)
    pub addr: u64,
}

impl Program {
    fn node(&self, id: Id) -> Node {
        self.nodes.get(id as usize).copied().unwrap_or(Node::Unknown)
    }

    /// The class of each use, counted.
    pub fn census(&self) -> Census {
        let mut c = Census::default();
        let mut memo = HashMap::new();
        for u in &self.uses {
            match self.use_class(u, &mut memo) {
                Class::Inline => c.inline += 1,
                Class::Constant => c.constant += 1,
                Class::Indexed => c.indexed += 1,
                Class::Dynamic => c.dynamic += 1,
            }
        }
        c
    }

    /// Whether some descriptor of the program is indexed by a value unknown before it runs.
    pub fn dynamic(&self) -> bool {
        self.census().dynamic > 0
    }

    pub fn use_class(&self, u: &Use, memo: &mut HashMap<u32, Class>) -> Class {
        // a static sampler: built in registers from constants (some not followed: `s_brev_b32`)
        if u.kind == Kind::Sampler && u.words.iter().all(|&w| !matches!(self.node(w), Node::Load { .. })) {
            return Class::Inline;
        }
        u.words.iter().map(|&w| self.word_class(w, memo)).max().unwrap_or(Class::Dynamic)
    }

    /// A descriptor word: `Inline` without loads, else the class of the load it comes from.
    fn word_class(&self, w: Id, memo: &mut HashMap<u32, Class>) -> Class {
        match self.node(w) {
            Node::Load { load, .. } => self.load_class(load, memo, 0),
            _ => self.value_class(w, memo, 0).max(Class::Inline),
        }
    }

    /// An address or offset term: `Inline` (root values, constants), `Indexed` (a loaded value in
    /// it), `Dynamic` (an unknown in it); a pointer loaded from memory takes its load's class.
    fn value_class(&self, id: Id, memo: &mut HashMap<u32, Class>, depth: u32) -> Class {
        if depth > 32 {
            return Class::Dynamic;
        }
        match self.node(id) {
            Node::Unknown => Class::Dynamic,
            Node::Const(_) | Node::Entry(_) => Class::Inline,
            Node::Load { load, .. } => self.load_class(load, memo, depth + 1).max(Class::Indexed),
            Node::Add(a, b) | Node::Or(a, b) => self.value_class(a, memo, depth + 1).max(self.value_class(b, memo, depth + 1)),
            Node::Mul(a, _) | Node::Shl(a, _) | Node::Shr(a, _) | Node::And(a, _) => self.value_class(a, memo, depth + 1),
        }
    }

    /// A pointer: loaded whole from memory (a table address in the spilled-argument table, a heap
    /// address in a constant buffer) it is as good as its load; plus a constant it stays so.
    fn pointer_class(&self, id: Id, memo: &mut HashMap<u32, Class>, depth: u32) -> Class {
        match self.node(id) {
            Node::Load { load, .. } => self.load_class(load, memo, depth + 1),
            Node::Add(a, b) if matches!(self.node(b), Node::Const(_)) => self.pointer_class(a, memo, depth + 1),
            _ => self.value_class(id, memo, depth + 1),
        }
    }

    fn load_class(&self, load: u32, memo: &mut HashMap<u32, Class>, depth: u32) -> Class {
        if let Some(&c) = memo.get(&load) {
            return c;
        }
        let Some(l) = self.loads.get(load as usize) else { return Class::Dynamic };
        if depth > 32 {
            return Class::Dynamic;
        }
        let (lo, hi) = match l.base {
            Base::Pointer { lo, hi } => (lo, hi),
            Base::Buffer { w0, w1, .. } => (w0, w1),
        };
        let mut c = self.pointer_class(lo, memo, depth).max(Class::Constant);
        if self.node(hi) == Node::Unknown {
            c = Class::Dynamic;
        }
        if !matches!(self.node(l.soff), Node::Const(_)) {
            c = c.max(self.value_class(l.soff, memo, depth + 1).max(Class::Indexed));
        }
        memo.insert(load, c);
        c
    }

    /// The descriptors the program reads with the root values `roots`, the memory its loads read
    /// from `read` (console address, the dwords to fill; false: not readable). A descriptor
    /// whose words cannot all be known is left out.
    pub fn reads(&self, roots: &UserData, read: &mut dyn FnMut(u64, &mut [u32]) -> bool) -> Vec<DescRead> {
        let mut ev = Eval { p: self, roots, read, loads: HashMap::new() };
        let mut out: Vec<DescRead> = Vec::new();
        for u in &self.uses {
            let mut words = [0u32; 8];
            let mut ok = true;
            for (i, &w) in u.words.iter().enumerate().take(8) {
                match ev.value(w, 0) {
                    Some(v) => words[i] = v,
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            let addr = match self.node(u.words[0]) {
                Node::Load { load, k } => ev.address(load, 0).map_or(0, |a| a.wrapping_add(4 * k as u64)),
                _ => 0,
            };
            let r = DescRead { kind: u.kind, dwords: u.words.len() as u8, words, addr };
            if !out.contains(&r) {
                out.push(r);
            }
        }
        out
    }
}

struct Eval<'a, 'r> {
    p: &'a Program,
    roots: &'a UserData<'a>,
    read: &'r mut dyn FnMut(u64, &mut [u32]) -> bool,
    /// load → its dwords (None: not readable)
    loads: HashMap<u32, Option<[u32; 16]>>,
}

impl Eval<'_, '_> {
    fn value(&mut self, id: Id, depth: u32) -> Option<u32> {
        if depth > 64 {
            return None;
        }
        let d = depth + 1;
        Some(match self.p.node(id) {
            Node::Unknown => return None,
            Node::Const(c) => c,
            Node::Entry(e) => *self.roots.dwords.get(e as usize)?,
            Node::Load { load, k } => self.load(load, depth)?[k as usize & 15],
            Node::Add(a, b) => self.value(a, d)?.wrapping_add(self.value(b, d)?),
            Node::Or(a, b) => self.value(a, d)? | self.value(b, d)?,
            Node::Mul(a, c) => self.value(a, d)?.wrapping_mul(c),
            Node::Shl(a, c) => self.value(a, d)? << (c & 31),
            Node::Shr(a, c) => self.value(a, d)? >> (c & 31),
            Node::And(a, c) => self.value(a, d)? & c,
        })
    }

    /// The console address load `load` reads (the dword-aligned start).
    fn address(&mut self, load: u32, depth: u32) -> Option<u64> {
        let l = *self.p.loads.get(load as usize)?;
        let d = depth + 1;
        let soff = self.value(l.soff, d)?;
        let base = match l.base {
            Base::Pointer { lo, hi } => (self.value(hi, d)? as u64) << 32 | self.value(lo, d)? as u64,
            Base::Buffer { w0, w1, records } => {
                // past num_records (a constant buffer's: bytes, stride 0) the hardware reads 0
                let at = (l.imm as u32).wrapping_add(soff);
                if at >= self.value(records, d)? {
                    return Some(u64::MAX);
                }
                ((self.value(w1, d)? & 0xFFFF) as u64) << 32 | self.value(w0, d)? as u64
            }
        };
        Some(base.wrapping_add(l.imm as i64 as u64).wrapping_add(soff as u64) & !3)
    }

    fn load(&mut self, load: u32, depth: u32) -> Option<[u32; 16]> {
        if let Some(v) = self.loads.get(&load) {
            return *v;
        }
        let v = self.fetch(load, depth);
        self.loads.insert(load, v);
        v
    }

    fn fetch(&mut self, load: u32, depth: u32) -> Option<[u32; 16]> {
        let n = (self.p.loads.get(load as usize)?.dwords as usize).min(16);
        let at = self.address(load, depth)?;
        let mut w = [0u32; 16];
        if at == u64::MAX {
            return Some(w);
        }
        // the spilled-argument table: from the bytes given, not memory
        let table = match self.roots.dwords {
            [lo, hi, ..] => (*hi as u64) << 32 | *lo as u64,
            _ => 0,
        };
        let t = self.roots.table;
        if table != 0 && at >= table && at - table < t.len() as u64 {
            let o = (at - table) as usize;
            for (i, x) in w.iter_mut().enumerate().take(n) {
                *x = t.get(o + 4 * i..o + 4 * i + 4).map_or(0, |b| u32::from_le_bytes(b.try_into().unwrap()));
            }
            return Some(w);
        }
        (self.read)(at, &mut w[..n]).then_some(w)
    }
}

// ---------------------------------------------------------------------------------------
// shaders
// ---------------------------------------------------------------------------------------

/// The programs of one hardware stage of a shader.
#[derive(Debug)]
pub struct Stage {
    /// PAL hardware stage (`xbox::hw_stage_name`; 6: CS, 7: NGG)
    pub hw: u8,
    pub programs: Vec<Program>,
}

/// The descriptor analysis of a console shader: one `Stage` for the first FSSM of each hardware
/// stage it has (the variant XCL runs of that stage).
#[derive(Debug, Default)]
pub struct Shader {
    pub stages: Vec<Stage>,
}

impl Shader {
    pub fn census(&self) -> Census {
        let mut c = Census::default();
        self.stages.iter().flat_map(|s| &s.programs).for_each(|p| c.add(p.census()));
        c
    }

    pub fn dynamic(&self) -> bool {
        self.census().dynamic > 0
    }
}

/// User-data dwords of a graphics stage: what its root layout names (the spilled-argument table
/// in dwords 0 – 1 always), all 32 when the layout cannot be read.
fn user_dwords(f: &xbox::Fssm, stage: &xbox::HwStage) -> u32 {
    let Ok(l) = xbox::root_layout(&f.data, stage) else { return 32 };
    let ends = l.slots.iter().map(|s| s.dword as u32 + s.sgprs.max(1)).chain([l.view_heap, l.sampler_heap].into_iter().flatten().map(|d| d as u32 + 1));
    ends.max().unwrap_or(0).max(2)
}

/// The analysis of the console shader `blob` (compute, one graphics stage, or an FSPX driver
/// container). None when no program can be read.
pub fn of_shader(blob: &[u8]) -> Option<Shader> {
    if let Some(f) = recomp::pso::fspx(blob) {
        let n = recomp::pso::compute_interface(blob).map_or(16, |c| c.user_sgprs);
        let p = analyze(f.code, &(0..n).collect::<Vec<_>>())?;
        return Some(Shader { stages: vec![Stage { hw: 6, programs: vec![p] }] });
    }
    let sh = xbox::parse(blob)?;
    let mut out = Shader::default();
    for f in &sh.fssm {
        let Some(stage) = f.stages.first() else { continue };
        if f.error.is_some() || out.stages.iter().any(|s| s.hw == stage.hw) {
            continue;
        }
        let n = if stage.hw == 6 { recomp::CsInterface::from_descriptor(&f.data, stage.offset).map_or(16, |c| c.user_sgprs) } else { user_dwords(f, stage) };
        let entry: Vec<u32> = (0..n.min(64)).filter_map(|d| xbox::user_sgpr(stage.hw, d as u8)).collect();
        let programs: Vec<Program> = f.programs.iter().filter_map(|p| analyze(f.data.get(p.offset..p.offset.checked_add(p.len)?)?, &entry)).collect();
        if !programs.is_empty() {
            out.stages.push(Stage { hw: stage.hw, programs });
        }
    }
    (!out.stages.is_empty()).then_some(out)
}

/// `of_shader`, once per shader (by its bytes' hash).
pub fn cached(blob: &[u8]) -> Option<Arc<Shader>> {
    static CACHE: OnceLock<Mutex<HashMap<u64, Option<Arc<Shader>>>>> = OnceLock::new();
    let h = crate::gpu::shader::fnv64(blob);
    let cache = CACHE.get_or_init(Default::default);
    if let Some(s) = cache.lock().unwrap().get(&h) {
        return s.clone();
    }
    let s = of_shader(blob).map(Arc::new);
    cache.lock().unwrap().insert(h, s.clone());
    s
}
