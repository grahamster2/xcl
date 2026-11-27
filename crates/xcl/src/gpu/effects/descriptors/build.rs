//! The analysis of `descriptors`: a forward walk of the scalar registers over the program's
//! control flow, each register an expression (`Node`) met at every join (two different ones:
//! unknown), until nothing changes.

use std::collections::HashMap;

use super::{Base, Id, Kind, Load, Node, Program, Use, MAX_NODES, SGPRS, UNKNOWN};
use crate::gpu::rdna2;

struct Builder {
    p: Program,
    interned: HashMap<Node, Id>,
    /// loads by pc (`Program::loads` index)
    load_at: HashMap<u32, u32>,
    /// use sites (pc, which operand) → the use, met over visits
    use_at: HashMap<(u32, u8), Use>,
}

impl Builder {
    fn new() -> Builder {
        let mut b = Builder { p: Program::default(), interned: HashMap::new(), load_at: HashMap::new(), use_at: HashMap::new() };
        b.p.nodes.push(Node::Unknown);
        b.interned.insert(Node::Unknown, UNKNOWN);
        b
    }

    fn cnst(&self, id: Id) -> Option<u32> {
        match self.p.node(id) {
            Node::Const(c) => Some(c),
            _ => None,
        }
    }

    /// The expression `n`, folded (unknown in, unknown out).
    fn mk(&mut self, n: Node) -> Id {
        let c = |b: &Builder, id| b.cnst(id);
        let n = match n {
            Node::Add(a, b) | Node::Or(a, b) if a == UNKNOWN || b == UNKNOWN => return UNKNOWN,
            Node::Mul(a, _) | Node::Shl(a, _) | Node::Shr(a, _) | Node::And(a, _) if a == UNKNOWN => return UNKNOWN,
            Node::Add(a, b) => match (c(self, a), c(self, b)) {
                (Some(x), Some(y)) => Node::Const(x.wrapping_add(y)),
                (_, Some(0)) => return a,
                (Some(0), _) => return b,
                (Some(_), None) => Node::Add(b, a),
                (None, Some(y)) => match self.p.node(a) {
                    // (x + c1) + c2 = x + (c1 + c2)
                    Node::Add(x, k) if c(self, k).is_some() => {
                        let k = self.mk(Node::Const(c(self, k).unwrap().wrapping_add(y)));
                        return self.mk(Node::Add(x, k));
                    }
                    _ => n,
                },
                _ => Node::Add(a.min(b), a.max(b)),
            },
            Node::Or(a, b) => match (c(self, a), c(self, b)) {
                (Some(x), Some(y)) => Node::Const(x | y),
                (_, Some(0)) => return a,
                (Some(0), _) => return b,
                _ => Node::Or(a.min(b), a.max(b)),
            },
            Node::Mul(a, k) => match c(self, a) {
                Some(x) => Node::Const(x.wrapping_mul(k)),
                None if k == 1 => return a,
                None => n,
            },
            Node::Shl(a, k) => match c(self, a) {
                Some(x) => Node::Const(x << (k & 31)),
                None if k == 0 => return a,
                None => n,
            },
            Node::Shr(a, k) => match c(self, a) {
                Some(x) => Node::Const(x >> (k & 31)),
                None if k == 0 => return a,
                None => n,
            },
            Node::And(a, k) => match c(self, a) {
                Some(x) => Node::Const(x & k),
                None if k == u32::MAX => return a,
                None => n,
            },
            n => n,
        };
        if let Some(&id) = self.interned.get(&n) {
            return id;
        }
        if self.p.nodes.len() >= MAX_NODES {
            return UNKNOWN;
        }
        let id = self.p.nodes.len() as Id;
        self.p.nodes.push(n);
        self.interned.insert(n, id);
        id
    }

    /// A 32-bit scalar source operand (an SGPR, an inline constant or the literal).
    fn src(&mut self, s: &[Id], d: &rdna2::Decoded, field: &str) -> Id {
        let Some(code) = rdna2::field_value(d.enc, &d.words, field) else { return UNKNOWN };
        let c = match code {
            0..=105 => return s[code as usize],
            125 => 0,
            128..=192 => (code - 128) as u32,
            193..=208 => (-((code - 192) as i32)) as u32,
            240 => 0x3F00_0000,
            241 => 0xBF00_0000,
            242 => 0x3F80_0000,
            243 => 0xBF80_0000,
            244 => 0x4000_0000,
            245 => 0xC000_0000,
            246 => 0x4080_0000,
            247 => 0xC080_0000,
            255 => match d.literal {
                Some(l) => l,
                None => return UNKNOWN,
            },
            _ => return UNKNOWN,
        };
        self.mk(Node::Const(c))
    }

    /// The high half of a 64-bit scalar source whose low half `src` gave (an SGPR pair: the next
    /// register; an inline constant: its sign; a literal: not known).
    fn src_hi(&mut self, s: &[Id], d: &rdna2::Decoded, field: &str) -> Id {
        match rdna2::field_value(d.enc, &d.words, field) {
            Some(code @ 0..=104) => s[code as usize + 1],
            Some(125 | 128..=192) => self.mk(Node::Const(0)),
            Some(193..=208) => self.mk(Node::Const(u32::MAX)),
            _ => UNKNOWN,
        }
    }

    fn load_index(&mut self, pc: u32, l: Load) -> u32 {
        if let Some(&i) = self.load_at.get(&pc) {
            // met with what another path (or pass) gave
            let old = &mut self.p.loads[i as usize];
            let meet = |a: Id, b: Id| if a == b { a } else { UNKNOWN };
            old.base = match (old.base, l.base) {
                (Base::Pointer { lo, hi }, Base::Pointer { lo: l2, hi: h2 }) => Base::Pointer { lo: meet(lo, l2), hi: meet(hi, h2) },
                (Base::Buffer { w0, w1, records }, Base::Buffer { w0: a, w1: b, records: r }) => Base::Buffer { w0: meet(w0, a), w1: meet(w1, b), records: meet(records, r) },
                _ => Base::Pointer { lo: UNKNOWN, hi: UNKNOWN },
            };
            old.soff = meet(old.soff, l.soff);
            return i;
        }
        let i = self.p.loads.len() as u32;
        self.p.loads.push(l);
        self.load_at.insert(pc, i);
        i
    }

    fn note_use(&mut self, s: &[Id], pc: u32, slot: u8, kind: Kind, first: u64, n: usize) {
        if first as usize + n > SGPRS {
            return;
        }
        let words: Vec<Id> = s[first as usize..first as usize + n].to_vec();
        match self.use_at.get_mut(&(pc, slot)) {
            Some(u) => {
                for (o, w) in u.words.iter_mut().zip(words) {
                    if *o != w {
                        *o = UNKNOWN;
                    }
                }
            }
            None => {
                self.use_at.insert((pc, slot), Use { kind, words });
            }
        }
    }

    /// One instruction on the register state `s`.
    fn step(&mut self, s: &mut [Id], d: &rdna2::Decoded) {
        let f = |name: &str| rdna2::field_value(d.enc, &d.words, name);
        let pc = d.pc as u32;
        let name = d.inst.name;
        match d.enc.name {
            "ENC_MIMG" => {
                let r128 = f("R128").is_some_and(|v| v != 0) || name.starts_with("IMAGE_BVH");
                if let Some(r) = f("SRSRC") {
                    self.note_use(s, pc, 0, Kind::Image, r * 4, if r128 { 4 } else { 8 });
                }
                if name.contains("SAMPLE") || name.contains("GATHER") || name.contains("GET_LOD") {
                    if let Some(r) = f("SSAMP") {
                        self.note_use(s, pc, 1, Kind::Sampler, r * 4, 4);
                    }
                }
            }
            "ENC_MUBUF" | "ENC_MTBUF" => {
                if let Some(r) = f("SRSRC") {
                    self.note_use(s, pc, 0, Kind::Buffer, r * 4, 4);
                }
            }
            _ => {}
        }
        let dst = f("SDST").filter(|&r| (r as usize) < SGPRS);
        let set = |s: &mut [Id], r: Option<u64>, v: Id| {
            if let Some(r) = r.filter(|&r| (r as usize) < SGPRS) {
                s[r as usize] = v;
            }
        };
        let modeled = match name {
            "S_MOV_B32" => {
                let v = self.src(s, d, "SSRC0");
                set(s, dst, v);
                true
            }
            "S_MOV_B64" => {
                let (lo, hi) = (self.src(s, d, "SSRC0"), self.src_hi(s, d, "SSRC0"));
                set(s, dst, lo);
                set(s, dst.map(|r| r + 1), hi);
                true
            }
            "S_BREV_B32" | "S_NOT_B32" => {
                let a = self.src(s, d, "SSRC0");
                let v = match self.cnst(a) {
                    Some(c) if name == "S_BREV_B32" => self.mk(Node::Const(c.reverse_bits())),
                    Some(c) => self.mk(Node::Const(!c)),
                    None => UNKNOWN,
                };
                set(s, dst, v);
                true
            }
            "S_MOVK_I32" => {
                let v = self.mk(Node::Const(f("SIMM16").unwrap_or(0) as u16 as i16 as i32 as u32));
                set(s, dst, v);
                true
            }
            "S_CMOV_B32" => {
                let v = self.src(s, d, "SSRC0");
                if let Some(r) = dst {
                    if s[r as usize] != v {
                        s[r as usize] = UNKNOWN;
                    }
                }
                true
            }
            "S_ADD_U32" | "S_ADD_I32" | "S_SUB_U32" | "S_SUB_I32" | "S_ADDC_U32" | "S_OR_B32" | "S_AND_B32" | "S_MUL_I32" | "S_LSHL_B32" | "S_LSHR_B32"
            | "S_LSHL1_ADD_U32" | "S_LSHL2_ADD_U32" | "S_LSHL3_ADD_U32" | "S_LSHL4_ADD_U32" => {
                let (a, b) = (self.src(s, d, "SSRC0"), self.src(s, d, "SSRC1"));
                let (ca, cb) = (self.cnst(a), self.cnst(b));
                let v = match name {
                    "S_ADD_U32" | "S_ADD_I32" => self.mk(Node::Add(a, b)),
                    // the carry into the high half is not followed: XCL's console addresses of
                    // descriptors are 32-bit, and a root view's offsets stay in its 4 GiB
                    "S_ADDC_U32" => self.mk(Node::Add(a, b)),
                    "S_SUB_U32" | "S_SUB_I32" => match cb {
                        Some(c) => {
                            let n = self.mk(Node::Const(c.wrapping_neg()));
                            self.mk(Node::Add(a, n))
                        }
                        None => UNKNOWN,
                    },
                    "S_OR_B32" => self.mk(Node::Or(a, b)),
                    "S_AND_B32" => match (ca, cb) {
                        (_, Some(c)) => self.mk(Node::And(a, c)),
                        (Some(c), _) => self.mk(Node::And(b, c)),
                        _ => UNKNOWN,
                    },
                    "S_MUL_I32" => match (ca, cb) {
                        (_, Some(c)) => self.mk(Node::Mul(a, c)),
                        (Some(c), _) => self.mk(Node::Mul(b, c)),
                        _ => UNKNOWN,
                    },
                    "S_LSHL_B32" => cb.map_or(UNKNOWN, |c| self.mk(Node::Shl(a, (c & 31) as u8))),
                    "S_LSHR_B32" => cb.map_or(UNKNOWN, |c| self.mk(Node::Shr(a, (c & 31) as u8))),
                    _ => {
                        let k = name.as_bytes()[6] - b'0';
                        let sh = self.mk(Node::Shl(a, k));
                        self.mk(Node::Add(sh, b))
                    }
                };
                set(s, dst, v);
                true
            }
            n if d.enc.name == "ENC_SMEM" && (n.starts_with("S_LOAD_DWORD") || n.starts_with("S_BUFFER_LOAD_DWORD")) => {
                self.smem(s, d);
                true
            }
            _ => false,
        };
        if !modeled {
            for w in rdna2::sgpr_uses(d).1 {
                if (w as usize) < SGPRS {
                    s[w as usize] = UNKNOWN;
                }
            }
        }
        // a call returns with any register changed
        if matches!(name, "S_CALL_B64" | "S_SWAPPC_B64") {
            s.iter_mut().for_each(|v| *v = UNKNOWN);
        }
    }

    fn smem(&mut self, s: &mut [Id], d: &rdna2::Decoded) {
        let f = |name: &str| rdna2::field_value(d.enc, &d.words, name);
        let name = d.inst.name;
        let dwords: u8 = match name.rsplit('_').next() {
            Some("DWORD") => 1,
            Some("DWORDX2") => 2,
            Some("DWORDX4") => 4,
            Some("DWORDX8") => 8,
            Some("DWORDX16") => 16,
            _ => 0,
        };
        let (Some(sdata), Some(sbase)) = (f("SDATA"), f("SBASE")) else { return };
        let sbase = sbase * 2;
        let imm = f("OFFSET").unwrap_or(0) as u32;
        let imm = ((imm << 11) as i32) >> 11; // 21-bit signed
        let soff = match f("SOFFSET") {
            None | Some(125) => self.mk(Node::Const(0)),
            Some(r @ 0..=105) => s[r as usize],
            Some(_) => UNKNOWN,
        };
        let reg = |i: u64| if ((sbase + i) as usize) < SGPRS { s[(sbase + i) as usize] } else { UNKNOWN };
        let base = if name.starts_with("S_BUFFER") {
            self.note_use(s, d.pc as u32, 0, Kind::Buffer, sbase, 4);
            Base::Buffer { w0: reg(0), w1: reg(1), records: reg(2) }
        } else {
            Base::Pointer { lo: reg(0), hi: reg(1) }
        };
        let load = self.load_index(d.pc as u32, Load { pc: d.pc as u32, base, imm, soff, dwords });
        for k in 0..dwords as u64 {
            if ((sdata + k) as usize) < SGPRS {
                s[(sdata + k) as usize] = self.mk(Node::Load { load, k: k as u8 });
            }
        }
    }
}

/// Where control goes after `d` (at index `i` of `insts`): the indexes of the next instructions.
fn successors(insts: &[rdna2::Decoded], at: &HashMap<u64, usize>, i: usize) -> Vec<usize> {
    let d = &insts[i];
    let flags = d.inst.flags;
    let next = (i + 1 < insts.len()).then_some(i + 1);
    let target = rdna2::branch_target(d).and_then(|t| at.get(&t).copied());
    if flags & rdna2::F_TERMINATOR != 0 && flags & rdna2::F_BRANCH == 0 {
        return Vec::new();
    }
    let call = matches!(d.inst.name, "S_CALL_B64" | "S_SWAPPC_B64");
    match (flags & rdna2::F_BRANCH != 0 || d.far_target.is_some(), flags & rdna2::F_CONDITIONAL != 0 || call) {
        (false, _) => next.into_iter().collect(),
        (true, false) => target.into_iter().collect(),
        (true, true) => target.into_iter().chain(next).collect(),
    }
}

/// The descriptor analysis of the RDNA2 program `code`, the user-data dword d at entry in SGPR
/// `entry[d]` (`xbox::user_sgpr`). None when the code cannot be read (`effects::decode_program`).
pub fn analyze(code: &[u8], entry: &[u32]) -> Option<Program> {
    let insts = crate::gpu::effects::decode_program(code)?;
    let at: HashMap<u64, usize> = insts.iter().enumerate().map(|(i, d)| (d.pc, i)).collect();
    // basic blocks: a leader at the start, at every branch target and after every branch
    let mut leader = vec![false; insts.len()];
    leader[0] = true;
    for i in 0..insts.len() {
        let succ = successors(&insts, &at, i);
        if succ != [i + 1] {
            succ.iter().for_each(|&j| leader[j] = true);
            if i + 1 < insts.len() {
                leader[i + 1] = true;
            }
        }
    }
    let mut b = Builder::new();
    let mut start = vec![UNKNOWN; SGPRS];
    for (dw, &r) in entry.iter().enumerate() {
        if (r as usize) < SGPRS {
            start[r as usize] = b.mk(Node::Entry(dw as u8));
        }
    }
    let mut state: Vec<Option<Vec<Id>>> = vec![None; insts.len()];
    state[0] = Some(start);
    let mut work = vec![0usize];
    let mut queued = vec![false; insts.len()];
    queued[0] = true;
    while let Some(first) = work.pop() {
        queued[first] = false;
        let mut s = state[first].clone()?;
        let mut i = first;
        loop {
            b.step(&mut s, &insts[i]);
            let succ = successors(&insts, &at, i);
            if succ.len() == 1 && succ[0] == i + 1 && !leader[i + 1] {
                i += 1;
                continue;
            }
            for j in succ {
                let changed = match &mut state[j] {
                    None => {
                        state[j] = Some(s.clone());
                        true
                    }
                    Some(old) => {
                        let mut changed = false;
                        for (o, &n) in old.iter_mut().zip(&s) {
                            if *o != n && *o != UNKNOWN {
                                *o = UNKNOWN;
                                changed = true;
                            }
                        }
                        changed
                    }
                };
                if changed && !queued[j] {
                    queued[j] = true;
                    work.push(j);
                }
            }
            break;
        }
    }
    let mut uses: Vec<((u32, u8), Use)> = b.use_at.into_iter().collect();
    uses.sort_by_key(|(k, _)| *k);
    let mut p = b.p;
    for (_, u) in uses {
        if !p.uses.contains(&u) {
            p.uses.push(u);
        }
    }
    Some(p)
}

