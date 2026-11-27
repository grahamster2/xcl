use super::*;
use crate::gpu::rdna2;

type I<'a> = (&'a str, &'a str, Vec<(&'a str, u64)>, Option<u32>);

fn asm(insts: &[I]) -> Vec<u8> {
    insts.iter().flat_map(|(n, e, f, l)| rdna2::encode(n, e, f, *l).unwrap_or_else(|e| panic!("{n}: {e}"))).collect()
}

fn mov(dst: u64, src: u64) -> I<'static> {
    ("S_MOV_B32", "ENC_SOP1", vec![("SDST", dst), ("SSRC0", src)], None)
}
fn movl(dst: u64, lit: u32) -> I<'static> {
    ("S_MOV_B32", "ENC_SOP1", vec![("SDST", dst), ("SSRC0", 255)], Some(lit))
}
/// `name` s[sdata..], s[2·sbase : 2·sbase + 1], OFFSET `off`, SOFFSET `soff` (125: none)
fn smem(name: &'static str, sdata: u64, sbase: u64, off: u64, soff: u64) -> I<'static> {
    (name, "ENC_SMEM", vec![("SDATA", sdata), ("SBASE", sbase), ("OFFSET", off), ("SOFFSET", soff)], None)
}
fn sample(srsrc: u64, ssamp: u64) -> I<'static> {
    ("IMAGE_SAMPLE", "ENC_MIMG", vec![("VDATA", 4), ("VADDR", 0), ("SRSRC", srsrc), ("SSAMP", ssamp), ("DMASK", 15), ("DIM", 1)], None)
}
fn image_load(srsrc: u64) -> I<'static> {
    ("IMAGE_LOAD", "ENC_MIMG", vec![("VDATA", 4), ("VADDR", 0), ("SRSRC", srsrc), ("DMASK", 15), ("DIM", 1)], None)
}
fn end() -> I<'static> {
    ("S_ENDPGM", "ENC_SOPP", vec![], None)
}

/// a compute program's entry: dword d in s(d), 16 of them
fn cs() -> Vec<u32> {
    (0..16).collect()
}

/// console memory: dwords by address
struct Mem(HashMap<u64, u32>);

impl Mem {
    fn new() -> Mem {
        Mem(HashMap::new())
    }
    fn put(&mut self, at: u64, words: &[u32]) {
        for (i, &w) in words.iter().enumerate() {
            self.0.insert(at + 4 * i as u64, w);
        }
    }
    fn reader(&self) -> impl FnMut(u64, &mut [u32]) -> bool + '_ {
        move |at, out| {
            for (i, o) in out.iter_mut().enumerate() {
                match self.0.get(&(at + 4 * i as u64)) {
                    Some(&w) => *o = w,
                    None => return false,
                }
            }
            true
        }
    }
}

const T: [u32; 8] = [42, 0, 0, 0x9000_0000, 0, 0, 0, 0];
const S: [u32; 4] = [1, 2, 3, 7];

fn roots(dw: &[u32]) -> UserData<'_> {
    UserData { dwords: dw, table: &[] }
}

#[test]
fn a_table_at_constant_offsets() {
    // s4: a descriptor table (32-bit, high half 0); T# at +0x40, S# at +0x80
    let p = analyze(&asm(&[mov(5, 128), smem("S_LOAD_DWORDX8", 8, 2, 0x40, 125), smem("S_LOAD_DWORDX4", 16, 2, 0x80, 125), sample(2, 4), end()]), &cs()).unwrap();
    assert_eq!(p.census(), Census { constant: 2, ..Default::default() });
    assert!(!p.dynamic());
    let mut m = Mem::new();
    m.put(0x1040, &T);
    m.put(0x1080, &S);
    let mut dw = vec![0u32; 16];
    dw[4] = 0x1000;
    let r = p.reads(&roots(&dw), &mut m.reader());
    assert_eq!(r.len(), 2);
    let t = r.iter().find(|r| r.kind == Kind::Image).unwrap();
    assert_eq!((t.addr, t.dwords, t.words), (0x1040, 8, T));
    let s = r.iter().find(|r| r.kind == Kind::Sampler).unwrap();
    assert_eq!((s.addr, &s.words[..4]), (0x1080, &S[..]));
    // another table address: other descriptors (none readable there)
    dw[4] = 0x2000;
    assert!(p.reads(&roots(&dw), &mut m.reader()).is_empty());
}

#[test]
fn a_heap_index_from_a_constant_buffer() {
    // s4: the view heap; s[6:7]: a root CBV whose dword at +0x10 is the texture's index
    let p = analyze(
        &asm(&[
            smem("S_LOAD_DWORD", 8, 3, 0x10, 125),
            ("S_LSHL_B32", "ENC_SOP2", vec![("SDST", 8), ("SSRC0", 8), ("SSRC1", 128 + 5)], None),
            mov(5, 128),
            smem("S_LOAD_DWORDX8", 12, 2, 0, 8),
            image_load(3),
            end(),
        ]),
        &cs(),
    )
    .unwrap();
    assert_eq!(p.census(), Census { indexed: 1, ..Default::default() });
    let mut m = Mem::new();
    m.put(0x7_0000_0010, &[3]);
    m.put(0x4000 + 3 * 32, &T);
    let mut dw = vec![0u32; 16];
    (dw[4], dw[6], dw[7]) = (0x4000, 0, 7);
    let r = p.reads(&roots(&dw), &mut m.reader());
    assert_eq!(r.iter().map(|r| (r.kind, r.addr, r.words[0])).collect::<Vec<_>>(), [(Kind::Image, 0x4060, 42)]);
}

#[test]
fn an_index_from_a_vector_register_is_dynamic() {
    let readfirstlane: I = ("V_READFIRSTLANE_B32", "ENC_VOP1", vec![("VDST", 8), ("SRC0", 256)], None);
    let p = analyze(&asm(&[readfirstlane, mov(5, 128), smem("S_LOAD_DWORDX8", 12, 2, 0, 8), image_load(3), end()]), &cs()).unwrap();
    assert_eq!(p.census(), Census { dynamic: 1, ..Default::default() });
    assert!(p.dynamic());
    let mut m = Mem::new();
    m.put(0x4000, &T);
    let mut dw = vec![0u32; 16];
    dw[4] = 0x4000;
    assert!(p.reads(&roots(&dw), &mut m.reader()).is_empty());
}

#[test]
fn paths_that_disagree_make_an_offset_unknown() {
    // 0x00 s_mov s5, 0 · 0x04 s_mov s8, 0x40 · 0x0c s_cbranch_scc1 → 0x18 · 0x10 s_mov s8, X ·
    // 0x18 s_load_dwordx8 s[12:19], s[4:5], s8 · image_load · end
    let prog = |second: u32| {
        asm(&[
            mov(5, 128),
            movl(8, 0x40),
            ("S_CBRANCH_SCC1", "ENC_SOPP", vec![("SIMM16", 2)], None),
            movl(8, second),
            smem("S_LOAD_DWORDX8", 12, 2, 0, 8),
            image_load(3),
            end(),
        ])
    };
    let same = analyze(&prog(0x40), &cs()).unwrap();
    assert_eq!(same.census(), Census { constant: 1, ..Default::default() }, "the same value on both paths");
    let other = analyze(&prog(0x60), &cs()).unwrap();
    assert_eq!(other.census(), Census { dynamic: 1, ..Default::default() });
}

#[test]
fn a_loop_counter_is_unknown() {
    // 0x00 s_mov s5, 0 · 0x04 s_mov s8, 0 · 0x08 s_load_dwordx8 s[12:19], s[4:5], s8 · 0x10 image_load ·
    // 0x18 s_add_u32 s8, s8, 32 · 0x1c s_cbranch_scc1 → 0x08 · 0x20 end
    let p = analyze(
        &asm(&[
            mov(5, 128),
            mov(8, 128),
            smem("S_LOAD_DWORDX8", 12, 2, 0, 8),
            image_load(3),
            ("S_ADD_U32", "ENC_SOP2", vec![("SDST", 8), ("SSRC0", 8), ("SSRC1", 128 + 32)], None),
            ("S_CBRANCH_SCC1", "ENC_SOPP", vec![("SIMM16", (-6i16) as u16 as u64)], None),
            end(),
        ]),
        &cs(),
    )
    .unwrap();
    assert_eq!(p.census(), Census { dynamic: 1, ..Default::default() });
}

#[test]
fn a_table_address_from_the_spilled_arguments() {
    // s[0:1]: the spilled-argument table; its dword at +8 the address of a descriptor table
    let p = analyze(&asm(&[smem("S_LOAD_DWORD", 6, 0, 8, 125), mov(7, 128), smem("S_LOAD_DWORDX8", 8, 3, 0x20, 125), image_load(2), end()]), &cs()).unwrap();
    assert_eq!(p.census(), Census { constant: 1, ..Default::default() }, "a pointer loaded at a constant offset is as good as a root one");
    let mut m = Mem::new();
    m.put(0x5020, &T);
    let mut dw = vec![0u32; 16];
    (dw[0], dw[1]) = (0x0BAD_0000, 0x7);
    let table: Vec<u8> = [0u32, 0, 0x5000].iter().flat_map(|w| w.to_le_bytes()).collect();
    let r = p.reads(&UserData { dwords: &dw, table: &table }, &mut m.reader());
    assert_eq!(r.iter().map(|r| (r.addr, r.words[0])).collect::<Vec<_>>(), [(0x5020, 42)], "the table from the bytes given, not memory");
}

#[test]
fn constant_buffers_and_buffers_are_v_sharps() {
    // V# from the table at +0; a constant read through it and a buffer load with it
    let load: I = ("BUFFER_LOAD_DWORD", "ENC_MUBUF", vec![("VDATA", 1), ("VADDR", 0), ("SRSRC", 2), ("SOFFSET", 128), ("OFFEN", 1)], None);
    let p = analyze(&asm(&[mov(5, 128), smem("S_LOAD_DWORDX4", 8, 2, 0, 125), smem("S_BUFFER_LOAD_DWORD", 12, 4, 0x10, 125), load, end()]), &cs()).unwrap();
    assert_eq!(p.census(), Census { constant: 1, ..Default::default() }, "one V#, used twice");
    assert_eq!(p.uses[0].kind, Kind::Buffer);
    // a V# built in registers from a root view is no load
    let built = analyze(&asm(&[mov(8, 6), mov(9, 7), mov(10, 193), movl(11, 0x2002_7FAC), smem("S_BUFFER_LOAD_DWORD", 12, 4, 0, 125), end()]), &cs()).unwrap();
    assert_eq!(built.census(), Census { inline: 1, ..Default::default() });
    let mut dw = vec![0u32; 16];
    (dw[6], dw[7]) = (0x8000, 3);
    let r = built.reads(&roots(&dw), &mut Mem::new().reader());
    assert_eq!(r.iter().map(|r| (r.kind, r.addr, r.words[..4].to_vec())).collect::<Vec<_>>(), [(Kind::Buffer, 0, vec![0x8000, 3, u32::MAX, 0x2002_7FAC])]);
}

#[test]
fn past_num_records_a_buffer_load_reads_zero() {
    // an index from a constant buffer of 16 bytes: one at +0x20 is 0 on the hardware
    let p = analyze(
        &asm(&[
            mov(5, 128),
            smem("S_LOAD_DWORDX4", 8, 2, 0, 125),
            smem("S_BUFFER_LOAD_DWORD", 12, 4, 0x20, 125),
            smem("S_LOAD_DWORDX8", 16, 2, 0x100, 12),
            image_load(4),
            end(),
        ]),
        &cs(),
    )
    .unwrap();
    let mut m = Mem::new();
    m.put(0x1000, &[0x9000, 0, 16, 0]);
    m.put(0x1100, &T);
    let mut dw = vec![0u32; 16];
    dw[4] = 0x1000;
    let r = p.reads(&roots(&dw), &mut m.reader());
    assert_eq!(r.iter().map(|r| r.addr).collect::<Vec<_>>(), [0x1000, 0x1100]);
}

/// A console shader of the retail pack the analysis was measured on (None: not on this machine).
fn retail(name: &str) -> Option<Vec<u8>> {
    std::fs::read(std::path::Path::new(r"D:\leonida-work\shaders\dxbc").join(name)).ok()
}

/// (user-data dword, offset) of each descriptor the program loads straight from a root pointer
fn root_loads(p: &Program) -> Vec<(u8, i32)> {
    p.uses
        .iter()
        .filter_map(|u| {
            let Node::Load { load, .. } = p.node(u.words[0]) else { return None };
            let l = p.loads[load as usize];
            let Base::Pointer { lo, .. } = l.base else { return None };
            let Node::Entry(d) = p.node(lo) else { return None };
            Some((d, l.imm))
        })
        .collect()
}

#[test]
fn a_retail_compute_shader_reads_what_its_compiler_recorded() {
    // deferred_tiled CS_DepthProcessingHBO: two tables (s4, s5), every descriptor at a constant
    // offset; the console compiler's used-resource table names the same offsets
    let Some(blob) = retail("6deferred_tiled_CS_DepthProcessingHBO_59d77dff_Wrapped.dxbc") else { return };
    let sh = of_shader(&blob).expect("analysed");
    let p = &sh.stages.iter().find(|s| s.hw == 6).expect("a CS").programs[0];
    let c = p.census();
    assert_eq!((c.dynamic, c.indexed), (0, 0), "{c:?}");
    let parsed = xbox::parse(&blob).unwrap();
    let f = parsed.fssm.iter().find(|f| f.stages[0].hw == 6).unwrap();
    let layout = xbox::root_layout(&f.data, &f.stages[0]).unwrap();
    let loads = root_loads(p);
    for u in layout.used.iter().filter(|u| u.via == xbox::RootKind::Table) {
        assert!(loads.contains(&(u.dword, 4 * u.offset as i32)), "{u:?} not found in {loads:?}");
    }
}

/// `XCL_DESC_CENSUS=<file with one shader path a line>`: the census of each
/// (`cargo test -p xcl --lib census -- --ignored --nocapture`).
#[test]
#[ignore]
fn census() {
    let Ok(list) = std::env::var("XCL_DESC_CENSUS") else { return };
    let mut all = Census::default();
    for path in std::fs::read_to_string(list).unwrap().lines().map(str::trim).filter(|l| !l.is_empty()) {
        let Ok(blob) = std::fs::read(path) else { continue };
        match of_shader(&blob) {
            Some(sh) => {
                let c = sh.census();
                all.add(c);
                let stages: Vec<String> = sh.stages.iter().map(|s| xbox::hw_stage_name(s.hw)).collect();
                println!("{c:?} {} {path}", stages.join("+"));
            }
            None => println!("not analysed: {path}"),
        }
    }
    println!("all: {all:?}");
}
