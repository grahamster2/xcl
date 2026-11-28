//! The constant data after the program, for the ten ACES LUT bakes only.
//!
//! #2229 (MacieGx) lets every program read the tables the console compiler put after its code
//! through `s_getpc_b64`. Builds with it froze on the way into the retail title's world, also
//! with the tables withheld. The picture needs only the ten `postfx_aces` LUT bakes to get them
//! (without them the tone map LUT is one grey texel). Here only those ten records get #2229's
//! translation: their HLSL is edited after the recompiler made it, which is unchanged, so every
//! other record translates byte for byte as before, and the recompiler's source hash (build.rs
//! `TRANSLATOR_SOURCES`) and every cache entry stay as they were.
//!
//! The edit is #2229's HLSL for a program whose table loads all take their address from
//! `s_getpc_b64` on every path (`code_loads::CodeLoad::Code`: no `xcl_gload`, `s_getpc_b64`'s
//! high dword stays 0): those `s_load_dword*` and FLAT / GLOBAL `*_load_dword*` read the table
//! (`xcl_cload*`) instead of the memory, and the table and its loaders (`rodata_loader`, from
//! #2229's `emit.rs`) go where #2229 puts them. A bake with a load that may read either is left
//! as it is (logged).
//!
//! `XCL_ACES_RODATA=0` turns it off.

use std::fmt::Write as _;
use std::sync::OnceLock;

use super::code_loads::{code_loads, CodeLoad};
use crate::gpu::{rdna2, xbox};

/// The ten `postfx_aces:CS_Bake*LUT*` records of the 1.0.2.1 shader pack (`shader::fnv64`).
pub(crate) const ACES: [u64; 10] = [
    0xb3fe_6de7_72ed_4c3c, // CS_BakeCESLUT_LogShaper
    0x7248_b44b_1ad6_cd22, // CS_BakeCESLUT_LinearShaper
    0xe019_530c_e2a8_3e0a, // CS_BakeCESLUT_PQShaper
    0x5115_9a17_706f_a4eb, // CS_BakeCESLUT_QuasiLogShaper
    0x422c_ad81_d02c_76e8, // CS_BakeCESTonemapOnlyLUT_QuasiLogShaper
    0xa069_2f03_22ff_0efe, // CS_BakeACESWithGamutCompressLUT_LogShaper
    0x3ddc_c407_83c4_06b2, // CS_BakeACESWithGamutCompressLUT_LinearShaper
    0xf5e4_54e8_c979_fb77, // CS_BakeACESWithGamutCompressLUT_PQShaper
    0x14f3_3bcc_cd1e_c01d, // CS_BakeACESWithGamutCompressLUT_QuasiLogShaper
    0x9136_eb6a_fa6d_8a2e, // CS_BakeACESWithGamutCompressTonemapOnlyLUT_QuasiLogShaper
];

/// The high dword `s_getpc_b64` gives with a mixed load (#2229's `CODE_ADDRESS_HI`; only in
/// `xcl_code`, which the edited bakes do not call).
const CODE_ADDRESS_HI: u32 = 0xC0DE;

pub(crate) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("XCL_ACES_RODATA").map_or(true, |v| v.trim() != "0");
        crate::xlog_info!(
            "[ACES] constant data after the program for the ten postfx_aces LUT bakes only: {}",
            if on { "on (XCL_ACES_RODATA=0 turns it off)" } else { "off (XCL_ACES_RODATA=0)" }
        );
        on
    })
}

pub(crate) fn allowed(blob: &[u8]) -> bool {
    ACES.contains(&crate::gpu::shader::fnv64(blob))
}

/// `ident` for `blob`: another identity for an allowed bake while this is on (the source of this
/// file and `code_loads.rs` in it), the same for every other record.
pub(crate) fn identity(ident: u64, blob: &[u8]) -> u64 {
    if ident == 0 || !enabled() || !allowed(blob) {
        return ident;
    }
    static TAG: OnceLock<String> = OnceLock::new();
    let tag = TAG.get_or_init(|| {
        let src = concat!(include_str!("aces.rs"), include_str!("code_loads.rs"));
        format!("xcl-aces-rodata-{:016x}", crate::gpu::shader::fnv64(src.replace("\r\n", "\n").as_bytes()))
    });
    super::translated::identity_for(&format!("{ident:016x}"), tag)
}

/// The data blocks after program `p` of `f`, as the program addresses them (from its start):
/// #2229's `xbox::Fssm::rodata`.
pub(crate) fn rodata(f: &xbox::Fssm, p: &xbox::Program) -> Vec<(u32, Vec<u32>)> {
    f.data_blocks
        .iter()
        .filter(|b| b.offset >= p.offset + p.len)
        .filter_map(|b| {
            let bytes = f.data.get(b.offset..b.offset + b.len)?;
            Some(((b.offset - p.offset) as u32, bytes.chunks_exact(4).map(|w| u32::from_le_bytes(w.try_into().unwrap())).collect()))
        })
        .collect()
}

/// #2229's `emit::rodata_loader`, as it is there.
pub(crate) fn rodata_loader(rodata: &[(u32, Vec<u32>)]) -> String {
    let total: usize = rodata.iter().map(|b| b.1.len()).sum();
    if total == 0 {
        return String::new();
    }
    let words: Vec<String> = rodata.iter().flat_map(|b| b.1.iter()).map(|w| format!("0x{w:08X}u")).collect();
    let mut s = format!("static const uint xcl_rodata[{total}] = {{ {} }};\nuint xcl_cload(uint a) {{\n    uint r = 0u;\n", words.join(", "));
    let mut first = 0usize;
    let mut inside = Vec::new();
    for (at, block) in rodata.iter().filter(|b| !b.1.is_empty()) {
        let bytes = 4 * block.len();
        let _ = writeln!(s, "    r = select(a - {at}u < {bytes}u, xcl_rodata[min({first}u + ((a - {at}u) >> 2), {}u)], r);", total - 1);
        inside.push(format!("a - {at}u < {bytes}u"));
        first += block.len();
    }
    s.push_str("    return r;\n}\n");
    s.push_str("uint2 xcl_cload2(uint a) { return uint2(xcl_cload(a), xcl_cload(a + 4u)); }\n");
    s.push_str("uint4 xcl_cload4(uint a) { return uint4(xcl_cload(a), xcl_cload(a + 4u), xcl_cload(a + 8u), xcl_cload(a + 12u)); }\n");
    let inside = inside.into_iter().reduce(|x, y| format!("or({x}, {y})")).expect("a block");
    let _ = writeln!(s, "bool xcl_code(uint hi, uint a) {{ return and(hi == {CODE_ADDRESS_HI:#X}u, {inside}); }}");
    for (n, ty) in [("", "uint"), ("2", "uint2"), ("4", "uint4")] {
        let c = if n.is_empty() { "c".to_string() } else { format!("(bool{n})c") };
        let _ = writeln!(s, "{ty} xcl_gload{n}(uint hi, uint a) {{ {ty} m = mem.Load{n}(a); bool c = xcl_code(hi, a); return select({c}, xcl_cload{n}(a), m); }}");
    }
    s
}

/// The decoded program, as `recomp::translate` decodes it.
fn decode(code: &[u8]) -> Option<Vec<rdna2::Decoded>> {
    let mut insts = Vec::new();
    let mut off = 0usize;
    while off + 4 <= code.len() {
        let d = rdna2::decode(&code[off..], off as u64).ok()?;
        off += d.len;
        insts.push(d);
    }
    rdna2::resolve_far_jumps(&mut insts);
    Some(insts)
}

/// Whether `d` is a load #2229's HLSL path reads the table with: `s_load_dword*` (not
/// `s_buffer_load*`), FLAT / GLOBAL `*_load_dword*`.
fn table_load(d: &rdna2::Decoded) -> bool {
    let n = d.inst.name;
    (n.starts_with("S_LOAD_DWORD") || n.starts_with("GLOBAL_LOAD_DWORD") || n.starts_with("FLAT_LOAD_DWORD"))
        && !n.starts_with("S_BUFFER")
}

/// Where #2229 puts the loader: after the memory declarations and accessors, before the static
/// samplers and the prelude (`static uint lane_id;`).
fn loader_at(hlsl: &str) -> Option<usize> {
    let prelude = hlsl.find("\nstatic uint lane_id;\n")?;
    let head = &hlsl[..prelude];
    let sampler = ["\nSamplerState xcl_ss", "\nSamplerComparisonState xcl_ss"].iter().filter_map(|s| head.find(s)).min();
    Some(sampler.map_or(prelude, |s| s + 1))
}

/// `hlsl` (the recompiler's, of `code`) with #2229's table reads for the loads `loads` marks
/// `Code`. None when a load is `Maybe` or the HLSL is not of the form looked for.
pub(crate) fn edit(hlsl: &str, code: &[u8], rodata: &[(u32, Vec<u32>)]) -> Result<String, &'static str> {
    if rodata.iter().all(|b| b.1.is_empty()) {
        return Err("no constant data");
    }
    let insts = decode(code).ok_or("cannot decode")?;
    let loads = code_loads(&insts);
    if loads.values().any(|k| *k == CodeLoad::Maybe) {
        return Err("a load that may read the table or the memory");
    }
    let pcs: Vec<u64> = insts.iter().filter(|d| table_load(d) && loads.get(&d.pc) == Some(&CodeLoad::Code)).map(|d| d.pc).collect();
    if pcs.is_empty() {
        return Err("no load of the table");
    }
    // the lines of each instruction: from its `// pc: …` comment to the next one
    let mut out = String::with_capacity(hlsl.len() + 4096);
    let mut edited = 0usize;
    let mut current: Option<u64> = None;
    for line in hlsl.split_inclusive('\n') {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix("// ") {
            if let Some((pc, _)) = rest.split_once(": ") {
                if pc.len() >= 4 && pc.bytes().all(|b| b.is_ascii_hexdigit()) {
                    current = u64::from_str_radix(pc, 16).ok();
                }
            }
        }
        // (several memory windows: the recompiler's `windowed` made `mem.Load*(` `xcl_mLoad*(`)
        if current.is_some_and(|pc| pcs.contains(&pc)) && (line.contains("mem.Load") || line.contains("xcl_mLoad")) {
            out.push_str(&line.replace("mem.Load", "xcl_cload").replace("xcl_mLoad", "xcl_cload"));
            edited += 1;
        } else {
            out.push_str(line);
        }
    }
    if edited == 0 {
        return Err("no table load found in the HLSL");
    }
    let at = loader_at(&out).ok_or("no prelude in the HLSL")?;
    // #2229 adds the loader before `windowed` rewrites the memory accesses
    let windowed = out.contains("static uint xcl_w = 0u;\n");
    let loader = rodata_loader(rodata);
    out.insert_str(at, &if windowed { loader.replace("mem.Load", "xcl_mLoad") } else { loader });
    Ok(out)
}

/// The translation of record `blob` (program `p` of `f`, `code` its bytes): edited for an
/// allowed bake while this is on, as it is otherwise.
pub(crate) fn apply(hlsl: String, blob: &[u8], f: &xbox::Fssm, p: &xbox::Program, code: &[u8]) -> String {
    if !allowed(blob) || !enabled() {
        return hlsl;
    }
    match edit(&hlsl, code, &rodata(f, p)) {
        Ok(h) => h,
        Err(why) => {
            crate::xlog_warn!("[ACES] {:016x}: left as it is ({why})", crate::gpu::shader::fnv64(blob));
            hlsl
        }
    }
}
