//! Diagnostic safety nets for the HLSL path, off by default, after VIctory-NVIDIA's
//! translator (the technique, not its code: VIctory builds DXIL itself, XCL edits the HLSL
//! its recompiler made). Ported from trustno1-coder/VIctory-NVIDIA
//! crates/victory-shader/src/emit/lower.rs @9ce3cf9e (GPL-3.0-or-later), modified for XCL
//! 2026-11-28.
//!
//! - `XCL_LOOP_CAP=<n>`: a compute program's lanes run at most `n` loop iterations in all
//!   (VIctory, `count_iteration`: 4 096 per loop for compute). XCL's recompiler already caps
//!   them, at 65 536 for a lane over all its loops (`xcl_budget`, `LOOP_GUARD`); this sets that
//!   budget. Only compute programs with a backward branch are changed.
//! - `XCL_RODATA_CLAMP=1`: a read of the constant data after the program (`Options::rodata`)
//!   outside its blocks gives the table's last dword, not 0 (VIctory, `code_dword`: the index
//!   into its code array is clamped to the last dword), and a load that may read it
//!   (`xcl_gload`) takes the table for every code address (high dword `CODE_ADDRESS_HI`), not
//!   only for one inside a block.
//!
//! Both change the translation, so a record they change is cached under another identity
//! (`identity`): an entry made without them is never used with them, or the other way round.
//! The source hash (build.rs `TRANSLATOR_SOURCES`) does not cover this file, so with both off
//! every entry stays valid.
//!
//! Not covered: the direct DXIL translator (`XCL_DIRECT_DXIL`) and the per-lane form.

use std::sync::OnceLock;

use super::Stage;
use crate::gpu::{rdna2, xbox};

/// `XCL_LOOP_CAP`'s value: iterations, at least 1.
pub(crate) fn parse_cap(v: Option<&str>) -> Option<u32> {
    v.and_then(|v| v.trim().parse::<u32>().ok()).filter(|&n| n > 0)
}

pub(crate) fn loop_cap() -> Option<u32> {
    static C: OnceLock<Option<u32>> = OnceLock::new();
    *C.get_or_init(|| {
        let c = parse_cap(std::env::var("XCL_LOOP_CAP").ok().as_deref());
        if let Some(n) = c {
            crate::xlog_info!("[SAFETY] XCL_LOOP_CAP={n}: a compute program's lanes run at most {n} loop iterations (default 65536)");
        }
        c
    })
}

pub(crate) fn clamp() -> bool {
    static C: OnceLock<bool> = OnceLock::new();
    *C.get_or_init(|| {
        let on = std::env::var("XCL_RODATA_CLAMP").is_ok_and(|v| v.trim() == "1");
        if on {
            crate::xlog_info!("[SAFETY] XCL_RODATA_CLAMP=1: reads of a program's constant data outside its blocks give its last dword");
        }
        on
    })
}

/// Whether `code` has a backward branch: the recompiler's loops come from those only.
pub(crate) fn has_loop(code: &[u8]) -> bool {
    let mut insts = Vec::new();
    let mut off = 0usize;
    while off + 4 <= code.len() {
        match rdna2::decode(&code[off..], off as u64) {
            Ok(d) => {
                off += d.len;
                insts.push(d);
            }
            Err(_) => off += 4,
        }
    }
    rdna2::resolve_far_jumps(&mut insts);
    insts.iter().any(|d| rdna2::branch_target(d).is_some_and(|t| t <= d.pc))
}

const BUDGET: &str = "static uint xcl_budget = 1u << 16;";

/// The loop budget of `hlsl` set to `n`.
pub(crate) fn cap_hlsl(hlsl: &str, n: u32) -> String {
    hlsl.replacen(BUDGET, &format!("static uint xcl_budget = {n}u;"), 1)
}

/// `hlsl`'s constant data loader (`emit::rodata_loader`) in the clamped form.
pub(crate) fn clamp_hlsl(hlsl: &str) -> String {
    const DECL: &str = "static const uint xcl_rodata[";
    let Some(at) = hlsl.find(DECL) else { return hlsl.to_string() };
    let Some(total) = hlsl[at + DECL.len()..].split(']').next().and_then(|n| n.parse::<usize>().ok()).filter(|&n| n > 0) else {
        return hlsl.to_string();
    };
    let mut s = hlsl.replacen("uint xcl_cload(uint a) {\n    uint r = 0u;\n", &format!("uint xcl_cload(uint a) {{\n    uint r = xcl_rodata[{}u];\n", total - 1), 1);
    const CODE: &str = "bool xcl_code(uint hi, uint a) { return ";
    if let Some(c) = s.find(CODE) {
        if let Some(end) = s[c..].find('\n') {
            s.replace_range(c..c + end, &format!("{CODE}hi == {:#X}u; }}", super::super::CODE_ADDRESS_HI));
        }
    }
    s
}

/// The translation of `code` (a program of `stage`, `rodata` its constant data) with the
/// switches that are on.
pub(crate) fn apply(hlsl: String, stage: Stage, code: &[u8], rodata: bool) -> String {
    let hlsl = match loop_cap() {
        Some(n) if stage == Stage::Compute && has_loop(code) => cap_hlsl(&hlsl, n),
        _ => hlsl,
    };
    if rodata && clamp() { clamp_hlsl(&hlsl) } else { hlsl }
}

/// Which switches change the translation of record `blob` in `stage`: (loop cap, clamp).
/// A superset of what `apply` changes for the program it translates.
fn changes(blob: &[u8], stage: Stage, cap: bool, clamp: bool) -> (bool, bool) {
    if !cap && !clamp {
        return (false, false);
    }
    let cap = cap && stage == Stage::Compute;
    if let Some(f) = super::fspx(blob) {
        return (cap && has_loop(f.code), false);
    }
    let Some(sh) = xbox::parse(blob) else { return (false, false) };
    let progs = || sh.fssm.iter().flat_map(|f| f.programs.iter().map(move |p| (f, p)));
    (
        cap && progs().any(|(f, p)| f.data.get(p.offset..p.offset + p.len).is_some_and(has_loop)),
        clamp && progs().any(|(f, p)| !super::rodata_gate::rodata(blob, f, p).is_empty()),
    )
}

/// `ident` for `blob` in `stage`, under another identity for each switch that changes it.
pub(crate) fn identity(ident: u64, blob: &[u8], stage: Stage) -> u64 {
    identity_with(ident, blob, stage, loop_cap(), clamp())
}

pub(crate) fn identity_with(ident: u64, blob: &[u8], stage: Stage, cap: Option<u32>, clamp: bool) -> u64 {
    if ident == 0 {
        return 0;
    }
    let (c, k) = changes(blob, stage, cap.is_some(), clamp);
    let mut id = ident;
    if c {
        id = super::translated::identity_for(&format!("{id:016x}"), &format!("xcl-loop-cap-{}", cap.unwrap_or(0)));
    }
    if k {
        id = super::translated::identity_for(&format!("{id:016x}"), "xcl-rodata-clamp");
    }
    id
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_values() {
        assert_eq!(parse_cap(None), None);
        assert_eq!(parse_cap(Some("0")), None);
        assert_eq!(parse_cap(Some("x")), None);
        assert_eq!(parse_cap(Some(" 4096 ")), Some(4096));
    }

    #[test]
    fn the_budget_is_set() {
        let h = format!("a\n{BUDGET}\nb");
        assert_eq!(cap_hlsl(&h, 4096), "a\nstatic uint xcl_budget = 4096u;\nb");
    }

    #[test]
    fn outside_the_blocks_the_last_dword() {
        let h = "static const uint xcl_rodata[3] = { 1u, 2u, 3u };\nuint xcl_cload(uint a) {\n    uint r = 0u;\n    r = select(...);\n    return r;\n}\nbool xcl_code(uint hi, uint a) { return and(hi == 0xC0DEu, a - 8u < 12u); }\nrest\n";
        let c = clamp_hlsl(h);
        assert!(c.contains("uint r = xcl_rodata[2u];"), "{c}");
        assert!(c.contains("bool xcl_code(uint hi, uint a) { return hi == 0xC0DEu; }\nrest"), "{c}");
        assert_eq!(clamp_hlsl("no table"), "no table");
    }

    /// Only a record a switch changes moves to another identity; each switch has its own.
    #[test]
    fn identities() {
        // s_mov_b32 s0, s1 ; s_endpgm   and   s_nop ; s_branch -2 (to the nop) ; s_endpgm
        let flat: Vec<u8> = [0xBE80_0301u32, 0xBF81_0000].iter().flat_map(|w| w.to_le_bytes()).collect();
        let looped: Vec<u8> = [0xBF80_0000u32, 0xBF82_FFFE, 0xBF81_0000].iter().flat_map(|w| w.to_le_bytes()).collect();
        assert!(!has_loop(&flat));
        assert!(has_loop(&looped));
        use crate::gpu::xbox::tests::{fssm, sample};
        let cs = |code: &[u8]| sample(0, &[fssm(code, &[(6, 0)])]);
        let (a, b) = (cs(&flat), cs(&looped));
        assert_eq!(identity_with(7, &b, Stage::Compute, None, false), 7);
        assert_eq!(identity_with(0, &b, Stage::Compute, Some(4096), true), 0, "cache off");
        assert_eq!(identity_with(7, &a, Stage::Compute, Some(4096), false), 7, "no loop");
        let capped = identity_with(7, &b, Stage::Compute, Some(4096), false);
        assert_ne!(capped, 7);
        assert_ne!(capped, identity_with(7, &b, Stage::Compute, Some(1024), false), "the cap is in it");
        assert_eq!(identity_with(7, &b, Stage::Pixel, Some(4096), false), 7, "compute only");
        assert_eq!(identity_with(7, &b, Stage::Compute, None, true), 7, "no constant data");
    }
}
