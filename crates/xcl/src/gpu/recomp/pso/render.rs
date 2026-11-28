//! Render-side switches after VIctory-NVIDIA, off by default (the techniques, not its code:
//! VIctory builds DXIL itself, XCL edits the HLSL its recompiler made). Ported from
//! trustno1-coder/VIctory-NVIDIA crates/victory-shader/src/emit/mod.rs and
//! crates/victory-gpu/src/pipeline/make.rs @9ce3cf9e (GPL-3.0-or-later), modified for XCL
//! 2026-11-28.
//!
//! - `XCL_EARLY_DEPTH=1` (VIctory `VICTORY_EARLY_DEPTH`, emit/mod.rs:176-228, lower.rs:1725-1733):
//!   a pixel shader whose console code tests depth before it (DB_SHADER_CONTROL bit 12) and
//!   exports no depth or stencil gets D3D12's early depth-stencil test (`[earlydepthstencil]`).
//!   Without it D3D12 tests after a shader that writes UAVs or discards, so its UAV writes land
//!   for hidden pixels (VIctory: the glass pass's atomics replaced hair in depth and motion).
//! - `XCL_LEAVE_OUT=<list>` (VIctory `VICTORY_LEAVE_OUT`, pipeline/make.rs:60-76, 98-100): the
//!   compute programs named run as empty programs. An entry is an XCL record hash (16 hex
//!   digits, `shader::fnv64` of the container) or a part of a name in the container's bytes
//!   (case kept), split at `,` or `;`; `victory` stands for VIctory's launcher's two. VIctory's launcher leaves out the heat haze
//!   (`scattering_transmittance_system:CS_IndirectApplyDiffusionInScattering_HeatHaze`,
//!   `bca2f01e3e561c7b`; it smears blocks of pixels) and the decal heap's defrag
//!   (`vfx_triangleClip:CS_DefragHeap_ComputeRelocations`, `df231d75ee930b98`; 11.85 ms a
//!   frame on 32-lane hosts).
//! - `XCL_WATER_LIGHT=0` (VIctory `VICTORY_WATER_LIGHT=0`, pipeline/make.rs:77-96): on a GPU
//!   that runs 32-lane waves only (the split wave64 form), the water's tiled light
//!   (`water_shading:CS_TransmissionVolumetrics_Tiled` and `_Tiled_LocalLights`, `aff41dfda2b2b831`,
//!   `b7042d12da0ec706`) is left out: VIctory saw dark tiles of open water from them there.
//!
//! The hashes are XCL's, found by VIctory's (BLAKE3 of the same container bytes) in the
//! 1.0.2.1 shader pack. A record a switch changes is cached under another identity
//! (`identity`). The source hash (build.rs `TRANSLATOR_SOURCES`) does not cover this file.
//! Not covered: the direct DXIL translator (`XCL_DIRECT_DXIL`).

use std::sync::OnceLock;

use super::Stage;
use crate::gpu::xbox;

/// VIctory's 32-lane water light programs (`XCL_WATER_LIGHT=0`), as XCL record hashes.
pub(crate) const WATER_LIGHT: [u64; 2] = [0xaff4_1dfd_a2b2_b831, 0xb704_2d12_da0e_c706];
/// VIctory's launcher's `VICTORY_LEAVE_OUT` (heat haze, decal defrag), as XCL record hashes.
pub(crate) const VICTORY_LEAVE_OUT: &str = "bca2f01e3e561c7b,df231d75ee930b98";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Entry {
    Hash(u64),
    Name(String),
}

pub(crate) fn parse_list(s: &str) -> Vec<Entry> {
    let mut out = Vec::new();
    for e in s.split([',', ';']).map(str::trim).filter(|e| !e.is_empty()) {
        // `victory`: VIctory's launcher's two
        if e.eq_ignore_ascii_case("victory") {
            out.extend(parse_list(VICTORY_LEAVE_OUT));
            continue;
        }
        let hex = e.strip_prefix("0x").unwrap_or(e);
        out.push(match (hex.len() == 16).then(|| u64::from_str_radix(hex, 16).ok()).flatten() {
            Some(h) => Entry::Hash(h),
            None => Entry::Name(e.to_string()),
        });
    }
    out
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Switches {
    pub early_depth: bool,
    pub leave_out: Vec<Entry>,
    pub water_light_off: bool,
}

impl Switches {
    pub(crate) fn parse(early: Option<&str>, leave: Option<&str>, water: Option<&str>) -> Switches {
        Switches {
            early_depth: early.is_some_and(|v| v.trim() == "1"),
            leave_out: leave.map(parse_list).unwrap_or_default(),
            water_light_off: water.is_some_and(|v| v.trim() == "0"),
        }
    }

    fn any(&self) -> bool {
        self.early_depth || !self.leave_out.is_empty() || self.water_light_off
    }

    /// Whether compute record `blob` runs as an empty program (`split`: the 32-lane form).
    pub(crate) fn left_out(&self, blob: &[u8], split: bool) -> bool {
        if self.leave_out.is_empty() && !self.water_light_off {
            return false;
        }
        let hash = crate::gpu::shader::fnv64(blob);
        self.leave_out.iter().any(|e| match e {
            Entry::Hash(h) => *h == hash,
            Entry::Name(n) => !n.is_empty() && blob.windows(n.len()).any(|w| w == n.as_bytes()),
        }) || (self.water_light_off && split && WATER_LIGHT.contains(&hash))
    }
}

pub(crate) fn switches() -> &'static Switches {
    static S: OnceLock<Switches> = OnceLock::new();
    S.get_or_init(|| {
        let var = |n: &str| std::env::var(n).ok();
        let s = Switches::parse(var("XCL_EARLY_DEPTH").as_deref(), var("XCL_LEAVE_OUT").as_deref(), var("XCL_WATER_LIGHT").as_deref());
        if s.any() {
            crate::xlog_info!(
                "[RENDER] early depth {}, leave out {:?}, water light {}",
                if s.early_depth { "on (XCL_EARLY_DEPTH=1)" } else { "off" },
                s.leave_out,
                if s.water_light_off { "left out on 32-lane GPUs (XCL_WATER_LIGHT=0)" } else { "kept" }
            );
        }
        s
    })
}

/// DB_SHADER_CONTROL bit 12 (DEPTH_BEFORE_SHADER) and the Z / stencil export bits.
const DEPTH_BEFORE_SHADER: u32 = 1 << 12;
const DEPTH_STENCIL_EXPORT: u32 = 0b111;

/// The console's DB_SHADER_CONTROL of a PS FSSM: after the PS input table XCL reads
/// (`recomp::ps_input_table`: records at +0x130, their count at +0x100 from there and
/// SPI_PS_INPUT_ENA at +0x110), at +0x114 (VIctory native.rs:103-106: PS_INPUTS 0x24,
/// PS_INPUT_COUNT 0x124, PS_INPUT_ADDR 0x134, PS_DB_SHADER_CONTROL 0x138 from its stage data).
pub(crate) fn db_shader_control(fssm: &[u8]) -> Option<u32> {
    let shift = match fssm.get(0x10..0x14).map(|b| u32::from_le_bytes(b.try_into().unwrap()))? {
        5 => 0,
        6 => 8,
        _ => return None,
    };
    let at = 0x130 + shift + 0x114;
    fssm.get(at..at + 4).map(|b| u32::from_le_bytes(b.try_into().unwrap()))
}

/// Whether the console tests depth before PS `fssm` (its exports are told from the HLSL).
pub(crate) fn asks_early_depth(fssm: &[u8]) -> bool {
    db_shader_control(fssm).is_some_and(|c| c & DEPTH_BEFORE_SHADER != 0 && c & DEPTH_STENCIL_EXPORT == 0)
}

/// `hlsl` as an empty program: `main` returns at once.
pub(crate) fn empty_hlsl(hlsl: &str) -> String {
    let Some(m) = hlsl.find("\nvoid main(") else { return hlsl.to_string() };
    let Some(open) = hlsl[m..].find("\n{\n") else { return hlsl.to_string() };
    let at = m + open + 3;
    format!("{}    return; // XCL_LEAVE_OUT\n{}", &hlsl[..at], &hlsl[at..])
}

/// `hlsl` with D3D12's early depth-stencil test, unless it exports depth or stencil.
pub(crate) fn early_depth_hlsl(hlsl: &str) -> String {
    if hlsl.contains("SV_Depth") || hlsl.contains("SV_StencilRef") {
        return hlsl.to_string();
    }
    hlsl.replacen("\nXclOut main(XclIn xi)\n", "\n[earlydepthstencil]\nXclOut main(XclIn xi)\n", 1)
}

/// The translation of record `blob` (`fssm`: the data of the FSSM translated, if any) for
/// `stage` with the switches that are on.
pub(crate) fn apply(hlsl: String, blob: &[u8], fssm: Option<&[u8]>, stage: Stage, split: bool) -> String {
    apply_with(switches(), hlsl, blob, fssm, stage, split)
}

pub(crate) fn apply_with(s: &Switches, hlsl: String, blob: &[u8], fssm: Option<&[u8]>, stage: Stage, split: bool) -> String {
    match stage {
        Stage::Compute if s.left_out(blob, split) => empty_hlsl(&hlsl),
        Stage::Pixel if s.early_depth && fssm.is_some_and(asks_early_depth) => early_depth_hlsl(&hlsl),
        _ => hlsl,
    }
}

/// `ident` for `blob` in `stage`, under another identity when a switch changes it (a
/// superset of what `apply` changes).
pub(crate) fn identity(ident: u64, blob: &[u8], stage: Stage, split: bool) -> u64 {
    identity_with(switches(), ident, blob, stage, split)
}

pub(crate) fn identity_with(s: &Switches, ident: u64, blob: &[u8], stage: Stage, split: bool) -> u64 {
    if ident == 0 || !s.any() {
        return ident;
    }
    let tag = match stage {
        Stage::Compute if s.left_out(blob, split) => "xcl-leave-out",
        Stage::Pixel if s.early_depth && pixel_asks_early_depth(blob) => "xcl-early-depth",
        _ => return ident,
    };
    super::translated::identity_for(&format!("{ident:016x}"), tag)
}

fn pixel_asks_early_depth(blob: &[u8]) -> bool {
    xbox::parse(blob).is_some_and(|sh| sh.fssm.iter().any(|f| f.stages.first().is_some_and(|s| s.hw == 5) && asks_early_depth(&f.data)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switches_parse() {
        assert!(!Switches::parse(None, None, None).any());
        assert!(!Switches::parse(Some("0"), Some(""), Some("1")).any());
        let s = Switches::parse(Some("1"), Some(VICTORY_LEAVE_OUT), Some("0"));
        assert!(s.early_depth && s.water_light_off);
        assert_eq!(s.leave_out, vec![Entry::Hash(0xbca2_f01e_3e56_1c7b), Entry::Hash(0xdf23_1d75_ee93_0b98)]);
        assert_eq!(parse_list("victory"), parse_list(VICTORY_LEAVE_OUT));
        assert_eq!(parse_list("CS_DefragHeap; 0x00000000000000ff"), vec![Entry::Name("CS_DefragHeap".into()), Entry::Hash(0xff)]);
    }

    #[test]
    fn left_out_by_hash_name_or_water_on_32_lanes() {
        let blob = b"....vfx_triangleClip:CS_DefragHeap_ComputeRelocations_9e6def81_Wrapped....".to_vec();
        let hash = format!("{:016x}", crate::gpu::shader::fnv64(&blob));
        assert!(Switches::parse(None, Some(&hash), None).left_out(&blob, false));
        assert!(Switches::parse(None, Some("CS_DefragHeap_ComputeRelocations"), None).left_out(&blob, false));
        assert!(!Switches::parse(None, Some("cs_defragheap"), None).left_out(&blob, false), "names are matched as they are");
        assert!(!Switches::parse(None, None, Some("0")).left_out(&blob, true));
    }

    #[test]
    fn hlsl_edits() {
        let cs = "x\n[numthreads(64, 1, 1)]\nvoid main(uint3 gid : SV_GroupID)\n{\n    uint a = 1u;\n}\n";
        assert_eq!(empty_hlsl(cs), "x\n[numthreads(64, 1, 1)]\nvoid main(uint3 gid : SV_GroupID)\n{\n    return; // XCL_LEAVE_OUT\n    uint a = 1u;\n}\n");
        let ps = "struct XclOut { float4 t0 : SV_Target0; };\nXclOut main(XclIn xi)\n{\n}\n";
        assert!(early_depth_hlsl(ps).contains("\n[earlydepthstencil]\nXclOut main(XclIn xi)\n"));
        let depth = "struct XclOut { float xcl_depth : SV_Depth; };\nXclOut main(XclIn xi)\n{\n}\n";
        assert_eq!(early_depth_hlsl(depth), depth, "a depth export keeps the late test");
    }

    #[test]
    fn db_shader_control_follows_the_input_table() {
        let mut f = vec![0u8; 0x400];
        f[0x10] = 6; // sub-version 6: the table 8 bytes on
        f[0x130 + 8 + 0x114..][..4].copy_from_slice(&(DEPTH_BEFORE_SHADER | 0x10).to_le_bytes());
        assert_eq!(db_shader_control(&f), Some(DEPTH_BEFORE_SHADER | 0x10));
        assert!(asks_early_depth(&f));
        f[0x130 + 8 + 0x114] |= 1; // Z export
        assert!(!asks_early_depth(&f));
        f[0x10] = 7;
        assert_eq!(db_shader_control(&f), None);
    }

    #[test]
    fn identities() {
        let blob = b"CS_DefragHeap_ComputeRelocations".to_vec();
        let off = Switches::default();
        let on = Switches::parse(None, Some("CS_DefragHeap"), None);
        assert_eq!(identity_with(&off, 7, &blob, Stage::Compute, false), 7);
        assert_ne!(identity_with(&on, 7, &blob, Stage::Compute, false), 7);
        assert_eq!(identity_with(&on, 7, &blob, Stage::Pixel, false), 7, "compute only");
        assert_eq!(identity_with(&on, 7, b"other", Stage::Compute, false), 7);
        assert_eq!(identity_with(&on, 0, &blob, Stage::Compute, false), 0, "cache off");
    }
}
