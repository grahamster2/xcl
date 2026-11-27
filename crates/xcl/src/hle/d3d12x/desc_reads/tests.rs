use super::*;
use crate::gpu::effects::descriptors::{analyze, Stage};
use crate::gpu::rdna2;

fn enc(name: &str, e: &str, f: &[(&str, u64)]) -> Vec<u8> {
    rdna2::encode(name, e, f, None).unwrap()
}

/// a CS: s4 a descriptor table; T# at +0x40 sampled, V# at +0x80 loaded from
pub(crate) fn program() -> Shader {
    let mut code = enc("S_MOV_B32", "ENC_SOP1", &[("SDST", 5), ("SSRC0", 128)]);
    code.extend(enc("S_LOAD_DWORDX8", "ENC_SMEM", &[("SDATA", 8), ("SBASE", 2), ("OFFSET", 0x40), ("SOFFSET", 125)]));
    code.extend(enc("S_LOAD_DWORDX4", "ENC_SMEM", &[("SDATA", 16), ("SBASE", 2), ("OFFSET", 0x80), ("SOFFSET", 125)]));
    code.extend(enc("IMAGE_LOAD", "ENC_MIMG", &[("VDATA", 4), ("VADDR", 0), ("SRSRC", 2), ("DMASK", 15), ("DIM", 1)]));
    code.extend(enc("BUFFER_LOAD_DWORD", "ENC_MUBUF", &[("VDATA", 1), ("VADDR", 0), ("SRSRC", 4), ("SOFFSET", 128), ("OFFEN", 1)]));
    code.extend(enc("S_ENDPGM", "ENC_SOPP", &[]));
    let p = analyze(&code, &(0..16).collect::<Vec<_>>()).unwrap();
    Shader { stages: vec![Stage { hw: 6, programs: vec![p] }] }
}

pub(crate) fn memory(at: u64, out: &mut [u32]) -> bool {
    let words: &[u32] = match at {
        0x1040 => &[6, 0, 0, 0x9000_0000, 0, 0, 0, 0],
        0x1080 => &[0x2000, 0x0010_0003, 64, 0],
        _ => return false,
    };
    out.copy_from_slice(&words[..out.len()]);
    true
}

#[test]
fn a_dispatch_touches_the_texture_and_the_buffer_its_table_names() {
    let shaders = vec![Arc::new(program())];
    let mut dw = vec![0u32; 16];
    dw[4] = 0x1000;
    let roots = Roots { cs: Some(UserData { dwords: &dw, table: &[] }), ..Default::default() };
    let reads = reads_of(&shaders, &roots, &mut memory);
    assert_eq!(reads.len(), 2);
    // shared-heap index 6 names resource 0xAA; its UAV (7) another one never asked for
    let named = |i: u32| if i == 6 { 0xAA } else { 0 };
    let t = touches_of(&reads, 0x100_0000_0000, &named);
    assert_eq!(t, [Touch::Res(0xAA), Touch::Va { va: 0x3_0000_2000, bytes: 64 * 16 }]);
    // only graphics roots: a compute program reads nothing
    let gfx = Roots { gfx: roots.cs, ..Default::default() };
    assert!(reads_of(&shaders, &gfx, &mut memory).is_empty());
}

#[test]
fn a_view_of_a_destroyed_resource_or_none_touches_nothing() {
    let r = DescRead { kind: Kind::Image, dwords: 8, words: [6, 0, 0, 0x9000_0000, 0, 0, 0, 0], addr: 0x1040 };
    assert!(touches_of(&[r], 0, &|_| 0).is_empty());
    assert!(touches_of(&[r], 0, &|_| super::super::console_heap::GONE).is_empty());
    // the SRV unknown, the UAV next to it known
    assert_eq!(touches_of(&[r], 0, &|i| if i == 7 { 5 } else { 0 }), [Touch::Res(5)]);
    let s = DescRead { kind: Kind::Sampler, dwords: 4, words: [0; 8], addr: 0 };
    assert!(touches_of(&[s], 0, &|_| 1).is_empty(), "samplers have no memory");
}

#[test]
fn buffer_ranges() {
    // stride 0: num_records bytes; capped
    assert_eq!(buffer_range(0x100_0000_0000, &[0x40, 0, 256, 0, 0, 0, 0, 0]), Some((0x100_0000_0040, 256)));
    assert_eq!(buffer_range(0, &[0x40, 5 | 16 << 16, u32::MAX, 0, 0, 0, 0, 0]), Some((0x5_0000_0040, BUFFER_MAX)));
    assert_eq!(buffer_range(0, &[0, 0, 256, 0, 0, 0, 0, 0]), None, "a null V#");
    assert_eq!(host_of(0x100_0000_0000, 0x1234), 0x100_0000_1234);
    assert_eq!(host_of(0x100_0000_0000, 0x102_0000_1234), 0x102_0000_1234);
}

#[test]
fn roots_differ_by_every_value() {
    let (a, b) = ([1u32, 2, 3], [1u32, 2, 4]);
    let k = |cs: &[u32], t: &[u8]| Roots { cs: Some(UserData { dwords: cs, table: t }), ..Default::default() }.key();
    assert_ne!(k(&a, &[]), k(&b, &[]));
    assert_ne!(k(&a, &[]), k(&a, &[0]));
    assert_eq!(k(&a, &[9]), k(&a, &[9]));
    assert_ne!(Roots { gfx: Some(UserData { dwords: &a, table: &[] }), ..Default::default() }.key(), k(&a, &[]));
}
