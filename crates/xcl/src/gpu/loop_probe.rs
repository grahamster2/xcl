//! `XCL_LOOP_STATS`: do the translated compute programs end their loops because the loop guards
//! cut them short (a loop that spins on data XCL got wrong), or do they run as the console's?
//!
//! Every loop of a translated program has guards (`recomp::LOOP_GUARD`, `Em::loop_guard`): a
//! lane's 65 536 passes in all (`xcl_budget`), the passes one entry of a loop may run in a compute
//! program (`xcl_cap`, the dispatch's share: `IO_LOOP_BUDGET`), the 2^20 polls a split wave64
//! half waits for its partner (`xcl_x_meet`, then `xcl_x_dead`) and the early-exit-safe barrier's
//! 2^20 polls. Neither translator counts what they cut. This module adds the counters to the
//! HLSL of the compute programs named (`XCL_LOOP_STATS=<hash>,…` or `all`) after translation:
//! a text pass over the generated HLSL, so the translator sources (and with them the cache
//! identity of every program, `build.rs`) stay as they are. Those programs take the HLSL path
//! (not direct DXIL) and no disk cache entry (`recomp::pso::console_to_dxil_in`); every other
//! program is translated and cached as without the switch.
//!
//! The counters are words in the program's `res` buffer (u1, which a compute program does not
//! use otherwise: the pass refuses one that does). D3D12.X binds a block of its own there for
//! each named program, which the dispatches of the program add to and `hle::d3d12x::loop_stats`
//! reads every few seconds.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

/// Word of the counter block: lanes that spent their 65 536 loop passes (`xcl_budget`), once each.
pub const W_BUDGET_LANES: usize = 0;
/// Lanes whose loop entry ran out of its passes (`xcl_cap`), once each.
pub const W_CAP_LANES: usize = 1;
/// Split wave64 halves whose partner did not come within 2^20 polls (`xcl_x_dead` set).
pub const W_X_TIMEOUTS: usize = 2;
/// Waves whose early-exit-safe barrier gave up after 2^20 polls.
pub const W_BAR_TIMEOUTS: usize = 3;
/// Loop exits for a spent budget, every one (a lane whose budget is spent leaves each loop it
/// comes to at once).
pub const W_BUDGET_EXITS: usize = 4;
/// Loop entries ended by their passes, every one (lanes).
pub const W_CAP_EXITS: usize = 5;
/// Lanes that started a 1025th loop pass (all loops of the lane together).
pub const W_PAST_1K: usize = 6;
/// Lanes that started a 16 385th loop pass.
pub const W_PAST_16K: usize = 7;
/// Words of the counters.
pub const WORDS: usize = 8;

/// What `XCL_LOOP_STATS` asks for.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Switch {
    Off,
    /// `1`: dispatch statistics only, no program changed
    Host,
    /// `<hash>,…` (dump hashes or the `cs` hashes the GPU time tables give, hex): those programs counted too
    Named(Vec<u64>),
    /// `all`: every compute program counted (each translated again at each run)
    All,
}

impl Switch {
    pub fn on(&self) -> bool {
        *self != Switch::Off
    }
}

/// The switch from its value.
pub fn parse(v: Option<&str>) -> Switch {
    let Some(v) = v.map(str::trim).filter(|v| !v.is_empty() && *v != "0") else { return Switch::Off };
    match v {
        "1" => Switch::Host,
        "all" => Switch::All,
        _ => {
            let list: Vec<u64> = v.split(',').filter_map(|h| u64::from_str_radix(h.trim().trim_start_matches("0x"), 16).ok()).collect();
            if list.is_empty() {
                Switch::Host
            } else {
                Switch::Named(list)
            }
        }
    }
}

pub fn switch() -> &'static Switch {
    static S: OnceLock<Switch> = OnceLock::new();
    S.get_or_init(|| parse(std::env::var("XCL_LOOP_STATS").ok().as_deref()))
}

/// The `cs` hash the GPU time tables name a program by (`real::shader_key_of_hash`).
pub fn cs_hash(dump: u64) -> u64 {
    crate::hle::d3d12x::real::shader_key_of_hash(dump)
}

/// Whether `s` asks for the counters in the compute program of dump hash `dump`.
pub fn named_in(s: &Switch, dump: u64) -> bool {
    match s {
        Switch::All => true,
        Switch::Named(list) => list.iter().any(|&h| h == dump || h == cs_hash(dump)),
        _ => false,
    }
}

/// Whether the compute program of dump hash `dump` gets the counters in this run.
pub fn wanted(dump: u64) -> bool {
    named_in(switch(), dump)
}

/// The programs (dump hashes) whose DXIL has the counters in this process.
static COUNTED: Mutex<Option<HashSet<u64>>> = Mutex::new(None);

pub fn note_counted(dump: u64) {
    COUNTED.lock().unwrap().get_or_insert_with(HashSet::new).insert(dump);
}

/// Whether the program of dump hash `dump` was built with the counters (its dispatches bind a
/// counter block at `res`).
pub fn counted(dump: u64) -> bool {
    COUNTED.lock().unwrap().as_ref().is_some_and(|s| s.contains(&dump))
}

/// How many of each guard the pass found.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sites {
    /// lane budget checks (`LOOP_GUARD`), one a loop
    pub budget: usize,
    /// loop entry pass counts (`xcl_cap`)
    pub caps: usize,
    /// split wave64 exchange polls
    pub exchange: usize,
    /// early-exit-safe barrier polls
    pub barrier: usize,
}

const BUDGET_GUARD: &str = "if (xcl_budget == 0u) break; xcl_budget--;";
const EXCHANGE_POLL: &str = "if (n >= (1u << 20)) {";
const BARRIER_POLL: &str = "if (_n >= (1u << 20)) {";
/// loop passes left in a lane's budget when it starts its 1025th and its 16 385th pass
const LEFT_PAST_1K: u32 = (1 << 16) - 1024;
const LEFT_PAST_16K: u32 = (1 << 16) - 16384;

/// The helpers the guards call, adding to `buf` from byte `base`.
fn helpers(buf: &str, base: u32) -> String {
    format!(
        r#"// XCL_LOOP_STATS: counters of the loop guards (`gpu::loop_probe`)
static uint xcl_ls_seen = 0u;
void xcl_ls_add(uint w, uint n) {{ uint o; {buf}.InterlockedAdd({base}u + w * 4u, n, o); }}
void xcl_ls_lanes(uint w) {{ uint n = WaveActiveCountBits(true); if (WaveIsFirstLane()) xcl_ls_add(w, n); }}
void xcl_ls_exit(uint k) {{
    xcl_ls_lanes({events}u + k);
    bool fresh = (xcl_ls_seen & (1u << k)) == 0u;
    xcl_ls_seen |= 1u << k;
    uint n = WaveActiveCountBits(fresh);
    if (WaveIsFirstLane() && n != 0u) xcl_ls_add(k, n);
}}
"#,
        events = W_BUDGET_EXITS
    )
}

/// Whether `hlsl` names `name.` anywhere (a use of the buffer `name`).
fn uses(hlsl: &str, name: &str) -> bool {
    let pat = format!("{name}.");
    hlsl.match_indices(&pat).any(|(at, _)| at == 0 || !hlsl.as_bytes()[at - 1].is_ascii_alphanumeric() && hlsl.as_bytes()[at - 1] != b'_')
}

/// The HLSL of a translated compute program with counters at its loop guards: `buf` (a
/// `RWByteAddressBuffer` it declares) gets them at byte `base` on (`WORDS` words). Refused when the
/// program uses `buf` itself, or does not declare it.
pub fn instrument(hlsl: &str, buf: &str, base: u32) -> Result<(String, Sites), String> {
    if buf == "res" && uses(hlsl, buf) {
        return Err("the program uses res".into());
    }
    let decl = format!("RWByteAddressBuffer {buf} : register(");
    let at = hlsl.find(&decl).ok_or_else(|| format!("no {buf} declared"))?;
    let after = at + hlsl[at..].find('\n').ok_or("declaration without an end")? + 1;
    let mut sites = Sites::default();
    let mut out = String::with_capacity(hlsl.len() + 4096);
    out.push_str(&hlsl[..after]);
    out.push_str(&helpers(buf, base));
    let rest = &hlsl[after..];
    // a lane's budget: its exits, and lanes that ran long
    sites.budget = rest.matches(BUDGET_GUARD).count();
    let rest = rest.replace(
        BUDGET_GUARD,
        &format!(
            "if (xcl_budget == 0u) {{ xcl_ls_exit({W_BUDGET_LANES}u); break; }} if (xcl_budget == {LEFT_PAST_1K}u) xcl_ls_lanes({W_PAST_1K}u); if (xcl_budget == {LEFT_PAST_16K}u) xcl_ls_lanes({W_PAST_16K}u); xcl_budget--;"
        ),
    );
    // each loop entry's own count: `if (xcl_n<k> == 0u) break;`
    let mut capped = String::with_capacity(rest.len() + 1024);
    let mut tail = rest.as_str();
    while let Some(k) = tail.find("if (xcl_n") {
        let digits = tail[k + 9..].bytes().take_while(u8::is_ascii_digit).count();
        let end = k + 9 + digits;
        if digits > 0 && tail[end..].starts_with(" == 0u) break;") {
            capped.push_str(&tail[..end]);
            capped.push_str(&format!(" == 0u) {{ xcl_ls_exit({W_CAP_LANES}u); break; }}"));
            tail = &tail[end + " == 0u) break;".len()..];
            sites.caps += 1;
        } else {
            capped.push_str(&tail[..end]);
            tail = &tail[end..];
        }
    }
    capped.push_str(tail);
    sites.exchange = capped.matches(EXCHANGE_POLL).count();
    sites.barrier = capped.matches(BARRIER_POLL).count();
    let capped = capped
        .replace(EXCHANGE_POLL, &format!("{EXCHANGE_POLL} xcl_ls_add({W_X_TIMEOUTS}u, 1u);"))
        .replace(BARRIER_POLL, &format!("{BARRIER_POLL} xcl_ls_add({W_BAR_TIMEOUTS}u, 1u);"));
    out.push_str(&capped);
    Ok((out, sites))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu::rdna2::encode;
    use crate::gpu::recomp::{runner, translate, CsInterface, Mode, Options, IO_LOOP_BUDGET};

    fn prog(list: &[(&str, &str, &[(&str, u64)], Option<u32>)]) -> Vec<u8> {
        let mut v = Vec::new();
        for (n, e, f, l) in list {
            v.extend(encode(n, e, f, *l).unwrap_or_else(|x| panic!("{n}: {x}")));
        }
        v.extend(encode("S_ENDPGM", "ENC_SOPP", &[], None).unwrap());
        v
    }

    /// A loop that only ends when its counter wraps (the guards end it), then
    /// global_store_dword [v0 * 4], the counter (`recomp::tests::endless_loop_program`).
    fn endless_loop() -> Vec<u8> {
        prog(&[
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 128)], None),
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 129)], None),
            ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 10), ("SSRC1", 128)], None),
            ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFD)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 10)], None),
            ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 130), ("VSRC1", 0)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 128)], None),
            ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 6), ("DATA", 4), ("SADDR", 0x7D)], None),
        ])
    }

    /// The same loop inside one of three passes (`recomp::tests::endless_loop_entered_three_times`).
    fn endless_loop_entered_three_times() -> Vec<u8> {
        prog(&[
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 11), ("SSRC0", 128)], None),
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 12), ("SSRC0", 128)], None),
            ("S_MOV_B32", "ENC_SOP1", &[("SDST", 10), ("SSRC0", 128)], None),
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 10), ("SSRC0", 10), ("SSRC1", 129)], None),
            ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 10), ("SSRC1", 128)], None),
            ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFFD)], None),
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 12), ("SSRC0", 12), ("SSRC1", 10)], None),
            ("S_ADD_U32", "ENC_SOP2", &[("SDST", 11), ("SSRC0", 11), ("SSRC1", 129)], None),
            ("S_CMP_LG_U32", "ENC_SOPC", &[("SSRC0", 11), ("SSRC1", 131)], None),
            ("S_CBRANCH_SCC1", "ENC_SOPP", &[("SIMM16", 0xFFF8)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 4), ("SRC0", 12)], None),
            ("V_LSHLREV_B32", "ENC_VOP2", &[("VDST", 6), ("SRC0", 130), ("VSRC1", 0)], None),
            ("V_MOV_B32", "ENC_VOP1", &[("VDST", 7), ("SRC0", 128)], None),
            ("GLOBAL_STORE_DWORD", "ENC_FLAT_GLBL", &[("ADDR", 6), ("DATA", 4), ("SADDR", 0x7D)], None),
        ])
    }

    fn compute(threads: u32, wave64: bool) -> Options {
        let cs = CsInterface { threads: [threads, 1, 1], user_sgprs: 2, tgid: [false; 3], tg_size: false, tid_components: 1 };
        Options { mode: Mode::Compute(cs), wave64, ..Default::default() }
    }

    #[test]
    fn the_switch_names_programs_or_counts_dispatches_only() {
        assert_eq!(parse(None), Switch::Off);
        assert_eq!(parse(Some("0")), Switch::Off);
        assert_eq!(parse(Some(" ")), Switch::Off);
        assert_eq!(parse(Some("1")), Switch::Host);
        assert_eq!(parse(Some("all")), Switch::All);
        assert_eq!(parse(Some("b78fa10b425b8e54, 0x24de2aef6c31d55e")), Switch::Named(vec![0xb78f_a10b_425b_8e54, 0x24de_2aef_6c31_d55e]));
        assert_eq!(parse(Some("nonsense")), Switch::Host);
        let s = Switch::Named(vec![0x1234]);
        assert!(named_in(&s, 0x1234));
        // the cs hash of the GPU time tables names the same program
        assert!(named_in(&Switch::Named(vec![cs_hash(0x99)]), 0x99));
        assert!(!named_in(&s, 0x99) && !named_in(&Switch::Host, 0x1234) && named_in(&Switch::All, 7));
    }

    #[test]
    fn every_guard_of_a_compute_loop_gets_its_counter() {
        let t = translate(&endless_loop(), &compute(64, false)).unwrap();
        assert!(t.complete(), "{:?}", t.unsupported_why);
        let (h, sites) = instrument(&t.hlsl, "res", 0).unwrap();
        assert_eq!(sites, Sites { budget: 1, caps: 1, exchange: 0, barrier: 0 }, "{}", t.hlsl);
        assert!(h.contains("if (xcl_n1 == 0u) { xcl_ls_exit(1u); break; } xcl_n1--;"), "{h}");
        assert!(h.contains("if (xcl_budget == 0u) { xcl_ls_exit(0u); break; } if (xcl_budget == 64512u) xcl_ls_lanes(6u); if (xcl_budget == 49152u) xcl_ls_lanes(7u); xcl_budget--;"), "{h}");
        assert!(!h.contains(BUDGET_GUARD) && !h.contains("== 0u) break;"), "{h}");
        // the helpers come right after the buffer they add to
        let decl = h.find("RWByteAddressBuffer res : register(u1);").unwrap();
        assert!(decl < h.find("void xcl_ls_add").unwrap() && h.find("void xcl_ls_add").unwrap() < h.find("xcl_ls_exit(1u)").unwrap());
        // nothing else changed
        let undone = h.replace(&helpers("res", 0), "").replace(" { xcl_ls_exit(1u); break; }", " break;").replace(
            "if (xcl_budget == 0u) { xcl_ls_exit(0u); break; } if (xcl_budget == 64512u) xcl_ls_lanes(6u); if (xcl_budget == 49152u) xcl_ls_lanes(7u); xcl_budget--;",
            BUDGET_GUARD,
        );
        assert_eq!(undone, t.hlsl);
    }

    #[test]
    fn the_polls_of_split_waves_and_safe_barriers_count_their_timeouts() {
        let h = "RWByteAddressBuffer io : register(u0);\nRWByteAddressBuffer res : register(u1);\n\
                 if (n >= (1u << 20)) { ok = 0u; break; }\n            if (_n >= (1u << 20)) {\n                _ok = 0u;\n";
        let (out, sites) = instrument(h, "res", 0).unwrap();
        assert_eq!(sites, Sites { budget: 0, caps: 0, exchange: 1, barrier: 1 });
        assert!(out.contains("if (n >= (1u << 20)) { xcl_ls_add(2u, 1u); ok = 0u; break; }"), "{out}");
        assert!(out.contains("if (_n >= (1u << 20)) { xcl_ls_add(3u, 1u);\n                _ok = 0u;"), "{out}");
    }

    #[test]
    fn a_program_that_uses_res_is_left_alone() {
        let h = "RWByteAddressBuffer res : register(u1);\nvoid main() { res.Store(0u, 1u); }\n";
        assert!(instrument(h, "res", 0).is_err());
        // another name that ends in res is not a use
        let h = "RWByteAddressBuffer res : register(u1);\nvoid main() { xcl_res.Store(0u, 1u); }\n";
        assert!(instrument(h, "res", 0).is_ok());
        assert!(instrument("void main() {}\n", "res", 0).is_err());
    }

    /// DXC takes the counted HLSL, split wave64 too (where the exchange polls are).
    #[test]
    fn counted_programs_compile() {
        if crate::gpu::recomp::dxc::location().is_err() {
            return runner::skip_test("no DXC");
        }
        for (wave64, split) in [(false, false), (true, false), (true, true)] {
            let t = translate(&endless_loop(), &Options { split_wave64: split, ..compute(256, wave64) }).unwrap();
            let (h, sites) = instrument(&t.hlsl, "res", 0).unwrap();
            assert!(sites.budget == 1 && sites.caps == 1);
            crate::gpu::recomp::dxc::compile(&h, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("wave64 {wave64} split {split}: {e}\n{h}"));
        }
    }

    /// On the GPU: a loop that never ends runs out of its entry's passes (65 536 without a
    /// dispatch share, 1024 with a share of 1024 a lane), one entered again and again spends the
    /// lane's budget, and the counters say which, lane by lane.
    #[test]
    fn gpu_counters_tell_a_spent_budget_from_a_capped_entry() {
        if runner::adapter().is_err() || crate::gpu::recomp::dxc::location().is_err() {
            return runner::skip_test("no GPU");
        }
        let threads = 64u32;
        // (the test runner binds `mem` at every window: the counters go past what the program stores)
        let base = 4096u32;
        let dxil_of = |code: &[u8]| {
            let t = translate(code, &compute(threads, false)).unwrap();
            let (h, _) = instrument(&t.hlsl, "mem", base).unwrap();
            crate::gpu::recomp::dxc::compile(&h, "main", "cs_6_6", &[]).unwrap_or_else(|e| panic!("{e}\n{h}"))
        };
        let (once, thrice) = (dxil_of(&endless_loop()), dxil_of(&endless_loop_entered_three_times()));
        let run = |dxil: &[u8], slot: u32| -> Vec<u32> {
            let mut io = [0u32; 128];
            io[IO_LOOP_BUDGET] = slot;
            let mem = runner::dispatch(dxil, [1, 1, 1], &io, &vec![0u32; 2048]).unwrap();
            mem[base as usize / 4..base as usize / 4 + WORDS].to_vec()
        };
        let words = |slot: u32| run(&once, slot);
        let n = threads;
        // no share: the entry's 65 536 passes end the loop (before the budget's check)
        let w = words(0);
        assert_eq!((w[W_CAP_LANES], w[W_CAP_EXITS], w[W_BUDGET_LANES]), (n, n, 0), "{w:?}");
        assert_eq!((w[W_PAST_1K], w[W_PAST_16K]), (n, n), "{w:?}");
        assert_eq!((w[W_X_TIMEOUTS], w[W_BAR_TIMEOUTS]), (0, 0), "{w:?}");
        // entered again with no share: the budget is spent within the first entry's passes
        let w = run(&thrice, 0);
        assert_eq!(w[W_BUDGET_LANES], n, "{w:?}");
        assert!(w[W_BUDGET_EXITS] >= n, "{w:?}");
        assert_eq!((w[W_PAST_1K], w[W_PAST_16K]), (n, n), "{w:?}");
        // a share of 1024 passes a lane: the entry's count ends it first
        let w = words(1);
        assert_eq!((w[W_CAP_LANES], w[W_CAP_EXITS]), (n, n), "{w:?}");
        assert_eq!((w[W_BUDGET_LANES], w[W_PAST_1K], w[W_PAST_16K]), (0, 0, 0), "{w:?}");
        // 5000 passes: past 1024, not past 16 384
        let w = words(threads * 5000);
        assert_eq!((w[W_CAP_LANES], w[W_PAST_1K], w[W_PAST_16K], w[W_BUDGET_LANES]), (n, n, 0, 0), "{w:?}");
    }
}
