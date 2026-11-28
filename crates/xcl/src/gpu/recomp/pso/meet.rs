//! Bounds on the translation's own waits, off by default, after VIctory-NVIDIA
//! (crates/victory-shader/src/emit/lower.rs @9ce3cf9e, GPL-3.0-or-later: `MEETING_POLLS`,
//! `meet`; the technique, not its code), modified for XCL 2026-11-28.
//!
//! On a GPU of 32-lane waves a console wave64 program runs as two wave32 halves that meet
//! wherever a value has to be the whole wave's (`emit` `PRELUDE_X64`, `xcl_x_meet`): one lane of
//! each half polls an LDS counter for the other half, up to 2^20 times. An `s_barrier` with early
//! exits waits the same way (`PRELUDE_SAFE_BARRIER`, `xcl_s_barrier`). A half that gives up
//! (`xcl_x_dead`) reads the other half's masks as 0 from then on, and the console's loops then
//! run on half the wave's bits, bounded only by the lane's loop budget (65 536, `xcl_budget`).
//! VIctory polls a meeting 4 096 times ("a partner that never comes costs wrong values, not a
//! hung card") and cuts every compute loop at 4 096.
//!
//! - `XCL_X64_POLLS=<n>`: both waits poll at most `n` times.
//! - `XCL_X64_DEAD_BUDGET=1`: a wait that gives up also spends the lane's loop budget, so the
//!   wave's loops leave at their next check.
//! - Only records with constant data after a program (#2229, `rodata_gate`) by default;
//!   `XCL_X64_POLLS_ALL=1`: every record whose translation may have such a wait.
//!
//! The edits are of the recompiler's HLSL (the recompiler is unchanged, build.rs
//! `TRANSLATOR_SOURCES` too); a record they may change is cached under another identity
//! (`identity`), every other one keeps its entries.

use std::sync::OnceLock;

use crate::gpu::{rdna2, xbox};

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct Switches {
    pub polls: Option<u32>,
    pub dead_budget: bool,
    pub all: bool,
}

impl Switches {
    pub(crate) fn parse(polls: Option<&str>, dead: Option<&str>, all: Option<&str>) -> Switches {
        Switches {
            polls: polls.and_then(|v| v.trim().parse::<u32>().ok()).filter(|&n| n > 0),
            dead_budget: dead.is_some_and(|v| v.trim() == "1"),
            all: all.is_some_and(|v| v.trim() == "1"),
        }
    }

    pub(crate) fn any(&self) -> bool {
        self.polls.is_some() || self.dead_budget
    }

    fn tag(&self) -> String {
        format!("xcl-x64-polls-{}-dead-{}", self.polls.unwrap_or(0), self.dead_budget as u8)
    }
}

pub(crate) fn switches() -> &'static Switches {
    static S: OnceLock<Switches> = OnceLock::new();
    S.get_or_init(|| {
        let var = |n: &str| std::env::var(n).ok();
        let s = Switches::parse(var("XCL_X64_POLLS").as_deref(), var("XCL_X64_DEAD_BUDGET").as_deref(), var("XCL_X64_POLLS_ALL").as_deref());
        if s.any() {
            crate::xlog_info!(
                "[MEET] split wave64 meetings and safe barriers: polls {} (default 1048576), dead wave spends its loop budget: {}, for {}",
                s.polls.map_or("unchanged".to_string(), |n| n.to_string()),
                s.dead_budget,
                if s.all { "every record (XCL_X64_POLLS_ALL=1)" } else { "records with constant data after a program" }
            );
        }
        s
    })
}

const MEET_POLL: &str = "if (n >= (1u << 20)) { ok = 0u; break; }";
const MEET_DEAD: &str = "if (WaveReadLaneFirst(ok) == 0u) { xcl_x_dead = true; return false; }";
const BAR_POLL: &str = "if (_n >= (1u << 20)) {";
const BAR_DEAD: &str = "if (WaveReadLaneFirst(_ok) == 0u) xcl_bar_alive = false;";

/// `hlsl` with the waits bounded as `s` says (unchanged where it has none).
pub(crate) fn edit(s: &Switches, hlsl: &str) -> String {
    let mut h = hlsl.to_string();
    if let Some(n) = s.polls {
        h = h.replacen(MEET_POLL, &format!("if (n >= {n}u) {{ ok = 0u; break; }}"), 1);
        h = h.replacen(BAR_POLL, &format!("if (_n >= {n}u) {{"), 1);
    }
    if s.dead_budget {
        h = h.replacen(MEET_DEAD, "if (WaveReadLaneFirst(ok) == 0u) { xcl_x_dead = true; xcl_budget = 0u; return false; }", 1);
        h = h.replacen(BAR_DEAD, "if (WaveReadLaneFirst(_ok) == 0u) { xcl_bar_alive = false; xcl_budget = 0u; }", 1);
    }
    h
}

/// Whether program `code` may wait for other waves in its translation: a wave64 program (split
/// on a GPU of 32-lane waves, `split`), or one with `s_barrier`.
fn may_wait(f: &xbox::Fssm, code: &[u8], split: bool) -> bool {
    (split && super::super::fssm_wave64(f, code)) || has_barrier(code)
}

fn has_barrier(code: &[u8]) -> bool {
    let mut off = 0usize;
    while off + 4 <= code.len() {
        match rdna2::decode(&code[off..], off as u64) {
            Ok(d) if d.inst.name == "S_BARRIER" => return true,
            Ok(d) => off += d.len,
            Err(_) => off += 4,
        }
    }
    false
}

/// Whether the switches may change record `blob`'s translation (`split`: the split wave64 form).
pub(crate) fn changes(s: &Switches, blob: &[u8], split: bool) -> bool {
    if !s.any() {
        return false;
    }
    if let Some(f) = super::fspx(blob) {
        // an FSPX container has no constant data after its program
        return s.all && ((split && super::super::detect_wave64(f.code)) || has_barrier(f.code));
    }
    let Some(sh) = xbox::parse(blob) else { return false };
    sh.fssm.iter().any(|f| {
        f.programs.iter().any(|p| {
            let Some(code) = f.data.get(p.offset..p.offset + p.len) else { return false };
            (s.all || !super::rodata_gate::rodata(blob, f, p).is_empty()) && may_wait(f, code, split)
        })
    })
}

/// Vertex and pixel programs have neither wait (their wave64 form on 32-lane GPUs is
/// `upper_lanes`, without meetings).
fn waits_in(stage: super::Stage) -> bool {
    matches!(stage, super::Stage::Compute | super::Stage::Mesh)
}

/// The translation of record `blob` for `stage` with the switches that are on.
pub(crate) fn apply(hlsl: String, blob: &[u8], stage: super::Stage, split: bool) -> String {
    let s = switches();
    if waits_in(stage) && changes(s, blob, split) { edit(s, &hlsl) } else { hlsl }
}

/// `ident` for `blob` in `stage`, under another identity when the switches may change it.
pub(crate) fn identity(ident: u64, blob: &[u8], stage: super::Stage, split: bool) -> u64 {
    if !waits_in(stage) {
        return ident;
    }
    identity_with(switches(), ident, blob, split)
}

pub(crate) fn identity_with(s: &Switches, ident: u64, blob: &[u8], split: bool) -> u64 {
    if ident == 0 || !changes(s, blob, split) {
        return ident;
    }
    super::translated::identity_for(&format!("{ident:016x}"), &s.tag())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switches_parse() {
        assert!(!Switches::parse(None, None, None).any());
        assert!(!Switches::parse(Some("0"), Some("0"), Some("1")).any());
        let s = Switches::parse(Some(" 4096 "), Some("1"), Some("1"));
        assert_eq!(s, Switches { polls: Some(4096), dead_budget: true, all: true });
    }

    #[test]
    fn both_waits_are_edited() {
        let h = format!("a\n{MEET_POLL}\n{MEET_DEAD}\nb\n{BAR_POLL}\n{BAR_DEAD}\n");
        let e = edit(&Switches::parse(Some("4096"), Some("1"), None), &h);
        assert!(e.contains("if (n >= 4096u) { ok = 0u; break; }") && e.contains("if (_n >= 4096u) {"), "{e}");
        assert!(e.contains("xcl_x_dead = true; xcl_budget = 0u; return false;") && e.contains("xcl_bar_alive = false; xcl_budget = 0u; }"), "{e}");
        assert!(!e.contains("1u << 20"));
        let only_polls = edit(&Switches::parse(Some("64"), None, None), &h);
        assert!(only_polls.contains(MEET_DEAD) && only_polls.contains(BAR_DEAD));
        assert_eq!(edit(&Switches::default(), &h), h, "off: unchanged");
    }

    /// Only a record the switches may change moves to another identity, one per setting.
    #[test]
    fn identities() {
        // s_mov ; s_endpgm   and   s_barrier ; s_endpgm, with or without data after them
        let words = |w: &[u32]| w.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>();
        let table = [0xBF81_0000u32, 4, 0x3e63_3333, 0x3df5_c28f, 0, 1, 0xBF81_0000];
        let cs = |head: &[u32], data: bool| {
            let mut w = head.to_vec();
            if data {
                w.extend(table);
            }
            use crate::gpu::xbox::tests::{fssm, sample};
            sample(1, &[fssm(&words(&w), &[(6, 0)])])
        };
        let plain = cs(&[0xBE80_0301, 0xBF81_0000], false);
        let barrier = cs(&[0xBF8A_0000, 0xBF81_0000], false);
        let barrier_data = cs(&[0xBF8A_0000, 0xBF81_0000], true);
        let on = Switches::parse(Some("4096"), Some("1"), None);
        let all = Switches::parse(Some("4096"), Some("1"), Some("1"));
        assert_eq!(identity_with(&Switches::default(), 7, &barrier_data, false), 7, "off");
        assert_eq!(identity_with(&on, 7, &barrier, false), 7, "no constant data");
        assert_ne!(identity_with(&on, 7, &barrier_data, false), 7);
        assert_ne!(identity_with(&all, 7, &barrier, false), 7);
        assert_eq!(identity_with(&all, 7, &plain, false), 7, "no wait");
        assert_ne!(identity_with(&all, 7, &barrier, false), identity_with(&Switches::parse(Some("64"), Some("1"), Some("1")), 7, &barrier, false));
        assert_eq!(identity_with(&all, 0, &barrier, false), 0, "cache off");
    }
}
