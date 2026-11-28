//! Which records' programs get the constant data after their code (`Options::rodata`), for
//! narrowing down a fault to some of them:
//!
//! - `XCL_RODATA` unset, empty or `all`: every program (as without this switch);
//! - `XCL_RODATA=none` (or `0`): none – every load through `s_getpc_b64` reads zeros again;
//! - `XCL_RODATA=<list>`: only the records the list names. An entry of 16 hex digits (an
//!   optional `0x`) is a record hash (`shader::fnv64` of the container, the `record` of the
//!   driver compile lines and `xcl shaders --record-hash`); anything else is a part of a
//!   record name (`postfx_aces:CS_Bake…`, case ignored). Entries are split at `,` or `;`;
//!   `@<path>` takes the entries of a file, one per line (`<hash> <name>` lines: the hash).
//! - `XCL_RODATA_SKIP=<list>`, the same form: these records never get it.
//!
//! Only a program that takes a code address (`takes_code_address`) and has data after it is
//! gated: the others keep the data they had, so their translation does not change.
//!
//! Names come from the shader preparation before the title starts (`note_record`): the title
//! passes containers only. With `XCL_PREPARE=0` only hashes match.
//!
//! A withheld record's translation differs from the full one under the same bytes, so its
//! disk cache entries go under another identity (`translated::cached`); the records that get
//! the data keep the entries of a run without this switch.
//!
//! With the switch set (or `XCL_RODATA_LIST=1`), the preparation logs every record that takes
//! a code address once, with its hash, name and whether it gets the data, and a count.

use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};

use crate::gpu::xbox;

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Pat {
    Hash(u64),
    Name(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Gate {
    /// None: every record (`all`); Some(empty): none
    pub only: Option<Vec<Pat>>,
    pub skip: Vec<Pat>,
    pub list: bool,
}

fn pat(e: &str) -> Pat {
    let hex = e.strip_prefix("0x").or_else(|| e.strip_prefix("0X")).unwrap_or(e);
    match (hex.len() == 16).then(|| u64::from_str_radix(hex, 16).ok()).flatten() {
        Some(h) => Pat::Hash(h),
        None => Pat::Name(e.to_ascii_lowercase()),
    }
}

/// `@<path>`: the entries of a file, one per line – its first word (a line `<hash> <name>`
/// is the hash); blank lines and `#` comments are left out.
fn file_entries(text: &str) -> Vec<Pat> {
    text.lines()
        .map(|l| l.split('#').next().unwrap_or("").split_whitespace().next().unwrap_or(""))
        .filter(|e| !e.is_empty())
        .map(pat)
        .collect()
}

fn parse_list(s: &str) -> Vec<Pat> {
    let mut out = Vec::new();
    for e in s.split([',', ';']).map(str::trim).filter(|e| !e.is_empty()) {
        match e.strip_prefix('@') {
            Some(path) => match std::fs::read_to_string(path) {
                Ok(text) => out.extend(file_entries(&text)),
                Err(err) => crate::xlog_warn!("[RODATA] {path}: {err} – no entries from it"),
            },
            None => out.push(pat(e)),
        }
    }
    out
}

impl Gate {
    pub(crate) fn parse(only: Option<&str>, skip: Option<&str>, list: Option<&str>) -> Gate {
        let only = match only.map(str::trim) {
            None | Some("") => None,
            Some(v) if v.eq_ignore_ascii_case("all") => None,
            Some(v) if v.eq_ignore_ascii_case("none") || v == "0" => Some(Vec::new()),
            Some(v) => Some(parse_list(v)),
        };
        let skip = skip.map(parse_list).unwrap_or_default();
        let list = list.is_some_and(|v| v.trim() == "1");
        Gate { only, skip, list }
    }

    /// Anything else than every record getting the data.
    pub(crate) fn narrows(&self) -> bool {
        self.only.is_some() || !self.skip.is_empty()
    }

    fn names_wanted(&self) -> bool {
        let name = |p: &Pat| matches!(p, Pat::Name(_));
        self.only.as_ref().is_some_and(|o| o.iter().any(name)) || self.skip.iter().any(name)
    }

    /// Whether record `hash`, under `names`, gets the data.
    pub(crate) fn gives(&self, hash: u64, names: &[String]) -> bool {
        let hit = |p: &Pat| match p {
            Pat::Hash(h) => *h == hash,
            Pat::Name(n) => names.iter().any(|name| name.to_ascii_lowercase().contains(n.as_str())),
        };
        self.only.as_ref().is_none_or(|o| o.iter().any(hit)) && !self.skip.iter().any(hit)
    }

    fn describe(&self) -> String {
        let fmt = |l: &[Pat]| l.iter().map(|p| match p {
            Pat::Hash(h) => format!("{h:016x}"),
            Pat::Name(n) => format!("*{n}*"),
        }).collect::<Vec<_>>().join(",");
        let only = match &self.only {
            None => "all".to_string(),
            Some(l) if l.is_empty() => "none".to_string(),
            Some(l) => format!("only {}", fmt(l)),
        };
        if self.skip.is_empty() { only } else { format!("{only}, skip {}", fmt(&self.skip)) }
    }
}

fn gate() -> &'static Gate {
    static G: OnceLock<Gate> = OnceLock::new();
    G.get_or_init(|| {
        let var = |n: &str| std::env::var(n).ok();
        let g = Gate::parse(var("XCL_RODATA").as_deref(), var("XCL_RODATA_SKIP").as_deref(), var("XCL_RODATA_LIST").as_deref());
        if g.narrows() || g.list {
            crate::xlog_info!(
                "[RODATA] constant data after the program (XCL_RODATA / XCL_RODATA_SKIP): {}{}",
                g.describe(),
                if g.names_wanted() { " – names come from the shader preparation (XCL_PREPARE=0: hashes only)" } else { "" }
            );
        }
        g
    })
}

/// Record names the preparation saw, by hash.
fn names() -> &'static Mutex<HashMap<u64, Vec<String>>> {
    static N: OnceLock<Mutex<HashMap<u64, Vec<String>>>> = OnceLock::new();
    N.get_or_init(Default::default)
}

fn names_of(hash: u64) -> Vec<String> {
    names().lock().unwrap().get(&hash).cloned().unwrap_or_default()
}

/// Whether the program `p` of `f` is one the switch decides for: data after it, and an
/// `s_getpc_b64` that reaches it.
fn gated_program(f: &xbox::Fssm, p: &xbox::Program) -> bool {
    !f.rodata(p).is_empty()
        && f.data.get(p.offset..p.offset + p.len).is_some_and(super::takes_code_address)
}

fn takes_code_address(sh: &xbox::XboxShader) -> bool {
    sh.fssm.iter().any(|f| f.programs.iter().any(|p| gated_program(f, p)))
}

/// `f.rodata(p)` for the program of record `blob`, or nothing when the switch withholds it.
pub(crate) fn rodata(blob: &[u8], f: &xbox::Fssm, p: &xbox::Program) -> Vec<(u32, Vec<u32>)> {
    let g = gate();
    if !g.narrows() || !gated_program(f, p) {
        return f.rodata(p);
    }
    let hash = crate::gpu::shader::fnv64(blob);
    if g.gives(hash, &names_of(hash)) { f.rodata(p) } else { Vec::new() }
}

/// Whether the switch withholds the data from a program of `blob` (its translation then
/// differs from the full one: `translated::cached`).
pub(crate) fn withheld(blob: &[u8]) -> bool {
    withheld_by(gate(), blob)
}

fn withheld_by(g: &Gate, blob: &[u8]) -> bool {
    if !g.narrows() {
        return false;
    }
    let hash = crate::gpu::shader::fnv64(blob);
    !g.gives(hash, &names_of(hash)) && xbox::parse(blob).is_some_and(|sh| takes_code_address(&sh))
}

static SEEN: AtomicUsize = AtomicUsize::new(0);
static GIVEN: AtomicUsize = AtomicUsize::new(0);

/// The preparation's record `name` with container `blob` (`sh`): its name for the name
/// entries, and with the switch set, one log line for a record that takes a code address.
/// Call before the record is translated.
pub(crate) fn note_record(blob: &[u8], sh: &xbox::XboxShader, name: &str) {
    let g = gate();
    if !g.narrows() && !g.list {
        return;
    }
    let hash = crate::gpu::shader::fnv64(blob);
    // the record name without the pack offset (`name @0x…`)
    let name = name.split(" @0x").next().unwrap_or(name).to_string();
    {
        let mut n = names().lock().unwrap();
        let e = n.entry(hash).or_default();
        if e.contains(&name) {
            return;
        }
        e.push(name.clone());
    }
    if !takes_code_address(sh) {
        return;
    }
    let gives = g.gives(hash, &names_of(hash));
    static LOGGED: OnceLock<Mutex<HashSet<u64>>> = OnceLock::new();
    if !LOGGED.get_or_init(Default::default).lock().unwrap().insert(hash) {
        return;
    }
    let seen = SEEN.fetch_add(1, Ordering::Relaxed) + 1;
    let given = GIVEN.fetch_add(gives as usize, Ordering::Relaxed) + gives as usize;
    crate::xlog_info!("[RODATA] {hash:016x} {name}: {} ({given} of {seen} so far)", if gives { "given" } else { "withheld" });
}

/// One line after the preparation: how many records take a code address and get the data.
pub(crate) fn summary() {
    let g = gate();
    if g.narrows() || g.list {
        crate::xlog_info!(
            "[RODATA] {}: {} of {} records that take a code address get their constant data",
            g.describe(),
            GIVEN.load(Ordering::Relaxed),
            SEEN.load(Ordering::Relaxed)
        );
    }
}

#[cfg(test)]
mod tests {
    use super::{Gate, Pat};

    #[test]
    fn unset_or_all_gives_every_record() {
        for v in [None, Some(""), Some("all"), Some(" ALL ")] {
            let g = Gate::parse(v, None, None);
            assert!(!g.narrows(), "{v:?}");
            assert!(g.gives(1, &[]));
        }
    }

    #[test]
    fn none_gives_no_record() {
        for v in ["none", "0", "None"] {
            let g = Gate::parse(Some(v), None, None);
            assert!(g.narrows());
            assert!(!g.gives(0x14f3_3bcc_cd1e_c01d, &["postfx_aces:CS_BakeACESLUT".into()]));
        }
    }

    #[test]
    fn hashes_and_name_parts_select_records() {
        let g = Gate::parse(Some("14f33bcccd1ec01d; 0x56C0CD3B82A00C6E, CS_Bake"), None, None);
        assert_eq!(g.only.as_ref().unwrap()[0], Pat::Hash(0x14f3_3bcc_cd1e_c01d));
        assert_eq!(g.only.as_ref().unwrap()[1], Pat::Hash(0x56c0_cd3b_82a0_0c6e));
        assert!(g.gives(0x14f3_3bcc_cd1e_c01d, &[]));
        assert!(g.gives(0x56c0_cd3b_82a0_0c6e, &[]));
        assert!(g.gives(7, &["postfx_aces:cs_bakeACESWithGamutCompressLUT".into()]));
        assert!(!g.gives(7, &["prefilter_env_map:CS_Main".into()]));
        assert!(!g.gives(7, &[]), "a name entry needs the record's name");
        // not 16 hex digits: a name part
        assert_eq!(Gate::parse(Some("abc123"), None, None).only.unwrap(), vec![Pat::Name("abc123".into())]);
    }

    #[test]
    fn skip_wins_over_the_allow_list() {
        let g = Gate::parse(None, Some("scattering_"), None);
        assert!(g.narrows());
        assert!(g.gives(1, &["postfx_aces:CS_BakeLUT".into()]));
        assert!(!g.gives(1, &["scattering_transmittance_system:CS".into()]));
        let g = Gate::parse(Some("postfx_aces"), Some("0000000000000001"), None);
        assert!(!g.gives(1, &["postfx_aces:CS_BakeLUT".into()]));
        assert!(g.gives(2, &["postfx_aces:CS_BakeLUT".into()]));
    }

    /// Only a container with a program that takes a code address changes its cache identity.
    #[test]
    fn only_records_with_code_addresses_are_withheld() {
        let none = Gate::parse(Some("none"), None, None);
        assert!(!super::withheld_by(&none, b"not a console container"));
        assert!(!super::withheld_by(&Gate::parse(None, None, None), b"not a console container"));
        // program (`s_getpc_b64 s[0:1]` or not), guard `s_endpgm`, table (u32 N, N dwords), `s_endpgm`
        let blob = |getpc: bool| {
            let mut code = Vec::new();
            let head: &[u32] = if getpc { &[0xBE80_1F00, 0xBF81_0000] } else { &[0xBE80_0301, 0xBF81_0000] };
            for w in head.iter().chain(&[0xBF81_0000u32, 4, 0x3e63_3333, 0x3df5_c28f, 0, 1, 0xBF81_0000]) {
                code.extend_from_slice(&w.to_le_bytes());
            }
            use crate::gpu::xbox::tests::{fssm, sample};
            sample(1, &[fssm(&code, &[(4, 0)])])
        };
        let (with, without) = (blob(true), blob(false));
        let sh = crate::gpu::xbox::parse(&with).unwrap();
        assert!(!sh.fssm[0].rodata(&sh.fssm[0].programs[0]).is_empty(), "the table is the program's data");
        assert!(super::withheld_by(&none, &with));
        assert!(!super::withheld_by(&none, &without), "no s_getpc_b64: the data stays, the identity too");
        let hash = format!("{:016x}", crate::gpu::shader::fnv64(&with));
        assert!(!super::withheld_by(&Gate::parse(Some(&hash), None, None), &with));
        assert!(super::withheld_by(&Gate::parse(None, Some(&hash), None), &with));
        assert!(!super::withheld_by(&Gate::parse(Some("all"), None, None), &with));
    }

    #[test]
    fn files_give_one_entry_per_line() {
        let e = super::file_entries("# comment
14f33bcccd1ec01d	postfx_aces:CS_Bake

  water_  # part
");
        assert_eq!(e, vec![Pat::Hash(0x14f3_3bcc_cd1e_c01d), Pat::Name("water_".into())]);
        let path = std::env::temp_dir().join(format!("xcl-rodata-{}.txt", std::process::id()));
        std::fs::write(&path, "0000000000000002 b
").unwrap();
        let g = Gate::parse(Some(&format!("0000000000000001,@{}", path.display())), None, None);
        let _ = std::fs::remove_file(&path);
        assert!(g.gives(1, &[]) && g.gives(2, &[]) && !g.gives(3, &[]));
        assert_eq!(Gate::parse(Some(r"@Z:\no\such\file"), None, None).only, Some(Vec::new()), "an unreadable file gives nothing");
    }

    #[test]
    fn list_alone_does_not_narrow() {
        let g = Gate::parse(None, None, Some("1"));
        assert!(g.list && !g.narrows());
    }
}
