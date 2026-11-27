//! `XCL_NV_CACHE_TRIM_GB=<n>` (off when unset): at the start of a run, before a D3D12 device
//! exists, the NVIDIA driver's shader cache files that earlier runs of this title mapped are
//! deleted when together they are over n GB.
//!
//! The driver maps such a file (`*.nvph` in `%LOCALAPPDATA%\NVIDIA\DXCache`) whole into the
//! process: 8 – 16 GB files, 4 – 10 GB of them resident, mostly entries of earlier shader
//! preparations that are never read again. Without it the driver makes a new one and compiles
//! what it needs (XCL's own blob cache, `XCL_PSO_DISK_CACHE`, still holds its pipelines).
//!
//! Which files: only the names the memory map saw mapped into this process
//! (`nv_cache_files.txt` in the title's logs folder, next to `xcl.log`), and only in DXCache.
//! Nothing else is deleted: no file of another program, nothing a run of this one did not map.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// The list of mapped cache files, in the title's logs folder.
pub const LIST: &str = "nv_cache_files.txt";

/// The file name of `path` (a mapped file as Windows names it) when it is an NVIDIA D3D shader
/// cache file: `…\NVIDIA\DXCache\<name>.nvph`.
pub fn nvph_name(path: &str) -> Option<&str> {
    let mut parts = path.rsplit(['\\', '/']);
    let file = parts.next()?;
    let dir = parts.next()?;
    let vendor = parts.next()?;
    (dir.eq_ignore_ascii_case("dxcache") && vendor.eq_ignore_ascii_case("nvidia") && valid_name(file)).then_some(file)
}

/// A plain `*.nvph` file name: no folder, drive or stream in it.
fn valid_name(n: &str) -> bool {
    n.len() > ".nvph".len() && n.len() <= 255 && n.to_ascii_lowercase().ends_with(".nvph") && !n.starts_with('.') && !n.contains(['\\', '/', ':'])
}

fn read_list(p: &Path) -> BTreeSet<String> {
    std::fs::read_to_string(p).unwrap_or_default().lines().map(str::trim).filter(|l| valid_name(l)).map(String::from).collect()
}

fn write_list(p: &Path, names: &BTreeSet<String>) -> std::io::Result<()> {
    if let Some(dir) = p.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut text = String::new();
    for n in names {
        text.push_str(n);
        text.push('\n');
    }
    std::fs::write(p, text)
}

/// Names already written this run.
static SEEN: Mutex<BTreeSet<String>> = Mutex::new(BTreeSet::new());

/// The mapped files `paths` (as Windows names them): the NVIDIA cache files among them are added
/// to the list in `logs`. Returns the names that were new to it.
pub fn note_mapped<'a>(logs: &Path, paths: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let mut seen = SEEN.lock().unwrap();
    let new: Vec<String> = paths.into_iter().filter_map(nvph_name).filter(|n| !seen.contains(*n)).map(String::from).collect::<BTreeSet<_>>().into_iter().collect();
    if new.is_empty() {
        return new;
    }
    let p = logs.join(LIST);
    let mut list = read_list(&p);
    list.extend(new.iter().cloned());
    if let Err(e) = write_list(&p, &list) {
        crate::xlog_warn!("NVIDIA shader cache: cannot write {}: {e}", p.display());
        return Vec::new();
    }
    seen.extend(new.iter().cloned());
    new
}

/// What a trim found and did.
#[derive(Debug, Default, PartialEq)]
pub struct Trim {
    /// listed files that are in DXCache: how many, bytes in all
    pub files: usize,
    pub bytes: u64,
    /// deleted (name, bytes)
    pub deleted: Vec<(String, u64)>,
    /// not deleted (name, why)
    pub failed: Vec<(String, String)>,
}

/// Deletes the files the list in `logs` names in `dxcache` when together they are over `limit`
/// bytes. The list then keeps only the names of files that are still there.
pub fn trim(logs: &Path, dxcache: &Path, limit: u64) -> Trim {
    let p = logs.join(LIST);
    let names = read_list(&p);
    let mut t = Trim::default();
    let mut found = Vec::new();
    for n in &names {
        // (a plain file only: not a link to somewhere else)
        if let Ok(m) = std::fs::symlink_metadata(dxcache.join(n)) {
            if m.is_file() {
                t.files += 1;
                t.bytes += m.len();
                found.push((n.clone(), m.len()));
            }
        }
    }
    let mut kept: BTreeSet<String> = found.iter().map(|f| f.0.clone()).collect();
    if t.bytes > limit {
        for (n, len) in found {
            match std::fs::remove_file(dxcache.join(&n)) {
                Ok(()) => {
                    kept.remove(&n);
                    t.deleted.push((n, len));
                }
                Err(e) => t.failed.push((n, e.to_string())),
            }
        }
    }
    if kept != names {
        let _ = write_list(&p, &kept);
    }
    t
}

/// `XCL_NV_CACHE_TRIM_GB` in bytes (unset or not a number: off).
fn limit() -> Option<u64> {
    std::env::var("XCL_NV_CACHE_TRIM_GB").ok()?.trim().parse::<u64>().ok().map(|gb| gb.saturating_mul(1 << 30))
}

/// Whether `XCL_NV_CACHE_TRIM_GB` is set (the memory map then records the mapped files even
/// when it is off).
pub fn enabled() -> bool {
    limit().is_some()
}

/// `%LOCALAPPDATA%\NVIDIA\DXCache`.
fn dxcache() -> Option<PathBuf> {
    Some(PathBuf::from(std::env::var_os("LOCALAPPDATA")?).join("NVIDIA").join("DXCache"))
}

/// The trim at the start of a run, with the title's logs folder `logs` (call before a D3D12
/// device is made: the driver maps the file when it makes one).
pub fn trim_at_start(logs: &Path) {
    let Some(limit) = limit() else { return };
    let Some(dir) = dxcache() else {
        crate::xlog_warn!("NVIDIA shader cache: no LOCALAPPDATA – not trimmed");
        return;
    };
    let t = trim(logs, &dir, limit);
    let mb = |b: u64| b >> 20;
    if t.deleted.is_empty() && t.failed.is_empty() {
        crate::xlog_info!(
            "NVIDIA shader cache: {} files earlier runs mapped, {} MB in all (XCL_NV_CACHE_TRIM_GB={}) – kept",
            t.files,
            mb(t.bytes),
            limit >> 30
        );
        return;
    }
    for (n, len) in &t.deleted {
        crate::xlog_info!("NVIDIA shader cache: deleted {} ({} MB)", dir.join(n).display(), mb(*len));
    }
    for (n, why) in &t.failed {
        crate::xlog_warn!("NVIDIA shader cache: cannot delete {}: {why}", dir.join(n).display());
    }
    crate::xlog_info!(
        "NVIDIA shader cache: {} of {} files earlier runs mapped deleted, {} MB of {} MB (over XCL_NV_CACHE_TRIM_GB={})",
        t.deleted.len(),
        t.files,
        mb(t.deleted.iter().map(|d| d.1).sum()),
        mb(t.bytes),
        limit >> 30
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Dir(PathBuf);
    impl Drop for Dir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn temp(tag: &str) -> Dir {
        let d = std::env::temp_dir().join(format!("xcl-nv-cache-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        Dir(d)
    }

    #[test]
    fn only_nvidia_dxcache_nvph_files_are_named() {
        let p = r"\Device\HarddiskVolume3\Users\u\AppData\Local\NVIDIA\DXCache\4ac1c6b8a9d2d4a1_0.nvph";
        assert_eq!(nvph_name(p), Some("4ac1c6b8a9d2d4a1_0.nvph"));
        assert_eq!(nvph_name(r"C:\x\nvidia\dxcache\A.NVPH"), Some("A.NVPH"));
        assert_eq!(nvph_name(r"\Device\X\NVIDIA\DXCache\7f1c_0.toc"), None, "not a .nvph");
        assert_eq!(nvph_name(r"\Device\X\NVIDIA\GLCache\a.nvph"), None, "not DXCache");
        assert_eq!(nvph_name(r"\Device\X\Other\DXCache\a.nvph"), None, "not NVIDIA's");
        assert_eq!(nvph_name(r"\Device\X\NVIDIA\DXCache\sub\a.nvph"), None, "not directly in DXCache");
        assert_eq!(nvph_name("a.nvph"), None);
        assert!(!valid_name(".nvph") && !valid_name("..\\a.nvph") && !valid_name("c:a.nvph") && !valid_name("a.nvph:s"));
    }

    #[test]
    fn mapped_names_are_added_to_the_list_once() {
        let logs = temp("list");
        let a = r"\Device\X\NVIDIA\DXCache\aaaa_0.nvph";
        let b = r"\Device\X\NVIDIA\DXCache\bbbb_0.nvph";
        assert_eq!(note_mapped(&logs.0, [a, a, r"\Device\X\Windows\System32\d3d12.dll"]), ["aaaa_0.nvph"]);
        assert!(note_mapped(&logs.0, [a]).is_empty(), "already written");
        assert_eq!(note_mapped(&logs.0, [b]), ["bbbb_0.nvph"]);
        let text = std::fs::read_to_string(logs.0.join(LIST)).unwrap();
        assert_eq!(text, "aaaa_0.nvph\nbbbb_0.nvph\n");
    }

    #[test]
    fn the_trim_deletes_only_listed_files_in_dxcache_and_only_over_the_limit() {
        let logs = temp("trim-logs");
        let cache = temp("trim-cache");
        let other = temp("trim-other");
        std::fs::write(cache.0.join("old_0.nvph"), vec![0u8; 3000]).unwrap();
        std::fs::write(cache.0.join("old_1.nvph"), vec![0u8; 2000]).unwrap();
        std::fs::write(cache.0.join("other_0.nvph"), vec![0u8; 9000]).unwrap(); // not listed
        std::fs::write(cache.0.join("old_0.toc"), b"x").unwrap();
        std::fs::write(other.0.join("old_0.nvph"), b"x").unwrap(); // listed name, another folder
        let list = "old_0.nvph\nold_1.nvph\ngone_0.nvph\n..\\escape.nvph\nC:\\x\\y.nvph\nold_0.toc\n";
        std::fs::write(logs.0.join(LIST), list).unwrap();

        let under = trim(&logs.0, &cache.0, 5000);
        assert_eq!((under.files, under.bytes), (2, 5000));
        assert!(under.deleted.is_empty() && under.failed.is_empty(), "5000 bytes is not over 5000");
        assert_eq!(std::fs::read_to_string(logs.0.join(LIST)).unwrap(), "old_0.nvph\nold_1.nvph\n", "names of files not there are dropped");

        let over = trim(&logs.0, &cache.0, 4999);
        assert_eq!(over.deleted, [("old_0.nvph".to_string(), 3000), ("old_1.nvph".to_string(), 2000)]);
        assert!(over.failed.is_empty());
        assert!(!cache.0.join("old_0.nvph").exists() && !cache.0.join("old_1.nvph").exists());
        assert!(cache.0.join("other_0.nvph").exists(), "a file no run of this title mapped stays");
        assert!(cache.0.join("old_0.toc").exists() && other.0.join("old_0.nvph").exists(), "nothing else is touched");
        assert_eq!(std::fs::read_to_string(logs.0.join(LIST)).unwrap(), "");

        let none = trim(&logs.0, &cache.0, 0);
        assert_eq!(none, Trim::default(), "an empty list deletes nothing");
        let missing = trim(&other.0.join("no-logs"), &cache.0, 0);
        assert_eq!(missing, Trim::default(), "no list: nothing");
    }
}
