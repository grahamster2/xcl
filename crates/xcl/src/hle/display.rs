//! The console's display (TV) mode as the title sees it, `XCL_DISPLAY=<width>x<height>[@<hz>]`.
//!
//! Unset, nothing changes: the title's window is 1920x1080 (as on every console) and the
//! display output it reaches through its device (`IDXGIDevice` → adapter → `EnumOutputs`) is
//! the PC's own monitor. Set (`1920x1080`, `2560x1440`, `1920x1080@120`; 60 Hz when no rate
//! is given), every report agrees with that mode:
//! - the title's window and its client area are that size (`hle::user`);
//! - the adapter the title gets from its device has one output whose desktop is that size and
//!   whose only display mode is that mode (`d3d12x::dxgi_output`);
//! - the frame clock takes the 120 Hz intervals only when the mode is 120 Hz or more
//!   (`d3d12x::frame`).
//!
//! Many console titles do not ask: they pick their output size from the console model
//! (`XSystemGetDeviceType`) and let the console scale it to the TV. For those this changes
//! nothing; a title that does size its output from one of these reports presents at that size,
//! which the XCL window shows (the swap chain follows the planes' size and is stretched).

use std::sync::OnceLock;

/// what a console title's window is without `XCL_DISPLAY`
pub const DEFAULT_WINDOW: (u32, u32) = (1920, 1080);
const DEFAULT_HZ: u32 = 60;
const MIN_SIDE: (u32, u32) = (640, 360);
const MAX_SIDE: (u32, u32) = (7680, 4320);

/// A display mode: size in pixels and refresh rate in Hz.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mode {
    pub width: u32,
    pub height: u32,
    pub refresh_hz: u32,
}

/// `XCL_DISPLAY`'s value: `None` when unset or empty, an error for a value that names no mode.
pub fn parse(raw: Option<&str>) -> Result<Option<Mode>, String> {
    let Some(s) = raw.map(str::trim).filter(|s| !s.is_empty()) else { return Ok(None) };
    let (size, hz) = match s.split_once('@') {
        Some((size, hz)) => {
            let hz = hz.trim().trim_end_matches("hz").trim_end_matches("Hz").trim_end_matches("HZ").trim();
            (size, Some(hz.parse::<u32>().map_err(|_| format!("{s:?}: the refresh rate after '@' is not a number"))?))
        }
        None => (s, None),
    };
    let (w, h) = size
        .split_once(['x', 'X', '*'])
        .ok_or_else(|| format!("{s:?}: expected <width>x<height>, e.g. 1920x1080"))?;
    let num = |v: &str, what: &str| v.trim().parse::<u32>().map_err(|_| format!("{s:?}: the {what} is not a number"));
    let (width, height) = (num(w, "width")?, num(h, "height")?);
    if !(MIN_SIDE.0..=MAX_SIDE.0).contains(&width) || !(MIN_SIDE.1..=MAX_SIDE.1).contains(&height) {
        return Err(format!("{s:?}: {width}x{height} is outside {}x{} to {}x{}", MIN_SIDE.0, MIN_SIDE.1, MAX_SIDE.0, MAX_SIDE.1));
    }
    let refresh_hz = hz.unwrap_or(DEFAULT_HZ);
    if !(24..=240).contains(&refresh_hz) {
        return Err(format!("{s:?}: a refresh rate of {refresh_hz} Hz is outside 24 to 240"));
    }
    Ok(Some(Mode { width, height, refresh_hz }))
}

/// The mode `XCL_DISPLAY` names (`None`: unset or not a mode – the reports are the default
/// ones). Read once; the first call logs what the title is told.
pub fn mode() -> Option<Mode> {
    static MODE: OnceLock<Option<Mode>> = OnceLock::new();
    *MODE.get_or_init(|| match parse(std::env::var("XCL_DISPLAY").ok().as_deref()) {
        Ok(Some(m)) => {
            crate::xlog_info!("{}", describe(m));
            Some(m)
        }
        Ok(None) => None,
        Err(e) => {
            crate::xlog_warn!("display: ignoring XCL_DISPLAY={e} – the title gets the default display reports");
            None
        }
    })
}

/// The log line for an override.
pub fn describe(m: Mode) -> String {
    format!(
        "display: XCL_DISPLAY – the title is told the display is {w}x{h} at {hz} Hz: its window (client area) is {w}x{h}, \
         its device's adapter has one output with a {w}x{h} desktop and the one display mode {w}x{h} @ {hz} Hz",
        w = m.width,
        h = m.height,
        hz = m.refresh_hz
    )
}

/// The size of the title's window: the mode's, else the console's 1920x1080.
pub fn window_size_for(m: Option<Mode>) -> (u32, u32) {
    m.map_or(DEFAULT_WINDOW, |m| (m.width, m.height))
}

/// [`window_size_for`] the `XCL_DISPLAY` mode.
pub fn window_size() -> (i32, i32) {
    let (w, h) = window_size_for(mode());
    (w as i32, h as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(width: u32, height: u32, refresh_hz: u32) -> Mode {
        Mode { width, height, refresh_hz }
    }

    #[test]
    fn unset_or_empty_is_no_override() {
        assert_eq!(parse(None), Ok(None));
        assert_eq!(parse(Some("")), Ok(None));
        assert_eq!(parse(Some("   ")), Ok(None));
    }

    #[test]
    fn sizes_and_rates() {
        assert_eq!(parse(Some("1920x1080")), Ok(Some(m(1920, 1080, 60))));
        assert_eq!(parse(Some(" 2560X1440 ")), Ok(Some(m(2560, 1440, 60))));
        assert_eq!(parse(Some("1280*720")), Ok(Some(m(1280, 720, 60))));
        assert_eq!(parse(Some("3840x2160@120")), Ok(Some(m(3840, 2160, 120))));
        assert_eq!(parse(Some("1920x1080@ 50Hz")), Ok(Some(m(1920, 1080, 50))));
        assert_eq!(parse(Some("1920 x 1080")), Ok(Some(m(1920, 1080, 60))));
    }

    #[test]
    fn values_that_name_no_mode_are_errors() {
        for bad in ["1920", "1920x", "x1080", "axb", "1920x1080@", "1920x1080@fast", "0x0", "100x100", "16000x9000", "1920x1080@500", "1920x1080@0", "-1920x1080"] {
            assert!(parse(Some(bad)).is_err(), "{bad:?} parsed");
        }
    }

    #[test]
    fn window_is_the_mode_or_the_consoles() {
        assert_eq!(window_size_for(None), (1920, 1080));
        assert_eq!(window_size_for(Some(m(2560, 1440, 60))), (2560, 1440));
        assert_eq!(window_size_for(Some(m(1280, 720, 60))), (1280, 720));
    }

    #[test]
    fn the_log_line_names_the_mode() {
        let s = describe(m(1920, 1080, 60));
        assert!(s.starts_with("display: XCL_DISPLAY"), "{s}");
        assert!(s.contains("1920x1080 at 60 Hz") && s.contains("1920x1080 @ 60 Hz"), "{s}");
    }
}
