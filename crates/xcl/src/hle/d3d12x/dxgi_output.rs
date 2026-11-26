//! The display output a title reaches through its device under `XCL_DISPLAY` (`hle::display`).
//!
//! `IDXGIDevice::GetAdapter` gives the PC adapter (`dxgi_device`); its outputs are the PC's
//! monitors. With an `XCL_DISPLAY` mode the title gets a small adapter object instead: as the
//! PC adapter in everything but `EnumOutputs`, which has one output (as a console has), the
//! console's TV in that mode: `GetDesc` a desktop of the mode's size, `GetDisplayModeList` /
//! `FindClosestMatchingMode` the one mode. `WaitForVBlank` and the rest are the PC output's
//! (the PC adapter's first output; without one, `WaitForVBlank` waits a refresh period).
//! The newer output interfaces (IDXGIOutput1–6) are not answered: they would give the PC
//! monitor's modes. The newer adapter interfaces are the PC adapter's (memory budgets).

use std::ffi::c_void;
use std::sync::atomic::{AtomicU32, Ordering};

use windows::core::{Interface, GUID};
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Dxgi::Common::*;
use windows::Win32::Graphics::Dxgi::*;

use crate::hle::display::Mode;
use crate::util::{HRESULT, S_OK};

const E_NOINTERFACE: HRESULT = 0x8000_4002u32 as i32;
const E_POINTER: HRESULT = 0x8000_4003u32 as i32;
const E_INVALIDARG: HRESULT = 0x8007_0057u32 as i32;
const DXGI_ERROR_NOT_FOUND: HRESULT = 0x887A_0002u32 as i32;
const DXGI_ERROR_MORE_DATA: HRESULT = 0x887A_0003u32 as i32;

/// A COM object's vtable slot `n` (the object is a live interface pointer).
unsafe fn slot(obj: *mut c_void, n: usize) -> usize {
    *(*(obj as *const *const usize)).add(n)
}

unsafe fn com_release(obj: *mut c_void) {
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(slot(obj, 2));
    f(obj);
}

unsafe fn com_qi(obj: *mut c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    let f: unsafe extern "system" fn(*mut c_void, *const GUID, *mut *mut c_void) -> HRESULT = std::mem::transmute(slot(obj, 0));
    f(obj, riid, out)
}

/// The display formats a TV mode is listed in (the swap formats of `present`).
fn scanout(f: DXGI_FORMAT) -> bool {
    [
        DXGI_FORMAT_R8G8B8A8_UNORM,
        DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        DXGI_FORMAT_B8G8R8A8_UNORM,
        DXGI_FORMAT_B8G8R8A8_UNORM_SRGB,
        DXGI_FORMAT_R10G10B10A2_UNORM,
        DXGI_FORMAT_R16G16B16A16_FLOAT,
    ]
    .contains(&f)
}

/// The mode as DXGI describes it, in `format`.
pub fn mode_desc(m: Mode, format: DXGI_FORMAT) -> DXGI_MODE_DESC {
    DXGI_MODE_DESC {
        Width: m.width,
        Height: m.height,
        RefreshRate: DXGI_RATIONAL { Numerator: m.refresh_hz, Denominator: 1 },
        Format: format,
        ScanlineOrdering: DXGI_MODE_SCANLINE_ORDER_PROGRESSIVE,
        Scaling: DXGI_MODE_SCALING_UNSPECIFIED,
    }
}

/// The output's description: the PC output's name and monitor (if any), a desktop of the mode.
pub fn output_desc(pc: Option<DXGI_OUTPUT_DESC>, m: Mode) -> DXGI_OUTPUT_DESC {
    let mut d = pc.unwrap_or_default();
    if pc.is_none() {
        let name: Vec<u16> = r"\\.\DISPLAY1".encode_utf16().collect();
        d.DeviceName[..name.len()].copy_from_slice(&name);
    }
    d.DesktopCoordinates = RECT { left: 0, top: 0, right: m.width as i32, bottom: m.height as i32 };
    d.AttachedToDesktop = true.into();
    d.Rotation = DXGI_MODE_ROTATION_IDENTITY;
    d
}

/// `GetDisplayModeList`: the one mode in a display format, none in the others; `modes` null
/// asks for the count.
pub fn mode_list(m: Mode, format: DXGI_FORMAT, count: &mut u32, modes: Option<&mut [DXGI_MODE_DESC]>) -> HRESULT {
    let n = u32::from(scanout(format));
    let Some(modes) = modes else {
        *count = n;
        return S_OK;
    };
    let fit = (modes.len() as u32).min(n);
    if fit > 0 {
        modes[0] = mode_desc(m, format);
    }
    *count = fit;
    if fit < n {
        DXGI_ERROR_MORE_DATA
    } else {
        S_OK
    }
}

// --- the adapter -----------------------------------------------------------------------------

#[repr(C)]
struct Adapter {
    vtbl: *const [usize; 11],
    refs: AtomicU32,
    /// the PC adapter (IDXGIAdapter1, one reference held)
    pc: *mut c_void,
    mode: Mode,
}

fn adapter_vtbl() -> &'static [usize; 11] {
    static V: std::sync::OnceLock<[usize; 11]> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        [
            adapter_qi as usize,
            adapter_add_ref as usize,
            adapter_release as usize,
            set_private_data::<Adapter> as usize,
            set_private_data_interface::<Adapter> as usize,
            get_private_data::<Adapter> as usize,
            adapter_get_parent as usize,
            adapter_enum_outputs as usize,
            adapter_forward2::<8> as usize,  // GetDesc
            adapter_forward3::<9> as usize,  // CheckInterfaceSupport
            adapter_forward2::<10> as usize, // GetDesc1
        ]
    })
}

/// The title's adapter for the PC adapter `pc` (borrowed; the object takes its own reference),
/// or `pc` itself (one more reference) if it is no IDXGIAdapter1. One reference, the caller's.
pub unsafe fn wrap_adapter(pc: &IDXGIAdapter, mode: Mode) -> *mut c_void {
    let Ok(a1) = pc.cast::<IDXGIAdapter1>() else { return pc.clone().into_raw() };
    let o = Box::new(Adapter { vtbl: adapter_vtbl(), refs: AtomicU32::new(1), pc: a1.into_raw(), mode });
    Box::into_raw(o) as *mut c_void
}

/// the PC object behind one of these proxies (`T` is `Adapter` or `Output`)
trait Proxy {
    unsafe fn pc(p: *mut c_void) -> *mut c_void;
}

impl Proxy for Adapter {
    unsafe fn pc(p: *mut c_void) -> *mut c_void {
        (*(p as *const Adapter)).pc
    }
}

unsafe extern "system" fn adapter_qi(p: *mut c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if riid.is_null() || out.is_null() {
        return E_POINTER;
    }
    if [windows::core::IUnknown::IID, IDXGIObject::IID, IDXGIAdapter::IID, IDXGIAdapter1::IID].contains(&*riid) {
        adapter_add_ref(p);
        *out = p;
        return S_OK;
    }
    com_qi(Adapter::pc(p), riid, out)
}

unsafe extern "system" fn adapter_add_ref(p: *mut c_void) -> u32 {
    (*(p as *const Adapter)).refs.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn adapter_release(p: *mut c_void) -> u32 {
    let n = (*(p as *const Adapter)).refs.fetch_sub(1, Ordering::AcqRel) - 1;
    if n == 0 {
        let o = Box::from_raw(p as *mut Adapter);
        com_release(o.pc);
    }
    n
}

unsafe extern "system" fn adapter_get_parent(p: *mut c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    let f: unsafe extern "system" fn(*mut c_void, *const GUID, *mut *mut c_void) -> HRESULT = std::mem::transmute(slot(Adapter::pc(p), 6));
    f(Adapter::pc(p), riid, out)
}

unsafe extern "system" fn adapter_forward2<const N: usize>(p: *mut c_void, a: *mut c_void) -> HRESULT {
    let f: unsafe extern "system" fn(*mut c_void, *mut c_void) -> HRESULT = std::mem::transmute(slot(Adapter::pc(p), N));
    f(Adapter::pc(p), a)
}

unsafe extern "system" fn adapter_forward3<const N: usize>(p: *mut c_void, a: *mut c_void, b: *mut c_void) -> HRESULT {
    let f: unsafe extern "system" fn(*mut c_void, *mut c_void, *mut c_void) -> HRESULT = std::mem::transmute(slot(Adapter::pc(p), N));
    f(Adapter::pc(p), a, b)
}

/// one output, the console's TV
unsafe extern "system" fn adapter_enum_outputs(p: *mut c_void, index: u32, out: *mut *mut c_void) -> HRESULT {
    if out.is_null() {
        return E_POINTER;
    }
    *out = std::ptr::null_mut();
    if index != 0 {
        return DXGI_ERROR_NOT_FOUND;
    }
    let pc_raw = Adapter::pc(p);
    let pc = IDXGIAdapter1::from_raw_borrowed(&pc_raw).and_then(|a| a.EnumOutputs(0).ok());
    adapter_add_ref(p);
    let o = Box::new(Output {
        vtbl: output_vtbl(),
        refs: AtomicU32::new(1),
        pc: pc.map_or(std::ptr::null_mut(), |o| o.into_raw()),
        adapter: p,
        mode: (*(p as *const Adapter)).mode,
    });
    *out = Box::into_raw(o) as *mut c_void;
    S_OK
}

// --- private data, for both --------------------------------------------------------------------

unsafe extern "system" fn set_private_data<T: Proxy>(p: *mut c_void, name: *const GUID, size: u32, data: *const c_void) -> HRESULT {
    let pc = T::pc(p);
    if pc.is_null() {
        return E_INVALIDARG;
    }
    let f: unsafe extern "system" fn(*mut c_void, *const GUID, u32, *const c_void) -> HRESULT = std::mem::transmute(slot(pc, 3));
    f(pc, name, size, data)
}

unsafe extern "system" fn set_private_data_interface<T: Proxy>(p: *mut c_void, name: *const GUID, unknown: *mut c_void) -> HRESULT {
    let pc = T::pc(p);
    if pc.is_null() {
        return E_INVALIDARG;
    }
    let f: unsafe extern "system" fn(*mut c_void, *const GUID, *mut c_void) -> HRESULT = std::mem::transmute(slot(pc, 4));
    f(pc, name, unknown)
}

unsafe extern "system" fn get_private_data<T: Proxy>(p: *mut c_void, name: *const GUID, size: *mut u32, data: *mut c_void) -> HRESULT {
    let pc = T::pc(p);
    if pc.is_null() {
        return DXGI_ERROR_NOT_FOUND;
    }
    let f: unsafe extern "system" fn(*mut c_void, *const GUID, *mut u32, *mut c_void) -> HRESULT = std::mem::transmute(slot(pc, 5));
    f(pc, name, size, data)
}

// --- the output ------------------------------------------------------------------------------

#[repr(C)]
struct Output {
    vtbl: *const [usize; 19],
    refs: AtomicU32,
    /// the PC adapter's first output (one reference held), or null
    pc: *mut c_void,
    /// the adapter object (one reference held)
    adapter: *mut c_void,
    mode: Mode,
}

impl Proxy for Output {
    unsafe fn pc(p: *mut c_void) -> *mut c_void {
        (*(p as *const Output)).pc
    }
}

fn output_vtbl() -> &'static [usize; 19] {
    static V: std::sync::OnceLock<[usize; 19]> = std::sync::OnceLock::new();
    V.get_or_init(|| {
        [
            output_qi as usize,
            output_add_ref as usize,
            output_release as usize,
            set_private_data::<Output> as usize,
            set_private_data_interface::<Output> as usize,
            get_private_data::<Output> as usize,
            output_get_parent as usize,
            output_get_desc as usize,
            output_get_display_mode_list as usize,
            output_find_closest_matching_mode as usize,
            output_wait_for_vblank as usize,
            output_forward3::<11> as usize, // TakeOwnership(device, exclusive)
            output_release_ownership as usize,
            output_forward2::<13> as usize, // GetGammaControlCapabilities
            output_forward2::<14> as usize, // SetGammaControl
            output_forward2::<15> as usize, // GetGammaControl
            output_forward2::<16> as usize, // SetDisplaySurface
            output_forward2::<17> as usize, // GetDisplaySurfaceData
            output_forward2::<18> as usize, // GetFrameStatistics
        ]
    })
}

unsafe fn output<'a>(p: *mut c_void) -> &'a Output {
    &*(p as *const Output)
}

unsafe extern "system" fn output_qi(p: *mut c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    if riid.is_null() || out.is_null() {
        return E_POINTER;
    }
    if [windows::core::IUnknown::IID, IDXGIObject::IID, IDXGIOutput::IID].contains(&*riid) {
        output_add_ref(p);
        *out = p;
        return S_OK;
    }
    *out = std::ptr::null_mut();
    E_NOINTERFACE
}

unsafe extern "system" fn output_add_ref(p: *mut c_void) -> u32 {
    output(p).refs.fetch_add(1, Ordering::Relaxed) + 1
}

unsafe extern "system" fn output_release(p: *mut c_void) -> u32 {
    let n = output(p).refs.fetch_sub(1, Ordering::AcqRel) - 1;
    if n == 0 {
        let o = Box::from_raw(p as *mut Output);
        if !o.pc.is_null() {
            com_release(o.pc);
        }
        adapter_release(o.adapter);
    }
    n
}

/// the parent of the output is the adapter the title enumerated it from
unsafe extern "system" fn output_get_parent(p: *mut c_void, riid: *const GUID, out: *mut *mut c_void) -> HRESULT {
    adapter_qi(output(p).adapter, riid, out)
}

unsafe extern "system" fn output_get_desc(p: *mut c_void, desc: *mut DXGI_OUTPUT_DESC) -> HRESULT {
    if desc.is_null() {
        return E_INVALIDARG;
    }
    let o = output(p);
    let pc = IDXGIOutput::from_raw_borrowed(&o.pc).and_then(|x| x.GetDesc().ok());
    *desc = output_desc(pc, o.mode);
    S_OK
}

unsafe extern "system" fn output_get_display_mode_list(p: *mut c_void, format: DXGI_FORMAT, _flags: u32, count: *mut u32, modes: *mut DXGI_MODE_DESC) -> HRESULT {
    if count.is_null() {
        return E_INVALIDARG;
    }
    let list = (!modes.is_null()).then(|| std::slice::from_raw_parts_mut(modes, *count as usize));
    mode_list(output(p).mode, format, &mut *count, list)
}

unsafe extern "system" fn output_find_closest_matching_mode(p: *mut c_void, want: *const DXGI_MODE_DESC, closest: *mut DXGI_MODE_DESC, _device: *mut c_void) -> HRESULT {
    if want.is_null() || closest.is_null() {
        return E_INVALIDARG;
    }
    let f = (*want).Format;
    *closest = mode_desc(output(p).mode, if scanout(f) { f } else { DXGI_FORMAT_R8G8B8A8_UNORM });
    S_OK
}

unsafe extern "system" fn output_wait_for_vblank(p: *mut c_void) -> HRESULT {
    let o = output(p);
    if o.pc.is_null() {
        std::thread::sleep(std::time::Duration::from_micros(1_000_000 / o.mode.refresh_hz.max(1) as u64));
        return S_OK;
    }
    let f: unsafe extern "system" fn(*mut c_void) -> HRESULT = std::mem::transmute(slot(o.pc, 10));
    f(o.pc)
}

unsafe extern "system" fn output_release_ownership(p: *mut c_void) {
    let pc = Output::pc(p);
    if !pc.is_null() {
        let f: unsafe extern "system" fn(*mut c_void) = std::mem::transmute(slot(pc, 12));
        f(pc)
    }
}

const DXGI_ERROR_UNSUPPORTED: HRESULT = 0x887A_0004u32 as i32;

unsafe extern "system" fn output_forward2<const N: usize>(p: *mut c_void, a: *mut c_void) -> HRESULT {
    let pc = Output::pc(p);
    if pc.is_null() {
        return DXGI_ERROR_UNSUPPORTED;
    }
    let f: unsafe extern "system" fn(*mut c_void, *mut c_void) -> HRESULT = std::mem::transmute(slot(pc, N));
    f(pc, a)
}

unsafe extern "system" fn output_forward3<const N: usize>(p: *mut c_void, a: *mut c_void, b: usize) -> HRESULT {
    let pc = Output::pc(p);
    if pc.is_null() {
        return DXGI_ERROR_UNSUPPORTED;
    }
    let f: unsafe extern "system" fn(*mut c_void, *mut c_void, usize) -> HRESULT = std::mem::transmute(slot(pc, N));
    f(pc, a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Mode = Mode { width: 1920, height: 1080, refresh_hz: 60 };

    #[test]
    fn the_one_mode_in_display_formats() {
        let mut n = 99;
        assert_eq!(mode_list(M, DXGI_FORMAT_R8G8B8A8_UNORM, &mut n, None), S_OK);
        assert_eq!(n, 1);
        assert_eq!(mode_list(M, DXGI_FORMAT_R32_FLOAT, &mut n, None), S_OK);
        assert_eq!(n, 0);
        let mut modes = [DXGI_MODE_DESC::default(); 4];
        let mut n = 4;
        assert_eq!(mode_list(M, DXGI_FORMAT_R10G10B10A2_UNORM, &mut n, Some(&mut modes)), S_OK);
        assert_eq!(n, 1);
        let d = modes[0];
        assert_eq!((d.Width, d.Height, d.RefreshRate.Numerator, d.RefreshRate.Denominator, d.Format), (1920, 1080, 60, 1, DXGI_FORMAT_R10G10B10A2_UNORM));
        assert_eq!(d.ScanlineOrdering, DXGI_MODE_SCANLINE_ORDER_PROGRESSIVE);
        let mut n = 0;
        assert_eq!(mode_list(M, DXGI_FORMAT_B8G8R8A8_UNORM, &mut n, Some(&mut modes[..0])), DXGI_ERROR_MORE_DATA);
        assert_eq!(n, 0);
    }

    #[test]
    fn the_desktop_is_the_mode() {
        let d = output_desc(None, Mode { width: 2560, height: 1440, refresh_hz: 60 });
        assert_eq!((d.DesktopCoordinates.right, d.DesktopCoordinates.bottom, d.DesktopCoordinates.left, d.DesktopCoordinates.top), (2560, 1440, 0, 0));
        assert!(d.AttachedToDesktop.as_bool());
        assert!(String::from_utf16_lossy(&d.DeviceName).starts_with(r"\\.\DISPLAY1"));
        let mut pc = DXGI_OUTPUT_DESC::default();
        pc.DesktopCoordinates = RECT { left: -2560, top: 0, right: 0, bottom: 1440 };
        pc.Rotation = DXGI_MODE_ROTATION_ROTATE90;
        pc.DeviceName[0] = 'Q' as u16;
        let d = output_desc(Some(pc), M);
        assert_eq!((d.DesktopCoordinates.left, d.DesktopCoordinates.right, d.DesktopCoordinates.bottom), (0, 1920, 1080));
        assert_eq!(d.Rotation, DXGI_MODE_ROTATION_IDENTITY);
        assert_eq!(d.DeviceName[0], 'Q' as u16, "the PC output's name is kept");
    }

    /// The objects through the DXGI interfaces, over the WARP adapter (no outputs of its own)
    /// and the first hardware adapter (if it has a monitor).
    #[test]
    fn the_adapter_has_one_output_in_the_mode() {
        unsafe {
            let factory: IDXGIFactory4 = CreateDXGIFactory1().expect("DXGI factory");
            let mut adapters: Vec<IDXGIAdapter> = vec![factory.EnumWarpAdapter().expect("WARP adapter")];
            if let Ok(a) = factory.EnumAdapters(0) {
                adapters.push(a);
            }
            for pc in adapters {
                let raw = wrap_adapter(&pc, M);
                let a = IDXGIAdapter::from_raw(raw);
                assert_eq!(a.GetDesc().unwrap().AdapterLuid.LowPart, pc.GetDesc().unwrap().AdapterLuid.LowPart);
                assert!(a.cast::<IDXGIAdapter1>().unwrap().GetDesc1().is_ok());
                assert!(a.cast::<IDXGIAdapter3>().is_ok(), "the newer adapter interfaces are the PC adapter's");
                let o = a.EnumOutputs(0).expect("one output");
                assert!(a.EnumOutputs(1).is_err());
                let d = o.GetDesc().unwrap();
                assert_eq!((d.DesktopCoordinates.right, d.DesktopCoordinates.bottom), (1920, 1080));
                let mut n = 0;
                o.GetDisplayModeList(DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_ENUM_MODES(0), &mut n, None).unwrap();
                assert_eq!(n, 1);
                let mut modes = [DXGI_MODE_DESC::default(); 2];
                let mut n = 2;
                o.GetDisplayModeList(DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_ENUM_MODES(0), &mut n, Some(modes.as_mut_ptr())).unwrap();
                assert_eq!((n, modes[0].Width, modes[0].Height, modes[0].RefreshRate.Numerator), (1, 1920, 1080, 60));
                let want = DXGI_MODE_DESC { Width: 3840, Height: 2160, Format: DXGI_FORMAT_R8G8B8A8_UNORM, ..Default::default() };
                let mut got = DXGI_MODE_DESC::default();
                o.FindClosestMatchingMode(&want, &mut got, None).unwrap();
                assert_eq!((got.Width, got.Height), (1920, 1080));
                assert!(o.cast::<IDXGIOutput1>().is_err(), "the newer output interfaces would give the PC's modes");
                let parent: IDXGIAdapter = o.GetParent().unwrap();
                assert_eq!(parent.as_raw(), a.as_raw(), "the output's parent is the title's adapter");
                drop(parent);
                o.WaitForVBlank().unwrap();
            }
        }
    }
}
