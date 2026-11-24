//! Command lists of console code.
//!
//! The D3D12.X code compiled into a title (the SDK's inline functions) writes command records
//! straight into the list object when the list's buffer has room – +0x18 write pointer, +0x30
//! end, +0x28 the limit for dispatch records – and otherwise calls the list method the record
//! stands for, with the method's own arguments (HANDOFF: "inline writers"; e.g. 0xC000B600 +
//! a hardware topology, else #20 IASetPrimitiveTopology). A PC list object has other data at
//! those offsets. So console code gets a list object of the console's shape whose buffer never
//! has room: every inline write becomes the method call, and each method runs on the
//! (adopted) PC list behind it – with all of XCL's console slot handling.
//!
//! The object: the console fields, the PC list at `F_BACKING`; its vtable is a thunk per slot
//! (`mov rcx, [rcx + F_BACKING]` and on to the PC list's own vtable entry). QueryInterface
//! hands out the console object again, Release frees it with the PC list, and
//! ExecuteCommandLists runs the PC lists (`backing`).
//!
//! The PC list behind it can change while the title records: a wait recorded after other
//! commands closes the PC list there and recording goes on in another one
//! (`list_x::split_before_wait`). The PC list the object was made for keeps the title's
//! references (`F_IDENTITY`), and each forwarded call is counted (`F_CALLS`) so a wait that
//! comes first is told from one that follows other commands.

use std::ffi::c_void;
use std::sync::OnceLock;

use windows_sys::core::GUID;

use crate::util::HRESULT;

const LIST_SLOTS: usize = 600;
/// the object: console fields up to `ROOM`, no-room "buffer" at `ROOM` (with slack after it
/// for code that would write before checking), the PC list at `F_BACKING`
const OBJECT_BYTES: usize = 0x3000;
const ROOM: usize = 0x1000;
const F_BACKING: usize = 0x2FF8;
/// the PC list the object was made for: QueryInterface, AddRef and Release go to it
const F_IDENTITY: usize = 0x2FF0;
/// calls forwarded to the PC list (u64)
const F_CALLS: usize = 0x2FE8;
const F_WRITE: usize = 0x18;
const F_DISPATCH_END: usize = 0x28;
const F_END: usize = 0x30;
/// `XCL_LIST_TRACE`: the last `RING` slots called (u16 each) and a counter
const F_RING: usize = 0x2000;
const F_RING_AT: usize = 0x2100;
const RING: usize = 64;

struct Vtable(usize);

/// The console list vtable: QueryInterface / Release of their own, every other slot a thunk
/// to the PC list's entry.
fn vtable() -> usize {
    static V: OnceLock<Vtable> = OnceLock::new();
    V.get_or_init(|| {
        let mut v = vec![0usize; LIST_SLOTS];
        let trace = std::env::var_os("XCL_LIST_TRACE").is_some();
        for (slot, e) in v.iter_mut().enumerate() {
            *e = if trace { forward_traced(slot) } else { forward(slot) };
        }
        v[0] = qi as usize;
        v[1] = add_ref as usize;
        v[2] = release as usize;
        v[9] = close as usize;
        Vtable(Box::leak(v.into_boxed_slice()).as_ptr() as usize)
    })
    .0
}

/// `inc qword [rcx + F_CALLS]`
fn count_call() -> Vec<u8> {
    let mut c = vec![0x48, 0xFF, 0x81];
    c.extend_from_slice(&(F_CALLS as u32).to_le_bytes());
    c
}

/// `inc qword [rcx + F_CALLS]; mov rcx, [rcx + F_BACKING]; mov rax, [rcx]; jmp [rax + slot * 8]`
fn forward(slot: usize) -> usize {
    let mut c = count_call();
    c.extend_from_slice(&[0x48, 0x8B, 0x89]);
    c.extend_from_slice(&(F_BACKING as u32).to_le_bytes());
    c.extend_from_slice(&[0x48, 0x8B, 0x01, 0xFF, 0xA0]);
    c.extend_from_slice(&((slot * 8) as u32).to_le_bytes());
    let p = crate::stubs::alloc_code(c.len());
    unsafe {
        std::ptr::copy_nonoverlapping(c.as_ptr(), p, c.len());
    }
    p as usize
}

/// `forward` after `record(this, slot)` (argument registers, xmm0–3 kept)
fn forward_traced(slot: usize) -> usize {
    let mut c: Vec<u8> = count_call();
    c.extend_from_slice(&[0x51, 0x52, 0x41, 0x50, 0x41, 0x51]); // push rcx, rdx, r8, r9
    c.extend_from_slice(&[0x48, 0x83, 0xEC, 0x68]); // sub rsp, 0x68
    for (reg, off) in [(0x44u8, 0x20u8), (0x4C, 0x30), (0x54, 0x40), (0x5C, 0x50)] {
        c.extend_from_slice(&[0xF3, 0x0F, 0x7F, reg, 0x24, off]); // movdqu [rsp+off], xmmN
    }
    c.push(0xBA); // mov edx, slot
    c.extend_from_slice(&(slot as u32).to_le_bytes());
    c.extend_from_slice(&[0x48, 0xB8]); // mov rax, record
    c.extend_from_slice(&(record as usize as u64).to_le_bytes());
    c.extend_from_slice(&[0xFF, 0xD0]); // call rax
    for (reg, off) in [(0x44u8, 0x20u8), (0x4C, 0x30), (0x54, 0x40), (0x5C, 0x50)] {
        c.extend_from_slice(&[0xF3, 0x0F, 0x6F, reg, 0x24, off]); // movdqu xmmN, [rsp+off]
    }
    c.extend_from_slice(&[0x48, 0x83, 0xC4, 0x68]); // add rsp, 0x68
    c.extend_from_slice(&[0x41, 0x59, 0x41, 0x58, 0x5A, 0x59]); // pop r9, r8, rdx, rcx
    c.extend_from_slice(&[0x48, 0x8B, 0x89]);
    c.extend_from_slice(&(F_BACKING as u32).to_le_bytes());
    c.extend_from_slice(&[0x48, 0x8B, 0x01, 0xFF, 0xA0]);
    c.extend_from_slice(&((slot * 8) as u32).to_le_bytes());
    let p = crate::stubs::alloc_code(c.len());
    unsafe {
        std::ptr::copy_nonoverlapping(c.as_ptr(), p, c.len());
    }
    p as usize
}

unsafe extern "system" fn record(this: *mut c_void, slot: u32) {
    let at = (this as usize + F_RING_AT) as *mut usize;
    let k = *at % RING;
    *((this as usize + F_RING + 2 * k) as *mut u16) = slot as u16;
    *at += 1;
}

/// #9 Close: a failure is logged with the last methods called (`XCL_LIST_TRACE`)
unsafe extern "system" fn close(this: *mut c_void) -> HRESULT {
    let pc = *((this as usize + F_BACKING) as *const *mut c_void);
    let f: unsafe extern "system" fn(*mut c_void) -> HRESULT = std::mem::transmute(pc_slot(pc, 9));
    let hr = f(pc);
    let n = *((this as usize + F_RING_AT) as *const usize);
    if hr < 0 && n > 0 {
        let first = n.saturating_sub(RING);
        let slots: Vec<String> = (first..n).map(|k| (*((this as usize + F_RING + 2 * (k % RING)) as *const u16)).to_string()).collect();
        crate::xlog_warn!("D3D12.X: console list Close → {:#x}; console slots called since Reset (last {}): {}", hr as u32, slots.len(), slots.join(" "));
    }
    // a new recording starts at Reset
    *((this as usize + F_RING_AT) as *mut usize) = 0;
    hr
}

/// The console object for the PC list `pc` (adopted; one reference moves to it).
pub unsafe fn wrap(pc: *mut c_void) -> *mut c_void {
    if pc.is_null() {
        return pc;
    }
    let layout = std::alloc::Layout::from_size_align(OBJECT_BYTES, 64).unwrap();
    let o = std::alloc::alloc_zeroed(layout);
    if o.is_null() {
        return pc;
    }
    *(o as *mut usize) = vtable();
    let room = o as usize + ROOM;
    for f in [F_WRITE, F_DISPATCH_END, F_END] {
        *(o.add(f) as *mut usize) = room;
    }
    *(o.add(F_BACKING) as *mut usize) = pc as usize;
    *(o.add(F_IDENTITY) as *mut usize) = pc as usize;
    super::list_x::note_wrapper(pc, o as *mut c_void);
    o as *mut c_void
}

/// Recording on console list object `list` goes on in PC list `pc`.
pub(super) unsafe fn set_backing(list: *mut c_void, pc: *mut c_void) {
    *((list as usize + F_BACKING) as *mut *mut c_void) = pc;
}

/// Calls forwarded to the PC lists behind console list object `list` so far.
pub(super) unsafe fn calls(list: *mut c_void) -> u64 {
    *((list as usize + F_CALLS) as *const u64)
}

/// What a forwarding thunk counts, for a test that calls the PC list's hook itself (calls from
/// outside the title go to the PC methods: `guard`).
#[cfg(test)]
pub(super) unsafe fn count_call_for_test(list: *mut c_void) {
    *((list as usize + F_CALLS) as *mut u64) += 1;
}

unsafe fn identity(this: *mut c_void) -> *mut c_void {
    *((this as usize + F_IDENTITY) as *const *mut c_void)
}

/// The PC list behind a console list object (None: not one).
pub fn backing(list: *mut c_void) -> Option<*mut c_void> {
    if list.is_null() || !crate::gpu::shader::readable(list as usize, 8) {
        return None;
    }
    unsafe { (*(list as *const usize) == vtable()).then(|| *((list as usize + F_BACKING) as *const *mut c_void)) }
}

unsafe fn pc_slot(pc: *mut c_void, slot: usize) -> usize {
    *(*(pc as *const *const usize)).add(slot)
}

/// QueryInterface: the PC list answers; where it hands out itself, the console object goes out.
/// Command list interfaces of the console's headers that the PC does not know: the console's
/// ID3D12GraphicsCommandList5 / 6 (the PC's data1–data3, another tail – titles ask for them
/// and use the answer without a check). The console object has the console's whole vtable,
/// so it is the answer.
const CONSOLE_LIST_IIDS: [(u32, u16, u16, [u8; 8]); 2] = [
    (0x5505_0859, 0x4024, 0x474c, [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74]),
    (0xc382_7890, 0x4024, 0x474c, [0x98, 0xcf, 0x64, 0x5b, 0x51, 0xdc, 0x48, 0x74]),
];

unsafe extern "system" fn qi(this: *mut c_void, riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
    let pc = identity(this);
    let f: unsafe extern "system" fn(*mut c_void, *const GUID, *mut *mut c_void) -> HRESULT = std::mem::transmute(pc_slot(pc, 0));
    if !riid.is_null() && !ppv.is_null() {
        let g = &*riid;
        if CONSOLE_LIST_IIDS.contains(&(g.data1, g.data2, g.data3, g.data4)) {
            let add_ref: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(pc_slot(pc, 1));
            add_ref(pc);
            *ppv = this;
            return 0;
        }
    }
    let hr = f(pc, riid, ppv);
    if hr >= 0 && !ppv.is_null() && *ppv == pc {
        *ppv = this;
    }
    hr
}

/// AddRef: on the PC list the object was made for (the one Release counts on).
unsafe extern "system" fn add_ref(this: *mut c_void) -> u32 {
    let pc = identity(this);
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(pc_slot(pc, 1));
    f(pc)
}

/// Release: the console object goes with the PC list's last reference (and the PC lists XCL
/// made for its recordings with it).
unsafe extern "system" fn release(this: *mut c_void) -> u32 {
    let pc = identity(this);
    let f: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(pc_slot(pc, 2));
    let n = f(pc);
    if n == 0 {
        super::bind::forget(pc);
        super::list_x::forget_split(this);
        std::alloc::dealloc(this as *mut u8, std::alloc::Layout::from_size_align(OBJECT_BYTES, 64).unwrap());
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stand-in PC object: slot 5 returns `this`, slots 0 / 2 count.
    #[repr(C)]
    struct Fake {
        vt: *const usize,
        refs: u32,
    }

    unsafe extern "system" fn fake_qi(this: *mut c_void, _riid: *const GUID, ppv: *mut *mut c_void) -> HRESULT {
        (*(this as *mut Fake)).refs += 1;
        *ppv = this;
        0
    }
    unsafe extern "system" fn fake_add_ref(this: *mut c_void) -> u32 {
        (*(this as *mut Fake)).refs += 1;
        (*(this as *mut Fake)).refs
    }
    unsafe extern "system" fn fake_release(this: *mut c_void) -> u32 {
        (*(this as *mut Fake)).refs -= 1;
        (*(this as *mut Fake)).refs
    }
    unsafe extern "system" fn fake_this(this: *mut c_void, a: u64, b: u64, c: u64, d: u64) -> u64 {
        this as u64 + a + 10 * b + 100 * c + 1000 * d
    }

    /// The SDK's inline writers check `write < end` (or `write + n <= end`): never room here.
    #[test]
    fn console_list_objects_have_no_room_and_forward_to_the_pc_list() {
        let mut vt = vec![0usize; LIST_SLOTS];
        vt[0] = fake_qi as usize;
        vt[1] = fake_add_ref as usize;
        vt[2] = fake_release as usize;
        vt[20] = fake_this as usize;
        let fake = Box::into_raw(Box::new(Fake { vt: vt.as_ptr(), refs: 1 }));
        unsafe {
            let o = wrap(fake as *mut c_void);
            assert_eq!(backing(o), Some(fake as *mut c_void));
            assert_eq!(backing(fake as *mut c_void), None);
            let (w, e) = (*((o as usize + F_WRITE) as *const usize), *((o as usize + F_END) as *const usize));
            assert!(w >= e && *((o as usize + F_DISPATCH_END) as *const usize) == w);
            let slot = |i: usize| *(*(o as *const *const usize)).add(i);
            // a method: the PC list's, with the PC list as `this` and the arguments kept
            let m: unsafe extern "system" fn(*mut c_void, u64, u64, u64, u64) -> u64 = std::mem::transmute(slot(20));
            assert_eq!(m(o, 1, 2, 3, 4), fake as u64 + 4321);
            // QueryInterface hands out the console object – also for the console-only list IIDs
            let q: unsafe extern "system" fn(*mut c_void, *const GUID, *mut *mut c_void) -> HRESULT = std::mem::transmute(slot(0));
            let mut out = std::ptr::null_mut();
            assert_eq!(q(o, std::ptr::null(), &mut out), 0);
            assert_eq!(out, o);
            let (d1, d2, d3, d4) = CONSOLE_LIST_IIDS[0];
            let gcl5 = GUID { data1: d1, data2: d2, data3: d3, data4: d4 };
            let mut out = std::ptr::null_mut();
            assert_eq!(q(o, &gcl5, &mut out), 0);
            assert_eq!(out, o);
            let r: unsafe extern "system" fn(*mut c_void) -> u32 = std::mem::transmute(slot(2));
            assert_eq!(r(o), 2);
            assert_eq!(r(o), 1);
            assert_eq!(r(o), 0); // the object is freed with it
            drop(Box::from_raw(fake));
        }
    }
}
