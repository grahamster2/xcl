//! What XCL handed the driver for the pipelines it made, by the pipeline: the kind and the bytes
//! of DXIL (`XCL_MEMORY_MAP`). A retail title's driver memory (NVIDIA's write-combined private
//! allocations, ~50 MB each) grows with the pipelines alive; the memory map sets the one beside
//! the other in each sample.
use std::collections::HashMap;
use std::sync::Mutex;

/// The kind of a pipeline XCL made from translated console shaders.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Kind {
    /// a compute pipeline of one translated program
    Compute,
    /// a piece of a split compute program (`dispatch_chain`)
    Piece,
    /// the whole form of a split compute program, built when one is used
    Whole,
    /// a graphics pipeline of translated shaders
    Graphics,
    /// a console mesh or geometry pipeline in two passes (compute pass + mesh pipeline)
    TwoPass,
}

pub(crate) const KINDS: [Kind; 5] = [Kind::Compute, Kind::Piece, Kind::Whole, Kind::Graphics, Kind::TwoPass];

impl Kind {
    pub(crate) fn name(self) -> &'static str {
        match self {
            Kind::Compute => "compute",
            Kind::Piece => "split pieces",
            Kind::Whole => "split whole forms",
            Kind::Graphics => "graphics",
            Kind::TwoPass => "two-pass",
        }
    }
}

/// pipeline → (kind, DXIL bytes)
static LIVE: Mutex<Option<HashMap<usize, (Kind, usize)>>> = Mutex::new(None);

/// `pso` was made with `bytes` of DXIL.
pub(crate) fn note(pso: usize, kind: Kind, bytes: usize) {
    if pso != 0 {
        LIVE.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, (kind, bytes));
    }
}

/// `pso` is gone (`forget_pso`).
pub(crate) fn forget(pso: usize) {
    if let Some(m) = LIVE.lock().unwrap().as_mut() {
        m.remove(&pso);
    }
}

/// (pipelines, DXIL bytes) alive of each of `KINDS`.
pub(crate) fn counts() -> [(usize, u64); 5] {
    let g = LIVE.lock().unwrap();
    tally(g.as_ref().into_iter().flat_map(|m| m.values().copied()))
}

fn tally(entries: impl IntoIterator<Item = (Kind, usize)>) -> [(usize, u64); 5] {
    let mut out = [(0usize, 0u64); 5];
    for (kind, bytes) in entries {
        let i = KINDS.iter().position(|k| *k == kind).unwrap();
        out[i].0 += 1;
        out[i].1 += bytes as u64;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipelines_are_counted_by_kind_with_their_dxil() {
        let t = tally([(Kind::Compute, 100), (Kind::Graphics, 7), (Kind::Compute, 50), (Kind::Whole, 1)]);
        assert_eq!(t, [(2, 150), (0, 0), (1, 1), (1, 7), (0, 0)]);
    }

    #[test]
    fn a_forgotten_pipeline_is_no_longer_counted() {
        let (a, b) = (0x7E57_D0A1usize, 0x7E57_D0A2usize);
        let has = |p: usize| LIVE.lock().unwrap().as_ref().and_then(|m| m.get(&p).copied());
        note(a, Kind::TwoPass, 1000);
        note(b, Kind::TwoPass, 24);
        note(0, Kind::TwoPass, 5);
        forget(a);
        assert_eq!((has(a), has(b), has(0)), (None, Some((Kind::TwoPass, 24)), None));
        forget(b);
        forget(b);
        assert_eq!(has(b), None);
    }
}
