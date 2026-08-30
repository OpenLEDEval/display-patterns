# display-patterns — Roadmap

Derived from [SPEC.md](SPEC.md). Sections are in build-dependency
order. New work enters at the tail; completed work is deleted.
Consumer adoption is tracked in the consumers' own roadmaps
(bmd-signal-gen `§road:consume-display-patterns`).

## Extraction spine §road:extraction-spine

Walking skeleton: an installable package whose patterns are
bmd-signal-gen's, moved verbatim so the bit-identical adoption claim
is checkable before any reshaping (§spec:extraction).

### First release §road:first-release

Publish 0.1 to PyPI via trusted publishing. §spec:package-shape.
Blocked — deferred until the first rounds of cross-repo integration
testing complete and the PyPI org is confirmed; consumers integrate via
a git dependency meanwhile. (The OpenLEDEval → OpenDisplayEval rename
that previously blocked this is done.)

**Verify:** In a fresh environment, `pip install display-patterns`
imports and renders a checkerboard with numpy alone; `pip install
"display-patterns[charts,io]"` renders a YAML-authored chart and
writes it to TIFF. (The array-for-array equivalence claim against
bmd-signal-gen is banked and pinned in §spec:extraction; its
scaffolding retired with the rendering-model reshape.)

## Fused backend rendering §road:fused-backends

Make the multi-backend claim true and checkable: render bodies a
consumer's compiler can fuse, an output dtype that works on the
backend it develops on, and the torch leg running on every change
instead of skipping (§spec:backend-portability). A render-runtime
consumer repoints its duplicated counter-panel math here once these
land; its adoption is tracked in its own roadmap.

### Torch as a gate, not a skip §road:torch-gate

Add torch's CPU build to the dev dependency group and register `cuda`
and `mps` markers, so the backend-equivalence legs in
`tests/conftest.py` run on every change and the device legs are opt-in
— `pyproject.toml`, `tests/conftest.py`, `.github/workflows/ci.yml`.
§spec:backend-portability.

### Functional fills at a caller-stated dtype §road:functional-fills

Render `checkerboard` from broadcast arithmetic rather than strided
writes into a zero buffer, and take the output dtype as a parameter
defaulting to `uint16`, rejecting one that cannot hold the stated bit
depth — `display_patterns/patterns/_fills.py`,
`display_patterns/patterns/_backend.py`. Depends on §road:torch-gate.
§spec:backend-portability.

### Counter panel on an array frame index §road:functional-counter

Render the counter panel from broadcast arithmetic over bit positions
with `frame` accepted as a scalar or zero-dimensional array, bounding
the counter at 31 bits —
`display_patterns/patterns/_counter_panel.py`. Depends on
§road:torch-gate. §spec:backend-portability.

### Backend support on the public surface §road:backend-surface

State the supported backends, the dtype a caller passes on each, and
the counter's bit bound in the package's README and entry-point
docstrings, and record the measured device legs — `README.md`,
`display_patterns/patterns/__init__.py`. Depends on
§road:functional-fills and §road:functional-counter.
§spec:backend-portability.

**Verify:** In a fresh environment with torch installed, `pytest`
passes with no skipped backend legs. On an Apple-silicon host,
`pytest -m mps` renders a checkerboard and a counter panel on
`torch.device("mps")` — the checkerboard at a caller-stated `int32`,
since `uint16` is rejected there — and `decode_counter` recovers the
encoded index from the MPS-resident overlay. On a CUDA host,
`pytest -m cuda` does the same on `torch.device("cuda")`. Wrapping a
`render_counter_panel` call in `torch.compile` and stepping the frame
index over a hundred frames compiles once, not once per frame. The
numpy path renders the same arrays as before these changes.
