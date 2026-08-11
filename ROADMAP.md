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
testing complete and the PyPI org is confirmed (pending the possible
OpenLEDEval → OpenDisplayEval rename); consumers integrate via a git
dependency meanwhile.

**Verify:** In a fresh environment, `pip install display-patterns`
imports and renders a checkerboard with numpy alone; `pip install
"display-patterns[charts,io]"` renders a YAML-authored chart and
writes it to TIFF; for identical parameters the rendered arrays equal
bmd-signal-gen's in-tree output array-for-array.

## Rendering model reshape §road:render-reshape

The catalog moves onto the spec'd rendering model once the verbatim
equivalence is banked — visible, separate changes (§spec:extraction).

### Frame-indexed namespace API §road:frame-indexed-api

Reshape catalog entry points to the pure rendering signature —
parameters, frame index, caller-supplied array namespace with
optional device, numpy default — with stills ignoring the index.
§spec:render-model. Depends on §road:extraction-spine.

### Counter panel codec §road:counter-panel

Port the temporal-alignment counter panel (geometry, encode, decode)
from backlit_molecule's probe math into the core catalog on the new
signature. §spec:catalog. Depends on §road:frame-indexed-api.

**Verify:** Render counter-panel frame N under numpy and decode N
back from the array; repeat under torch when available and decoded
values match; a 12-bit checkerboard authored at (4095, 2048, 0)
contains exactly those integers; rendering the same still at two
frame indices yields identical arrays.
