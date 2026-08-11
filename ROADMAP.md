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
