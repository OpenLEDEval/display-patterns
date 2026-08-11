# display-patterns — Specification

Declarative description of the system. Each section states what the
system does and why. See ROADMAP.md for work remaining.

## Problem §spec:problem

*Status: complete*

Display test-pattern math lives trapped inside device tools
(§req:problem-statement). bmd-signal-gen's pattern and chart modules
import nothing from its DeckLink layer yet ship only inside the device
tool; backlit_molecule re-derived frame-counter panel math for want of
an importable source. General imaging libraries carry color-management
opinions and cannot make a measurement claim: driving *exact* integer
code values at a stated bit depth is the point, and a library that
rescales or quantizes behind the caller's back defeats it.

display-patterns is that importable source: deterministic pattern
math, device-free, exact by construction. bmd-signal-gen's SPEC
records the extraction decision on its side (`§spec:pattern-library`
there).

## Package shape §spec:package-shape

*Status: in progress*

The distribution is `display-patterns` on PyPI; the import package is
`display_patterns`. The core installs with numpy as its only
dependency and renders every core catalog pattern
(§req:success-criteria). Optional extras price heavier dependencies
separately (§req:quality-attributes footprint):

- `charts` — chart definitions, colorimetric conversion, and chart
  rendering (colour-science, Pillow, PyYAML).
- `io` — file export (tifffile).

**Why extras and not packages:** one repository, one version, one
release cycle keeps the extraction and its consumers simple; the
extras keep a numpy-only consumer from paying for an imaging stack it
never imports. Two packages would add a version seam with no consumer
on the far side of it.

The package runs on CPython 3.12+ on macOS, Linux, and Windows
(§req:constraints); nothing in it opens a device, a socket, or a
display.

## Rendering model §spec:render-model

*Status: complete*

A pattern is a pure function: parameters (geometry, values) and a
frame index in, an array out. The same inputs produce identical
arrays on every run, platform, and backend (§req:success-criteria).
Stills ignore the frame index; temporal patterns use it as their only
time source. The render path performs no I/O and holds no state.

**Why frame-indexed and not stateful:** consumers render inside their
own frame loops on their own clocks (§req:user-stories) — a DeckLink
frame loop, a GPU graph's per-frame tick. A pure function of frame
index is seekable, parallelizable, and testable in isolation; an
iterator or clock abstraction would drag playback concerns into pure
math. Playback, pacing, and scheduling stay with consumers
(§req:constraints).

Pattern math renders through a caller-supplied array namespace (numpy
on CPU hosts, torch on GPU hosts) with an optional device placement,
so a GPU-resident consumer pays no copy through host memory
(§req:constraints). **Why a namespace parameter and not a backend
registry:** the caller already knows its backend; a parameter is the
whole mechanism, and numpy remains the default for the common case.

Values are the caller's contract (§req:quality-attributes exactness).
Integer patterns deliver requested code values unmodified and validate
them against the stated bit depth; float patterns state their range
convention. **Why no canonical value space:** measurement drives exact
drive-space integers, render graphs author scene-referred floats —
privileging either would force the other through a conversion that
manufactures error. The library owns geometry; meaning stays with the
caller.

## Catalog §spec:catalog

*Status: complete*

The core catalog, renderable with numpy alone (§req:success-criteria):

- **Solid** and **checkerboard** fills with the established color
  expansion (one color solid, two alternating, four in a 2×2 tile),
  bounded by an optional **region of interest**, validated against
  the stated bit depth with a specific error for out-of-range values.
- **Temporal-alignment counter panel** — a frame counter encoded as
  binary bit-cells in a title-safe row, with the matching decoder.
  Encode and decode ship together so any consumer can render the
  panel into one end of a video chain and read frame skew and latency
  back out of the other (§req:user-stories). The geometry is
  deterministic, so encoder and decoder agree on every cell from
  parameters alone, and decode thresholds at the cell midpoint to
  survive a lossy chain (§req:success-criteria). Ported from
  backlit_molecule's probe math (`§spec:alignment-probe` there),
  which retires its bespoke node in favor of generic primitives.

The `charts` extra adds chart production (§req:user-stories): a chart
is authored as a YAML patch list carrying colorimetric values,
converted to display RGB, and rendered with patch labels. The
conversion states its colorimetric conventions; chart math makes no
claim about the display beyond what the author provides.

**Why the counter panel is core, not an extra:** it is pure geometry
with no dependencies beyond the array namespace, and its decoder is
the library's distinguishing measurement feature — a machine-readable
pattern, not just a visible one.

Catalog growth (motion material, PLUGE, ramps, zone plates) enters as
consumers need it (§req:priorities); each entry follows the rendering
model and carries a decode side when one is meaningful.

## Extraction and compatibility §spec:extraction

*Status: complete*

The initial code was bmd-signal-gen's `bmd_sg/image_generators/` and
`bmd_sg/charts/` with their tests, moved verbatim before any reshaping
(§req:constraints). bmd-signal-gen adopted the package at its v0.2.1
behind one release cycle of deprecation shims, with bit-identical CLI
output verified array-for-array against its last pre-split commit
(§req:success-criteria); the shims live on its side
(`§road:pattern-library` there).

**Why verbatim first:** bit-identical adoption is checkable only
against an unchanged implementation. With that claim banked, the
equivalence scaffolding (test module and CI job) is retired and the
rendering-model reshape lands as separate, visible changes. The
extraction-era `image_generators` class surface stays importable and
delegates to the catalog, so adopters upgrade without code changes.

## File export §spec:file-export

*Status: in progress*

The `io` extra writes rendered arrays to 16-bit TIFF, serving the
chart review workflow (§req:user-stories). Export is one-way — the
library reads nothing back but its own counter panels.

**Why TIFF only:** the one consumer workflow that needs a file gets
one. Consumers with artifact contracts of their own — byte-exact,
deliberately untagged probe imagery — write those files themselves so
no library default can reinterpret them (ocio-display-gen records
that rationale in its `§spec:verification`).

## Scope boundaries §spec:non-goals

*Status: complete*

Out of scope, with their owners: device output and signaling
(bmd-signal-gen, pydecklink); playback, clocks, and frame pacing
(consumers' runtimes); color management and display characterization
(ocio-display-gen, color-wrangler); instrument I/O and measurement
sessions (color-wrangler, colour-specio); runtime graph integration
(backlit_molecule). The library defines the mapping from parameters
to image and nothing on either side of it.
