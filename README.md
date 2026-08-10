# display-patterns

**Deterministic display test patterns, device-free and exact.**

Test-pattern math tends to live trapped inside device tools. This
library is the importable version: solid fields, checkerboards,
charts, and machine-readable temporal-alignment patterns, rendered as
pure functions of their parameters and a frame index. Integer
patterns deliver exact code values at a stated bit depth — the
property measurement work depends on — and the render path never
touches a device, a clock, or a file.

The core depends on numpy alone and renders into a caller-supplied
array namespace (numpy or torch), so it serves CLI signal generators
and GPU-resident render graphs alike. Chart authoring and TIFF export
install as extras.

Extracted from
[bmd-signal-gen](https://github.com/OpenLEDEval/bmd-signal-gen),
which consumes it back over SDI. Sibling consumers and context:
[color-wrangler](https://github.com/Fuse-Technical-Group/color-wrangler)
(LED-surface characterization umbrella) and its component repos.

## Installation

```sh
pip install display-patterns              # core: numpy only
pip install "display-patterns[charts,io]" # chart authoring + TIFF export
```

Until the first PyPI release, install from a checkout:
`pip install .` (or `uv pip install .`).

## Usage

Render a 12-bit checkerboard at exact code values:

```python
from display_patterns.image_generators import ROI, PatternGenerator

generator = PatternGenerator(
    bit_depth=12, width=1920, height=1080, roi=ROI(0, 0, 1920, 1080)
)
frame = generator.generate([[4095, 2048, 0], [0, 0, 0]])  # uint16 (1080, 1920, 3)
```

With the `charts` and `io` extras, author a chart in YAML, render it,
and write a 16-bit TIFF:

```python
from display_patterns.charts import render_chart, write_chart_tiff
from display_patterns.charts.loaders import load_chart

layout = load_chart("my_chart.yaml")
image = render_chart(layout, bit_depth=12)
write_chart_tiff("my_chart.tiff", image, layout)
```

## API

- `display_patterns.image_generators` — core pattern math:
  `PatternGenerator`, `ROI`, `ColorRangeError`. Numpy only.
- `display_patterns.charts` — chart types, colorimetric conversion, and
  rendering (`charts` extra); TIFF read/write (`io` extra).
- `display_patterns.charts.loaders` — YAML chart definitions
  (`charts` extra).

All public functions and classes carry NumPy-style docstrings; the
package ships `py.typed` for static type checking.

## Governance

- [REQUIREMENTS.md](REQUIREMENTS.md) — the problem space
- [SPEC.md](SPEC.md) — design and rationale
- [ROADMAP.md](ROADMAP.md) — work remaining

## License

BSD 3-Clause.
