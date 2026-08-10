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

## Governance

- [REQUIREMENTS.md](REQUIREMENTS.md) — the problem space
- [SPEC.md](SPEC.md) — design and rationale
- [ROADMAP.md](ROADMAP.md) — work remaining

## License

BSD 3-Clause.
