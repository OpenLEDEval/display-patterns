## [0.2.1](https://github.com/OpenDisplayEval/display-patterns/compare/v0.2.0...v0.2.1) (2026-08-30)


### Bug Fixes

* **counter-panel:** widen the frame index annotation to array data ([0d34786](https://github.com/OpenDisplayEval/display-patterns/commit/0d347869c0e66733ff5bea1eb0edb1f1d7856c63))


### Performance Improvements

* **counter-panel:** extract bits arithmetically, from an array index ([d7dc20e](https://github.com/OpenDisplayEval/display-patterns/commit/d7dc20e1172737514865c11d8472d10a4e731ee4))
* **fills:** render the checkerboard functionally, at a caller's dtype ([f8ae0f3](https://github.com/OpenDisplayEval/display-patterns/commit/f8ae0f3d1d27bc5c8efc6cc758b21be2e26fa46f))

# [0.2.0](https://github.com/OpenLEDEval/display-patterns/compare/v0.1.0...v0.2.0) (2026-08-11)


### Bug Fixes

* **counter-panel:** one host transfer per decode; reject zero-bit geometry ([e83f790](https://github.com/OpenLEDEval/display-patterns/commit/e83f790fc998a9f2df27c11376f46bd0aa6ad41c))


### Features

* **patterns:** port the temporal-alignment counter panel codec ([c7d38af](https://github.com/OpenLEDEval/display-patterns/commit/c7d38af17bd382565f717392ae204cadac78c573))
* **patterns:** reshape the catalog onto the frame-indexed namespace API ([826b1a2](https://github.com/OpenLEDEval/display-patterns/commit/826b1a2ddbed2fa30ea5f94d5b55b63f2fb5e3e8))


### Performance Improvements

* **fills:** write tiles directly at the target dtype ([f0b54d6](https://github.com/OpenLEDEval/display-patterns/commit/f0b54d653eded700fa8e3ecf266ffcf1c63c70e3))
* **package:** resolve root exports lazily ([6b1521b](https://github.com/OpenLEDEval/display-patterns/commit/6b1521b618fec6ca5c3d5d0f1b6fabc1e1341366))

# [0.1.0](https://github.com/OpenLEDEval/display-patterns/compare/v0.0.0...v0.1.0) (2026-08-11)


### Bug Fixes

* **tests:** find the bmd-signal-gen checkout from a nested worktree ([e14caa6](https://github.com/OpenLEDEval/display-patterns/commit/e14caa6ce5e4ecb3d2c4cfaeaf687a028907dd61))


### Features

* move the bmd-signal-gen pattern and chart catalog in verbatim ([5ac4107](https://github.com/OpenLEDEval/display-patterns/commit/5ac410703072e9b2f5a14aa087d62fa240e7adc5))
