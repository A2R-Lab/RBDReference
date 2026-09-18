# Pinocchio Equivalence Suite

This suite validates the pure-Python `RBDReference` implementation against
Pinocchio (C++) on the same URDF inputs — parse behavior, metadata sanity, and
every dynamics / kinematics / derivative algorithm the package exposes. It is
the trust anchor the parent GRiD repo's CUDA equivalence tests build on, but it
runs standalone on CPU only.

## Running standalone

1. Check out this repo and its sibling parser under a common parent, with the
   directories named **exactly** `RBDReference` and `URDFParser` (the suite's
   absolute imports are `RBDReference.tests.*`):

   ```bash
   git clone <RBDReference>            # -> ./RBDReference
   git clone -b modernizing-tests https://github.com/A2R-Lab/URDFParser  # -> ./URDFParser
   ```

   `URDFParser` must be on the `modernizing-tests` branch (the modernized
   parser: floating-base convention, mimic/planar/spherical tiers).

2. Install the developer dependencies:

   ```bash
   pip install -r RBDReference/requirements-dev.txt
   ```

3. Run pytest **from the common parent** (so the parent lands on `sys.path`
   and `RBDReference.*` / `URDFParser` imports resolve):

   ```bash
   python -m pytest RBDReference/tests -q
   ```

## Fast default vs. `--runslow`

A plain run is fast by default (a few minutes): the heavy big-robot cells —
second-order tensors, per-DoF finite-difference plant / parameter-gradient /
pose-Hessian oracles, the floating-base FD cross-checks — are tagged `@slow`
and deselected automatically. Every algorithm keeps live small/medium-robot
coverage, so the fast set is still a real regression gate.

```bash
python -m pytest RBDReference/tests --runslow           # everything
python -m pytest RBDReference/tests --runslow -m slow   # only the heavy cells
```

The policy lives in `tests/conftest.py` (`pytest_collection_modifyitems`), not
in ini `addopts` — when run from a parent repo, that parent's pytest config is
the active one and would ignore a submodule-local `pytest.ini`.

## Pieces

- `conftest.py` — fixtures that resolve each manifest robot and build the two
  adapters (`reference` and `pinocchio`) per (robot, base mode); the
  fast/slow policy above.
- `robot_manifest.json` + `model_sources.py` — manifest-driven robot
  acquisition. Each robot tries its vendored URDF in `robot_assets/` FIRST
  (found in this repo or a parent repo), then falls back to a
  `robot_descriptions` download.
- `tolerances.py` — per-(algorithm, robot) tolerance buckets, each with a
  written justification. Loosen a bucket only with a documented triangulation
  note explaining where the extra round-off/FD error comes from.
- `comparators.py` — `assert_close`, the single comparison helper all tests
  use.
- `ROBOT_SOURCE_LOCK.json` / `source_lock.py` — provenance for resolved robots.

## Robots

The manifest currently pins 9 robots, each in fixed- and floating-base modes:
`iiwa14`, `go2`, `g1`, `h1_2`, `fr3`, `rizon4`, `gen3`, `fetch`, `baxter`.

## pin_so_ext (second-order C++ oracle)

`equivalents/pin_so_ext/` wraps Pinocchio's C++
`ComputeRNEASecondOrderDerivatives` (not exposed by the Python bindings). Its
loader builds it on demand at first import; it needs `g++` and resolves
Pinocchio via `pkg-config` (the `pin` wheel's cmeel pkgconfig dirs are
prepended automatically) or, failing that, by probing the venv's
`cmeel.prefix` directly. No system Pinocchio install is required — the pinned
`pin<4` wheel in `requirements-dev.txt` is the whole toolchain.

## CI

`.github/workflows/ci.yml` runs exactly this shape on every push/PR: two
sibling checkouts (`RBDReference` + `URDFParser@modernizing-tests`), the
requirements-dev install, and the fast set via `python -m pytest
RBDReference/tests` from the parent (workflow-dispatch can opt into
`--runslow`).
