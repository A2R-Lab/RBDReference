import importlib.util
from functools import lru_cache
from pathlib import Path

import pytest

from RBDReference.equivalents.pinocchio_backend import build_pinocchio_adapter
from RBDReference.equivalents.reference_backend import (
    ProjectParseError,
    build_project_adapter,
)
from RBDReference.tests.model_sources import (
    iter_robot_cases,
    load_manifest,
    resolve_robot_spec,
)
from RBDReference.tests.source_lock import build_lock_entry
from RBDReference.tests import MANIFEST_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUITE_ROOT = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# Fast-default / opt-in-slow policy
# ---------------------------------------------------------------------------
# The full Pinocchio equivalence suite (~1035 tests) is dominated by a small
# tail of big-robot comparisons whose *numpy* references are O(NB) or O(NB^2)
# in per-sample finite-difference / second-order work (h1_2 and g1 are the
# worst: fdsva_so alone is ~25 s/cell, the plant + parameter-gradient + pose
# Hessian references add several more multi-second cells each). That long tail
# pushes a serial run past ~2 h and an `-n 12` run past ~2 h, which blocks
# routine post-change validation.
#
# Policy: a plain `pytest RBDReference/tests/` runs the FAST set — full
# algorithm + small/medium-robot coverage, finishing in a few minutes. The
# heavy big-robot cells are tagged `@slow` (dynamically, see
# `pytest_collection_modifyitems` below) and DESELECTED by default. Run the
# exhaustive set with:
#       pytest RBDReference/tests/ --runslow          # fast + slow
#       pytest RBDReference/tests/ --runslow -m slow   # ONLY the heavy cells
#
# NOTE: the active pytest rootdir/config for this suite is the parent repo's
# `pyproject.toml` (the suite must be run from the repo root so the
# `RBDReference.*` absolute imports resolve), so an `addopts`/`pytest.ini`
# inside this submodule would be IGNORED. The deselection is therefore driven
# from this always-loaded conftest rather than from ini `addopts`.

# --- Group 1: ROBOT-driven heavy cells -------------------------------------
# These references scale with body count, so only the high-DoF
# humanoid/dual-arm/mobile bots are expensive (measured via `pytest
# --durations`: h1_2 dominates, g1 next). The small manipulators + quadruped
# stay in the fast set, so every algorithm keeps live small-robot coverage by
# default. A cell is slow only when its node id names a slow robot AND matches
# one of the heavy test substrings, so cheap algorithms are never deselected.
_SLOW_ROBOT_IDS = frozenset({"h1_2", "g1", "baxter", "fetch"})
_SLOW_ROBOT_TEST_SUBSTRINGS = (
    "fdsva_so",            # second-order FD composition (the worst tail, ~25 s/cell)
    "idsva_so",            # second-order RNEA derivatives
    "_plant_reference",    # per-DoF FD barrier/dynamics plant checks
    "parameter_gradient",  # -Minv . Y(q,qd,qdd) parameter-gradient oracle (~12 s)
    "pose_hessian",        # end-effector pose Hessian vs pinocchio / FD
    # fixed-base analytic-dJ^T/dq self-check: central-differences the exact J^T
    # over every DoF (O(nv) full Jacobian rebuilds) — only costly on big robots.
    "test_fixed_base_djt_dq_analytic_matches_fd",
)

# A few FD-oracle cross-checks finite-difference BOTH sides over every DoF, so
# they are expensive for EVERY robot (not just the big ones) — the floating
# base inflates them further. These are slow regardless of robot id:
#   * the floating-base plant FD cross-check (~8 s small bots, 50-110 s humanoids);
#   * the floating-base f_ext-gradient (FD on both project + pinocchio sides —
#     the whole `test_floating_base_f_ext_gradient...` parametrization exceeds
#     500 s wall on its own).
# Fixed-base plant cells stay fast except for the big robots (caught above), so
# every robot keeps a fast fixed-base smoke check of each by default.
_SLOW_FD_SUBSTRINGS = (
    "test_floating_base_plant_reference",
    "test_floating_base_f_ext_gradient_matches_pinocchio",
)

# --- Group 2: INTEGRATOR-driven heavy cells --------------------------------
# The `integrator_with_gradient` combined-surface cross-check is heavy because
# of the integrator math, NOT robot size: the rk4 cells (4 sub-evals) and the
# FD-jacobian self-consistency test dominate regardless of robot (fr3-fixed-rk4
# is ~35 s, iiwa14-fixed-rk4 ~15 s). The plain integrator + gradient surfaces
# are still covered fast by test_integrator_pinocchio_equivalence.py and
# test_integrator_gradient_fd_sanity.py, so gating these is no coverage loss.
_SLOW_INTEGRATOR_SUBSTRINGS = (
    "test_combined_gradient_is_jacobian",  # FD-jacobian cross-check (the worst cells)
    "-rk4",                                # 4-stage integrator, heavy for every robot
)


def _is_slow_nodeid(nodeid: str) -> bool:
    if any(robot_id in nodeid for robot_id in _SLOW_ROBOT_IDS) and any(
        sub in nodeid for sub in _SLOW_ROBOT_TEST_SUBSTRINGS
    ):
        return True
    if "test_integrator_with_gradient_equivalence" in nodeid and any(
        sub in nodeid for sub in _SLOW_INTEGRATOR_SUBSTRINGS
    ):
        return True
    if any(sub in nodeid for sub in _SLOW_FD_SUBSTRINGS):
        return True
    return False


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help=(
            "Run the exhaustive big-robot RBDReference equivalence cells that "
            "are deselected by default (heavy numpy references; see conftest)."
        ),
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "pinocchio_equivalence")
    config.addinivalue_line("markers", "developer_only")
    config.addinivalue_line("markers", "floating_base")
    config.addinivalue_line("markers", "robot_smoke")
    config.addinivalue_line("markers", "robot_curated")
    config.addinivalue_line("markers", "robot_nightly")
    config.addinivalue_line(
        "markers",
        "slow: heavy big-robot equivalence cell; deselected unless --runslow.",
    )


def pytest_collection_modifyitems(config, items):
    """Tag the heavy big-robot cells `slow` and deselect them by default.

    With `--runslow` everything is kept (and `-m slow` can then select ONLY the
    heavy cells). Without it, the slow cells are removed from the run entirely
    (deselected, not skipped) so a plain `pytest` finishes in a few minutes.
    """
    runslow = config.getoption("--runslow")
    slow_marker = pytest.mark.slow
    kept, deselected = [], []
    for item in items:
        if _is_slow_nodeid(item.nodeid):
            item.add_marker(slow_marker)
            if not runslow:
                deselected.append(item)
                continue
        kept.append(item)
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        items[:] = kept


def require_developer_dependencies():
    missing = []
    for module_name in ("pinocchio", "robot_descriptions"):
        if importlib.util.find_spec(module_name) is None:
            missing.append(module_name)
    if missing:
        pytest.fail(
            "Missing developer-only dependencies for the Pinocchio equivalence suite: "
            f"{', '.join(missing)}. Run ./developer_install.sh before executing these tests."
        )


def build_case_params(base_mode=None):
    params = []
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        spec = case["spec"]
        mode = case["base_mode"]
        marks = [
            pytest.mark.pinocchio_equivalence,
            pytest.mark.developer_only,
            getattr(pytest.mark, f"robot_{spec.tier}"),
        ]
        if mode == "floating":
            marks.append(pytest.mark.floating_base)
        params.append(
            pytest.param(
                spec,
                mode,
                id=f"{spec.robot_id}-{mode}",
                marks=marks,
            )
        )
    return params


@pytest.fixture(scope="session")
def developer_environment():
    require_developer_dependencies()
    return True


@pytest.fixture(scope="session")
def manifest():
    return load_manifest(MANIFEST_PATH)


@pytest.fixture
def resolved_robot_spec(developer_environment, spec):
    return resolve_robot_spec(spec)


@lru_cache(maxsize=None)
def _project_model_attempt(robot_id, embodiment, source_kind, urdf_path, base_mode):
    from RBDReference.tests.model_sources import (
        ResolvedRobotModel,
        RobotSpec,
        SourceCandidate,
    )

    spec = RobotSpec(
        robot_id=robot_id,
        tier="smoke",
        embodiment=embodiment,
        source_kind=source_kind,
        description_name="",
        base_modes=[base_mode],
        preferred_variant="default",
        notes="",
        source_candidates=[SourceCandidate(source_kind=source_kind, description_name="")],
    )
    resolved = ResolvedRobotModel(
        robot_id=robot_id,
        source_kind=source_kind,
        description_name="",
        urdf_path=urdf_path,
        package_path=None,
        repository_path=None,
        repository_url=None,
        revision=None,
        notes="",
    )
    try:
        return build_project_adapter(spec, resolved, base_mode=base_mode), None
    except ProjectParseError as exc:
        return None, str(exc)


@pytest.fixture
def project_model_attempt(spec, base_mode, resolved_robot_spec):
    return _project_model_attempt(
        spec.robot_id,
        spec.embodiment,
        spec.source_kind,
        resolved_robot_spec.urdf_path,
        base_mode,
    )


@pytest.fixture
def project_model(spec, base_mode, resolved_robot_spec, project_model_attempt):
    model, error = project_model_attempt
    if error is not None:
        pytest.skip(
            "Skipping downstream comparison because GRiD strict parse already failed for "
            f"robot_id={spec.robot_id}, embodiment={spec.embodiment}, "
            f"source_kind={spec.source_kind}, urdf={resolved_robot_spec.urdf_path}, "
            f"base_mode={base_mode}: {error}"
        )
    return model


@pytest.fixture
def pinocchio_model(spec, base_mode, resolved_robot_spec):
    try:
        return build_pinocchio_adapter(spec, resolved_robot_spec, base_mode=base_mode)
    except Exception as exc:
        pytest.fail(
            "Pinocchio load failed for "
            f"robot_id={spec.robot_id}, embodiment={spec.embodiment}, "
            f"source_kind={spec.source_kind}, urdf={resolved_robot_spec.urdf_path}, "
            f"base_mode={base_mode}: {exc}"
        )


@pytest.fixture
def source_lock_entry(spec, resolved_robot_spec):
    return build_lock_entry(spec, resolved_robot_spec)
