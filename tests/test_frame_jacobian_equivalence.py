"""Equivalence tests for general-frame geometric Jacobians, their time
variation, and operational-space (OSC) inertia (E2).

Validates the project numpy reference (`RBDReference.frame_jacobian`,
`.frame_jacobian_dot`, `.osc_inertia`) against pinocchio's:

  * ``getFrameJacobian`` / ``getJointJacobian``               (J, 3 frames)
  * ``getFrameJacobianTimeVariation`` /
    ``getJointJacobianTimeVariation`` (via
    ``computeJointJacobiansTimeVariation``)                    (Jdot)
  * ``(J pin.computeMinverse J^T)^{-1}``                        (Lambda)

for the three ``pin.ReferenceFrame`` conventions (LOCAL, WORLD,
LOCAL_WORLD_ALIGNED) on a fixed-base (iiwa14) and a floating-base (go2) robot.
"""

import numpy as np
import pytest

from RBDReference.tests.conftest import build_case_params
from RBDReference.tests.model_sources import iter_robot_cases
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.state_sampling import build_dynamics_samples
from RBDReference.tests.comparators import assert_close
from RBDReference.tests.tolerances import get_tolerance


REFERENCE_FRAMES = ("LOCAL", "WORLD", "LOCAL_WORLD_ALIGNED")


def _frame_targets(project_model, pinocchio_model):
    """A leaf/end-effector joint target (>= 6 upstream DOFs for a well-posed
    OSC inertia) plus a non-leaf joint target (J / Jdot only)."""
    mimic = set(getattr(pinocchio_model, "mimic_joint_names", []))
    names = [n for n in project_model.joint_names if n not in mimic
             and n in pinocchio_model.frame_names]
    if not names:
        return [], []
    leaf = names[-1]
    mid = names[len(names) // 2]
    osc_targets = [leaf]
    jac_targets = sorted({leaf, mid})
    return jac_targets, osc_targets


def _make_frame_case_params(base_mode):
    params = []
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        spec = case["spec"]
        if spec.robot_id not in ("iiwa14", "go2", "gen3"):
            continue
        marks = [pytest.mark.pinocchio_equivalence, pytest.mark.developer_only,
                 pytest.mark.robot_smoke]
        if base_mode == "floating":
            marks.append(pytest.mark.floating_base)
        params.append(pytest.param(spec, base_mode,
                                    id=f"{spec.robot_id}-{base_mode}", marks=marks))
    return params


_PARAMS = _make_frame_case_params("fixed") + _make_frame_case_params("floating")


@pytest.mark.parametrize(("spec", "base_mode"), _PARAMS)
def test_frame_jacobian_matches_pinocchio(spec, base_mode, project_model, pinocchio_model):
    jac_targets, _ = _frame_targets(project_model, pinocchio_model)
    if not jac_targets:
        pytest.skip(f"No frame targets for {spec.robot_id}")
    for target in jac_targets:
        for sample in build_dynamics_samples(project_model):
            for ref in REFERENCE_FRAMES:
                actual = project_model.frame_jacobian(sample.q, target, ref)
                expected = pinocchio_model.frame_jacobian(sample.q, target, ref)
                assert_close(actual, expected, algorithm="frame_jacobian", robot_id=spec.robot_id)


@pytest.mark.parametrize(("spec", "base_mode"), _PARAMS)
def test_frame_jacobian_dot_matches_pinocchio(spec, base_mode, project_model, pinocchio_model):
    jac_targets, _ = _frame_targets(project_model, pinocchio_model)
    if not jac_targets:
        pytest.skip(f"No frame targets for {spec.robot_id}")
    for target in jac_targets:
        for sample in build_dynamics_samples(project_model):
            for ref in REFERENCE_FRAMES:
                actual = project_model.frame_jacobian_dot(sample.q, sample.qd, target, ref)
                expected = pinocchio_model.frame_jacobian_dot(sample.q, sample.qd, target, ref)
                assert_close(actual, expected, algorithm="frame_jacobian_dot", robot_id=spec.robot_id)


@pytest.mark.parametrize(("spec", "base_mode"), _PARAMS)
def test_osc_inertia_matches_pinocchio(spec, base_mode, project_model, pinocchio_model):
    _, osc_targets = _frame_targets(project_model, pinocchio_model)
    if not osc_targets:
        pytest.skip(f"No OSC frame targets for {spec.robot_id}")
    # Lambda = inv(A), A = J Minv J^T. A float64 inverse carries an entrywise
    # relative error of about u*cond(A), u = 2^-53, so the entrywise class is
    # meaningful below cond(A) ~1e8 and ill-posed above it; the check that
    # holds at every conditioning is the normwise backward error of each
    # column of the reference Lambda as the solution of A_pin x = e_j
    # (Rigal and Gaches 1967; the same two-tier treatment as the CUDA
    # harness, guide 7.z48). A sample used to be skipped above cond 1e8 and
    # the cell skipped when every sample was; now every sample is checked.
    for target in osc_targets:
        for sample in build_dynamics_samples(project_model):
            for ref in REFERENCE_FRAMES:
                J = pinocchio_model.frame_jacobian(sample.q, target, ref)
                Minv = pinocchio_model.minv(sample.q)
                A = J @ Minv @ J.T
                cond = float(np.linalg.cond(A))
                tag = f"{spec.robot_id}-{base_mode} target {target} {ref} cond {cond:.2e}"
                try:
                    L = np.asarray(project_model.osc_inertia(sample.q, target, ref), dtype=np.float64)
                except np.linalg.LinAlgError:
                    L = None
                if L is None or not np.isfinite(L).all():
                    # The reference's inverse refuses (or returns non-finite) only
                    # where A is singular at float64 working precision: the q=0
                    # corner of a planar-reachable arm. Anywhere else it is a bug.
                    assert cond >= 2.0 ** 53, f"Lambda {tag}: no inverse although A is nonsingular at float64"
                    continue
                resid = A @ L - np.eye(6)
                norm_a = float(np.linalg.norm(A, 2))
                eta = max(float(np.linalg.norm(resid[:, j]))
                          / (norm_a * max(float(np.linalg.norm(L[:, j])), np.finfo(np.float64).tiny))
                          for j in range(6))
                eta_tol = get_tolerance("minv", robot_id=spec.robot_id).rtol
                assert eta <= eta_tol, f"Lambda {tag}: normwise backward error {eta:.3e} > {eta_tol:.0e}"
                if cond < 1e8:
                    expected = pinocchio_model.osc_inertia(sample.q, target, ref)
                    assert_close(L, expected, algorithm="osc_inertia", robot_id=spec.robot_id)
