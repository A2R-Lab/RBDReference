"""Frame velocity and acceleration against pinocchio (2026-11-03, item C of 0.8).

``frame_velocity`` is ``J qd`` with the frame Jacobian of the chosen reference
frame, so it is ``getFrameVelocity`` for LOCAL, WORLD and LOCAL_WORLD_ALIGNED.
``frame_acceleration`` is its time derivative, ``J qdd + Jdot qd``: the spatial
acceleration (``getFrameAcceleration``) in LOCAL and WORLD, and in
LOCAL_WORLD_ALIGNED the classical acceleration of the frame origin in world
axes (``getFrameClassicalAcceleration``). Both a leaf joint and a mid-chain
joint are targets; iiwa14, go2 and gen3 on both bases.
"""
import numpy as np
import pytest

from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases
from RBDReference.tests.state_sampling import build_dynamics_samples
from RBDReference.tests.comparators import assert_close
from RBDReference.tests.test_frame_jacobian_equivalence import _frame_targets, REFERENCE_FRAMES


def _params():
    params = []
    for base_mode in ("fixed", "floating"):
        for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
            spec = case["spec"]
            if spec.robot_id not in ("iiwa14", "go2", "gen3"):
                continue
            marks = [pytest.mark.pinocchio_equivalence, pytest.mark.developer_only,
                     pytest.mark.robot_smoke]
            if base_mode == "floating":
                marks.append(pytest.mark.floating_base)
            params.append(pytest.param(spec, base_mode, id=f"{spec.robot_id}-{base_mode}", marks=marks))
    return params


@pytest.mark.parametrize(("spec", "base_mode"), _params())
def test_frame_velocity_matches_pinocchio(spec, base_mode, project_model, pinocchio_model):
    targets, _ = _frame_targets(project_model, pinocchio_model)
    assert targets, f"No frame targets for {spec.robot_id}"
    for target in targets:
        for sample in build_dynamics_samples(project_model):
            for ref in REFERENCE_FRAMES:
                actual = project_model.frame_velocity(sample.q, sample.qd, target, ref)
                expected = pinocchio_model.frame_velocity(sample.q, sample.qd, target, ref)
                assert_close(actual, expected, algorithm="frame_jacobian", robot_id=spec.robot_id)


@pytest.mark.parametrize(("spec", "base_mode"), _params())
def test_frame_acceleration_matches_pinocchio(spec, base_mode, project_model, pinocchio_model):
    targets, _ = _frame_targets(project_model, pinocchio_model)
    assert targets, f"No frame targets for {spec.robot_id}"
    for target in targets:
        for sample in build_dynamics_samples(project_model):
            for ref in REFERENCE_FRAMES:
                actual = project_model.frame_acceleration(sample.q, sample.qd, sample.qdd, target, ref)
                expected = pinocchio_model.frame_acceleration(sample.q, sample.qd, sample.qdd, target, ref)
                assert_close(actual, expected, algorithm="frame_jacobian_dot", robot_id=spec.robot_id)
