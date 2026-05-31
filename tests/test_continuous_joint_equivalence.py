"""End-to-end verification of continuous (unbounded revolute) joints.

A URDF ``continuous`` joint is a 1-DOF unbounded revolute. GRiD models its
generalized coordinate as a RAW SCALAR ANGLE (NQ=NV=1), whereas Pinocchio
encodes it as an SO(2) ``(cos theta, sin theta)`` pair (NQ=2, NV=1, joint
model ``JointModelRUBX/Y/Z``).

The two representations are numerically IDENTICAL for all dynamics and
kinematics OUTPUTS -- transforms, M, C, tau, qdd, gradients -- because those
depend on the angle only through ``cos``/``sin``. They diverge ONLY if one
(a) compares the raw q value directly (NQ=1 vs NQ=2) or (b) integrates over
many turns and cares about the unwrapped angle.

This module pins that contract for the gen3 robot (4 continuous + 3 revolute
joints): it asserts NQ==NV on the GRiD side, and -- crucially -- verifies that
GRiD's dynamics OUTPUTS match Pinocchio even at LARGE wrapped angles
(theta = 5pi + delta), where the raw-q representations differ the most but the
outputs must agree. This is the comparator guard the joint-types plan calls
for: compare outputs, never raw q, for continuous joints.
"""
import contextlib
import io

import numpy as np
import pytest

from RBDReference.tests.comparators import assert_close
from RBDReference.tests.conftest import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases


def _gen3_fixed_case():
    for case in iter_robot_cases(MANIFEST_PATH, base_mode="fixed"):
        if case["spec"].robot_id == "gen3":
            return case["spec"]
    return None


GEN3_SPEC = _gen3_fixed_case()

pytestmark = [
    pytest.mark.pinocchio_equivalence,
    pytest.mark.developer_only,
    pytest.mark.robot_smoke,
    pytest.mark.skipif(GEN3_SPEC is None, reason="gen3 (continuous-joint robot) not in manifest"),
]


def _continuous_joint_ids(robot):
    return [
        j.get_id()
        for j in robot.get_joints_ordered_by_id()
        if j.jtype == "continuous"
    ]


@pytest.mark.parametrize(("spec", "base_mode"), [pytest.param(GEN3_SPEC, "fixed", id="gen3-fixed")])
def test_gen3_has_continuous_joints_modeled_as_unbounded_revolute(
    spec, base_mode, project_model
):
    """gen3 carries continuous joints; each is 1-DOF and NQ==NV holds overall."""
    robot = project_model.robot
    cont_ids = _continuous_joint_ids(robot)
    assert cont_ids, "gen3 should expose continuous joints"
    for jid in cont_ids:
        joint = robot.get_joint_by_id(jid)
        assert joint.get_num_dof() == 1
        lower, upper = joint.get_joint_limits()
        assert not np.isfinite(lower) and not np.isfinite(upper)
    # Raw-scalar-angle representation: no SO(2) expansion, so NQ == NV.
    assert project_model.nq == project_model.nv


@pytest.mark.parametrize(("spec", "base_mode"), [pytest.param(GEN3_SPEC, "fixed", id="gen3-fixed")])
def test_continuous_joint_dynamics_match_pinocchio_at_large_wrapped_angles(
    spec, base_mode, project_model, pinocchio_model
):
    """Dynamics OUTPUTS agree with Pinocchio even when continuous-joint angles
    are wrapped many turns past [-pi, pi] -- where the raw-q (GRiD) vs SO(2)
    (Pinocchio) representations differ most, but the physics must not."""
    robot = project_model.robot
    cont_ids = set(_continuous_joint_ids(robot))
    rng = np.random.default_rng(11)
    nv = project_model.nv

    for trial in range(4):
        base = rng.uniform(-1.0, 1.0, size=nv)
        # push the continuous-joint coordinates many full turns away
        q = base.copy()
        for jid in cont_ids:
            iq = robot.get_joint_index_q(jid)
            q[iq] = base[iq] + 2.0 * np.pi * rng.integers(-3, 4)
        qd = rng.uniform(-0.8, 0.8, size=nv)
        qdd = rng.uniform(-0.8, 0.8, size=nv)

        # Pinocchio consumes its own nq layout; the adapter handles the
        # GRiD->pin q mapping (raw angle -> (cos,sin)) internally, so we pass
        # the GRiD-layout q to both adapters.
        actual = project_model.rnea(q, qd, qdd)
        expected = pinocchio_model.rnea(q, qd, qdd)
        assert_close(actual, expected, algorithm="rnea", robot_id=spec.robot_id)

        actual_m = project_model.crba(q)
        expected_m = pinocchio_model.crba(q)
        assert_close(actual_m, expected_m, algorithm="minv", robot_id=spec.robot_id)
