"""Rotor armature against Pinocchio's model.armature (2026-11-03).

The reference's ``set_armature`` adds a per-v-slot reflected inertia to the
joint-space inertia the way Pinocchio does since 3.0: ``rnea`` gains
``armature * qdd``, ``crba`` and ``computeMinverse`` see ``M + diag(a)``,
``aba`` solves with it, and forward dynamics with its first derivative
follow. The armature is random on every single-DoF slot and zero on the
floating root. Robots: iiwa14 (fixed and floating), go2 (floating), gen3
(continuous joints) and fr3 (mimic: the reduced vector is expanded onto the
mimicked joints on the pinocchio side).
"""
import numpy as np
import pytest

from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases
from RBDReference.tests.state_sampling import build_dynamics_samples
from RBDReference.tests.comparators import assert_close


def _params():
    params = []
    for base_mode in ("fixed", "floating"):
        for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
            spec = case["spec"]
            if (spec.robot_id, base_mode) not in {("iiwa14", "fixed"), ("iiwa14", "floating"),
                                                  ("go2", "floating"), ("gen3", "fixed"), ("fr3", "fixed")}:
                continue
            marks = [pytest.mark.pinocchio_equivalence, pytest.mark.developer_only,
                     pytest.mark.robot_smoke]
            if base_mode == "floating":
                marks.append(pytest.mark.floating_base)
            params.append(pytest.param(spec, base_mode, id=f"{spec.robot_id}-{base_mode}", marks=marks))
    return params


def _random_armature(project_model, seed=7):
    ref = project_model.reference
    robot = ref.robot
    nv = robot.get_num_vel()
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.05, 0.5, nv)
    for jid in range(robot.get_num_joints()):
        if (robot.floating_base and jid == 0) or robot.joint_is_spherical(jid):
            inds = robot.get_joint_index_v(jid)
            a[list(inds if isinstance(inds, (list, tuple)) else [inds])] = 0.0
    return a


@pytest.mark.parametrize(("spec", "base_mode"), _params())
def test_armature_matches_pinocchio(spec, base_mode, project_model, pinocchio_model):
    a = _random_armature(project_model)
    project_model.reference.set_armature(a)
    pinocchio_model.set_armature(a)
    rid = spec.robot_id
    for sample in build_dynamics_samples(project_model):
        q, qd, qdd = sample.q, sample.qd, sample.qdd
        assert_close(project_model.inverse_dynamics(q, qd, qdd), pinocchio_model.inverse_dynamics(q, qd, qdd),
                     algorithm="inverse_dynamics", robot_id=rid)
        assert_close(project_model.crba(q), pinocchio_model.crba(q), algorithm="minv", robot_id=rid)
        assert_close(project_model.minv(q), pinocchio_model.minv(q), algorithm="minv", robot_id=rid)
        tau = pinocchio_model.inverse_dynamics(q, qd, qdd)
        assert_close(project_model.aba(q, qd, tau), pinocchio_model.aba(q, qd, tau), algorithm="aba", robot_id=rid)
        # The same round-trip class as aba (the libraries' tau round-off through Minv).
        assert_close(project_model.forward_dynamics(q, qd, tau), pinocchio_model.forward_dynamics(q, qd, tau),
                     algorithm="aba", robot_id=rid)
    # the armature changes the answer: the plain model must differ
    project_model.reference.set_armature(np.zeros_like(a))
    sample = build_dynamics_samples(project_model)[-1]
    plain = project_model.minv(sample.q)
    project_model.reference.set_armature(a)
    assert not np.allclose(plain, project_model.minv(sample.q), rtol=1e-6, atol=1e-9)


@pytest.mark.parametrize(("spec", "base_mode"), [p for p in _params() if "fr3" not in p.id])
def test_armature_forward_dynamics_gradient_matches_pinocchio(spec, base_mode, project_model, pinocchio_model):
    """dqdd/dq and dqdd/dqd with armature: -Minv_a dc/dx against pin's computeABADerivatives."""
    a = _random_armature(project_model, seed=11)
    project_model.reference.set_armature(a)
    pinocchio_model.set_armature(a)
    for sample in build_dynamics_samples(project_model)[:2]:
        q, qd, qdd = sample.q, sample.qd, sample.qdd
        tau = pinocchio_model.inverse_dynamics(q, qd, qdd)
        actual = project_model.forward_dynamics_gradient(q, qd, tau)
        expected = pinocchio_model.forward_dynamics_gradient(q, qd, tau)
        assert_close(actual, expected, algorithm="forward_dynamics", robot_id=spec.robot_id)


@pytest.mark.parametrize(("spec", "base_mode"), [p for p in _params() if p.id == "go2-floating"])
def test_set_armature_refuses_root_and_bad_length(spec, base_mode, project_model):
    ref = project_model.reference
    nv = ref.robot.get_num_vel()
    bad = np.zeros(nv); bad[0] = 1.0
    with pytest.raises(ValueError):
        ref.set_armature(bad)
    with pytest.raises(ValueError):
        ref.set_armature(np.zeros(nv + 1))
    good = np.zeros(nv); good[6:] = 0.1
    ref.set_armature(good)
    assert np.array_equal(ref.armature, good)


@pytest.mark.parametrize(("spec", "base_mode"), [p for p in _params() if p.id == "go2-floating"])
def test_armature_validation_preserves_installed_values(spec, base_mode, project_model):
    ref = project_model.reference
    nv = ref.robot.get_num_vel()
    good = np.zeros(nv)
    good[6:] = .2
    ref.set_armature(good)
    cases = [good[None], good.astype(complex)]
    for value in (-.1, -1e-100, np.nan, np.inf):
        bad = good.copy()
        bad[6] = value
        cases.append(bad)
    bad_root = good.copy()
    bad_root[0] = 1e-100
    cases.append(bad_root)
    for bad in cases:
        with pytest.raises(ValueError):
            ref.set_armature(bad)
        np.testing.assert_array_equal(ref.armature, good)
    # The model owns a copy, not the caller's input buffer.
    good[:] = 0
    assert np.all(ref.armature[6:] == .2)
