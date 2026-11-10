"""Constant independent-slot armature composes with nonlinear followers."""
import numpy as np
import pytest

from RBDReference import RBDReference
from RBDReference.tests.test_embedding_projection import _parse, _poly_urdf, _random_state


@pytest.mark.parametrize("floating", [False, True])
def test_polynomial_armature_composition(tmp_path, floating):
    robot = _parse(_poly_urdf(tmp_path), floating)
    ref = RBDReference(robot)
    plain = RBDReference(robot)
    assert ref._embedding is not None
    armature = np.zeros(robot.get_num_vel())
    armature[list(ref.armature_slots)] = [.2, .4]
    ref.set_armature(armature)
    rng = np.random.default_rng(713)
    for _ in range(3):
        q, v, a = _random_state(robot, rng)
        loads = rng.uniform(-.1, .1, (robot.get_num_bodies(), 6))
        expected_mass = plain.crba(q) + np.diag(armature)
        np.testing.assert_allclose(ref.crba(q), expected_mass, rtol=1e-12, atol=1e-12)
        tau = ref.inverse_dynamics(q, v, a, f_ext=loads)[0]
        np.testing.assert_allclose(tau, plain.inverse_dynamics(q, v, a, f_ext=loads)[0]
                                   + armature * a, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ref.forward_dynamics(q, v, tau, f_ext=loads), a,
                                   rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(ref.minv(q), np.linalg.inv(expected_mass), rtol=1e-10, atol=1e-10)
        expected_ke = plain.kinetic_energy(q, v) + .5 * np.dot(armature, v * v)
        np.testing.assert_allclose(ref.kinetic_energy(q, v), expected_ke, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(ref.mechanical_energy(q, v), expected_ke + plain.potential_energy(q),
                                   rtol=1e-12, atol=1e-12)
        # Independent finite differences also exercise the compact identification
        # primitive through the projected inverse inertia, with external loads.
        analytic = ref.forward_dynamics_armature_gradient(q, v, tau, f_ext=loads)
        eps = 1e-5
        numeric = np.empty_like(analytic)
        try:
            for col, slot in enumerate(ref.armature_slots):
                plus, minus = armature.copy(), armature.copy()
                plus[slot] += eps
                minus[slot] -= eps
                ref.set_armature(plus)
                upper = ref.forward_dynamics(q, v, tau, f_ext=loads)
                ref.set_armature(minus)
                lower = ref.forward_dynamics(q, v, tau, f_ext=loads)
                numeric[:, col] = (upper - lower) / (2 * eps)
        finally:
            ref.set_armature(armature)
        np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-6)
