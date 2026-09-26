"""Independent CPU mathematical tests for the clean-break integration contract."""
from types import SimpleNamespace

import numpy as np
import pytest

from RBDReference import RBDReference


class ScalarSystem(RBDReference):
    """Exactly solvable scalar dynamics; use the real reference step/gradient."""
    def __init__(self, stiffness=0.0):
        self.stiffness = stiffness
        self.robot = SimpleNamespace(get_num_vel=lambda: 1, robot_has_spherical=lambda: False)

    def integrate(self, q, increment):
        return np.asarray(q) + increment

    def dIntegrate(self, q, increment, with_respect_to):
        return np.eye(1)

    def forward_dynamics(self, q, qd, u, f_ext=None):
        return -self.stiffness * np.asarray(q) + u + (0 if f_ext is None else f_ext)

    def forward_dynamics_gradient(self, q, qd, u, f_ext=None):
        return -self.stiffness * np.eye(1), np.zeros((1, 1))

    def minv(self, q):
        return np.eye(1)


@pytest.mark.parametrize("scheme", ["euler", "semi_implicit_euler", "constant_acceleration"])
def test_single_stage_exact_update_and_gradient(scheme):
    system = ScalarSystem()
    dt, q, v, u, force = 0.2, np.array([0.3]), np.array([-0.4]), np.array([0.7]), np.array([0.2])
    factor = {"euler": 0.0, "semi_implicit_euler": 1.0, "constant_acceleration": 0.5}[scheme]
    actual = system.integrator(q, v, u, dt, scheme, f_ext=force)
    np.testing.assert_allclose(actual, [q[0] + dt*v[0] + factor*dt**2*(u[0]+force[0]),
                                       v[0] + dt*(u[0]+force[0])])
    np.testing.assert_allclose(system.integrator_gradient(q, v, u, dt, scheme, f_ext=force),
                               [[1, dt, factor*dt**2], [0, 1, dt]])


@pytest.mark.parametrize("scheme", ["rk3", "si_euler", "not_a_scheme"])
@pytest.mark.parametrize("method", ["integrator", "integrator_gradient", "plant_step_hessian"])
def test_removed_or_unknown_names_fail_before_dynamics(scheme, method):
    with pytest.raises(ValueError, match="Unknown integrator_type"):
        getattr(ScalarSystem(), method)(np.zeros(1), np.zeros(1), np.zeros(1), 0.1, scheme)
