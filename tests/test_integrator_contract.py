"""Independent CPU mathematical tests for the clean-break integration contract."""
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from RBDReference import RBDReference
from URDFParser import URDFParser

SCHEMES = ("euler", "semi_implicit_euler", "constant_acceleration", "trapezoidal", "midpoint", "rk4")


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


@pytest.mark.parametrize("scheme,order", [("midpoint", 2), ("trapezoidal", 2), ("rk4", 4)])
def test_oscillator_independent_convergence_order(scheme, order):
    system = ScalarSystem(stiffness=4.0)
    # q''=-4q, q0=1, v0=0; analytic solution at T=1.
    expected = np.array([np.cos(2.0), -2*np.sin(2.0)])
    errors = []
    for steps in (20, 40, 80):
        state = np.array([1.0, 0.0])
        for _ in range(steps):
            state = system.integrator(state[:1], state[1:], np.zeros(1), 1/steps, scheme)
        errors.append(np.linalg.norm(state - expected))
    orders = np.log2(np.asarray(errors[:-1]) / errors[1:])
    np.testing.assert_allclose(orders, order, atol=0.12, rtol=0)


@pytest.mark.parametrize("scheme", ["constant_acceleration", "trapezoidal", "midpoint", "rk4"])
def test_full_state_constant_acceleration(scheme):
    state = np.zeros(2)
    for _ in range(10):
        state = ScalarSystem().integrator(state[:1], state[1:], np.ones(1), 0.1, scheme)
    np.testing.assert_allclose(state, [0.5, 1.0], atol=2e-15, rtol=0)


@pytest.mark.parametrize("scheme", SCHEMES)
def test_scalar_step_gradient_with_nonzero_force(scheme):
    system = ScalarSystem(stiffness=3.0)
    z = np.array([0.4, -0.2, 0.7])
    force = np.array([0.3])
    jac = system.integrator_gradient(z[:1], z[1:2], z[2:], 0.07, scheme, f_ext=force)
    eps = 1e-6
    numeric = np.empty_like(jac)
    for j in range(3):
        delta = eps*np.eye(3)[j]
        zp, zm = z+delta, z-delta
        numeric[:, j] = (system.integrator(zp[:1], zp[1:2], zp[2:], 0.07, scheme, f_ext=force)
                         - system.integrator(zm[:1], zm[1:2], zm[2:], 0.07, scheme, f_ext=force))/(2*eps)
    np.testing.assert_allclose(jac, numeric, rtol=1e-8, atol=1e-9)


@pytest.fixture(scope="module", params=[("iiwa14", False), ("go2", True)])
def robot_reference(request):
    name, floating = request.param
    robot = URDFParser().parse(str(Path(__file__).parents[1] / "robot_assets" / f"{name}.urdf"),
                               floating_base=floating)
    assert robot is not None
    ref = RBDReference(robot)
    q = np.zeros(robot.get_num_pos())
    if floating:
        q[6] = 1.0
    nv = robot.get_num_vel()
    q = ref.integrate(q, np.linspace(-0.15, 0.2, nv))
    return ref, q


@pytest.mark.parametrize("scheme", SCHEMES)
def test_robot_step_tangent_gradient_with_external_forces(robot_reference, scheme):
    ref, q = robot_reference
    nv, nq = ref.robot.get_num_vel(), len(q)
    v, u = np.linspace(-0.2, 0.3, nv), np.linspace(0.1, 0.5, nv)
    force = np.linspace(-0.02, 0.03, 6*ref.robot.get_num_bodies()).reshape(-1, 6)
    dt, eps = 0.004, 2e-6
    value = ref.integrator(q, v, u, dt, scheme, f_ext=force)
    analytic = ref.integrator_gradient(q, v, u, dt, scheme, f_ext=force)
    numeric = np.empty((2*nv, 3*nv))
    for j in range(3*nv):
        delta = eps*np.eye(3*nv)[j]
        plus = ref.integrator(ref.integrate(q, delta[:nv]), v+delta[nv:2*nv],
                              u+delta[2*nv:], dt, scheme, f_ext=force)
        minus = ref.integrator(ref.integrate(q, -delta[:nv]), v-delta[nv:2*nv],
                               u-delta[2*nv:], dt, scheme, f_ext=force)
        numeric[:nv, j] = (ref.difference(value[:nq], plus[:nq])
                           - ref.difference(value[:nq], minus[:nq]))/(2*eps)
        numeric[nv:, j] = (plus[nq:] - minus[nq:])/(2*eps)
    np.testing.assert_allclose(analytic, numeric, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("scheme", ["trapezoidal", "midpoint", "rk4"])
def test_floating_rotation_order_two_with_independent_quaternion_ode(scheme):
    from scipy.integrate import solve_ivp

    robot = URDFParser().parse(str(Path(__file__).parents[1] / "robot_assets/go2.urdf"), floating_base=True)
    ref = RBDReference(robot)
    nq, nv = robot.get_num_pos(), robot.get_num_vel()
    q0 = np.zeros(nq); q0[6] = 1
    v0 = np.zeros(nv); v0[3:6] = [0.7, -0.2, 0.3]
    acceleration = np.zeros(nv); acceleration[3:6] = [-0.3, 0.8, 0.1]
    # Prescribed acceleration isolates manifold stage math from robot dynamics.
    ref.forward_dynamics = lambda q, v, u, f_ext=None: acceleration.copy()

    def quaternion_rhs(t, quat):
        omega = v0[3:6] + t*acceleration[3:6]
        xyz, w = quat[:3], quat[3]
        return 0.5*np.r_[w*omega + np.cross(xyz, omega), -xyz @ omega]

    exact = solve_ivp(quaternion_rhs, (0, 0.5), q0[3:7], method="DOP853", rtol=2e-13, atol=2e-14)
    assert exact.success
    q_exact = q0.copy(); q_exact[3:7] = exact.y[:, -1]
    errors = []
    for steps in (10, 20, 40):
        state = np.r_[q0, v0]
        for _ in range(steps):
            state = ref.integrator(state[:nq], state[nq:], np.zeros(nv), 0.5/steps, scheme)
        np.testing.assert_allclose(np.linalg.norm(state[3:7]), 1, atol=2e-14)
        errors.append(np.linalg.norm(ref.difference(q_exact, state[:nq])))
    np.testing.assert_allclose(np.log2(np.asarray(errors[:-1])/errors[1:]), 2, atol=0.04, rtol=0)


@pytest.mark.parametrize("scheme", ["trapezoidal", "midpoint", "rk4"])
def test_spherical_multistage_value_and_explicit_gradient_limit(scheme):
    import URDFParser as parser_package

    fixture = Path(parser_package.__file__).parent / "tests/fixtures/spherical_arm.urdf"
    robot = URDFParser().parse(str(fixture))
    ref = RBDReference(robot)
    q = np.array([0., 0., 0., 1., 0.2])
    v = np.linspace(0.1, 0.4, robot.get_num_vel())
    state = ref.integrator(q, v, v, 0.002, scheme)
    assert state.shape == (robot.get_num_pos() + robot.get_num_vel(),)
    assert np.isfinite(state).all()
    np.testing.assert_allclose(np.linalg.norm(state[:4]), 1, atol=1e-14)
    with pytest.raises(NotImplementedError, match="Spherical-joint multi-stage"):
        ref.integrator_gradient(q, v, v, 0.002, scheme)
    with pytest.raises(NotImplementedError, match="only euler and semi_implicit_euler"):
        ref.plant_step_hessian(q, v, v, 0.002, scheme)
