"""Independent residual/value finite differences for full momentum Gauss-Newton."""
from pathlib import Path

import numpy as np
import pytest

from RBDReference import RBDReference
from URDFParser import URDFParser


@pytest.fixture(scope="module", params=[("iiwa14", False), ("go2", True), ("fr3", False)])
def model(request):
    name, floating = request.param
    robot = URDFParser().parse(str(Path(__file__).parents[1] / "robot_assets" / f"{name}.urdf"),
                               floating_base=floating)
    assert robot is not None
    ref = RBDReference(robot)
    neutral = np.zeros(robot.get_num_pos())
    if floating:
        neutral[6] = 1
    return ref, neutral


@pytest.mark.parametrize("seed", [13, 29])
def test_full_momentum_gradient_and_gn_hessian(model, seed):
    ref, neutral = model
    nv = ref.robot.get_num_vel()
    rng = np.random.default_rng(seed)
    q = ref.integrate(neutral, rng.uniform(-0.4, 0.4, nv))
    v = rng.uniform(-0.5, 0.5, nv)
    h_des = np.asarray(ref.ccrba(q, v)[1]) + rng.uniform(0.2, 0.7, 6)
    weights = np.linspace(0.4, 2.0, 6)
    value, grad, hessian = ref.momentum_cost(q, v, h_des, weights)
    assert grad.shape == (2*nv,)
    assert hessian.shape == (2*nv, 2*nv)
    residual = np.asarray(ref.ccrba(q, v)[1]) - h_des
    np.testing.assert_allclose(value, 0.5 * residual @ (weights * residual))

    # Only value-layer ccrba + retract enter this oracle, never analytic dccrba.
    eps = 2e-6
    numeric_J = np.empty((6, 2*nv))
    numeric_g = np.empty(2*nv)
    for column in range(2*nv):
        delta = eps*np.eye(2*nv)[column]
        plus = np.asarray(ref.ccrba(ref.integrate(q, delta[:nv]), v+delta[nv:])[1]) - h_des
        minus = np.asarray(ref.ccrba(ref.integrate(q, -delta[:nv]), v-delta[nv:])[1]) - h_des
        numeric_J[:, column] = (plus-minus)/(2*eps)
        numeric_g[column] = (0.5*plus @ (weights*plus) - 0.5*minus @ (weights*minus))/(2*eps)
    np.testing.assert_allclose(grad, numeric_g, atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(hessian, numeric_J.T @ (weights[:, None]*numeric_J), atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(hessian, hessian.T, atol=2e-12)
    assert np.linalg.eigvalsh(hessian).min() >= -1e-12*max(1, np.linalg.norm(hessian, 2))
    # These checks fail the former velocity-only approximation.
    assert np.linalg.norm(grad[:nv]) > 1e-4
    assert np.linalg.norm(hessian[:nv, nv:]) > 1e-4


def test_zero_residual_gn_matches_derivative_of_gradient(model):
    ref, neutral = model
    nv = ref.robot.get_num_vel()
    q = ref.integrate(neutral, np.linspace(-0.2, 0.3, nv))
    v = np.linspace(0.2, -0.4, nv)
    target = np.asarray(ref.ccrba(q, v)[1]).copy()
    weights = np.linspace(0.5, 1.7, 6)
    value, grad, hessian = ref.momentum_cost(q, v, target, weights)
    assert abs(value) < 1e-25
    np.testing.assert_allclose(grad, 0, atol=1e-12)
    eps = 2e-6
    numeric_H = np.empty_like(hessian)
    for j in range(2*nv):
        delta = eps*np.eye(2*nv)[j]
        gp = ref.momentum_cost(ref.integrate(q, delta[:nv]), v+delta[nv:], target, weights)[1]
        gm = ref.momentum_cost(ref.integrate(q, -delta[:nv]), v-delta[nv:], target, weights)[1]
        numeric_H[:, j] = (gp-gm)/(2*eps)
    # Equality to the exact Hessian is asserted only at zero residual.
    np.testing.assert_allclose(hessian, numeric_H, atol=2e-7, rtol=2e-6)
