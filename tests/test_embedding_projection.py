"""Constraint embedding by projection of the full spanning tree (2026-11-03,
item E of 0.8, _embedding.EmbeddedReference).

Two legs. (1) On a mimic robot (fr3, both bases) the projection of the full
tree through the constant coupling must equal the reduced recursions the
reference already carries, for every projected method. (2) On a synthetic
chain with a polynomial follower (GRiD's <mimic polycoef>), where the fold
multiplier varies with the leader, the reference routes through the
projection: the map and its derivatives are checked against finite
differences, inverse and forward dynamics against each other, the gradients
against chart finite differences, and the frame acceleration against the
time derivative of the frame velocity along the flow.
"""
import contextlib
import io

import numpy as np
import pytest

from URDFParser import URDFParser
from RBDReference import RBDReference
from RBDReference._embedding import EmbeddedReference
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec


def _parse(path, floating):
    with contextlib.redirect_stdout(io.StringIO()):
        return URDFParser().parse(str(path), floating_base=floating)


def _random_state(robot, rng):
    nq, nv = robot.get_num_pos(), robot.get_num_vel()
    q = rng.standard_normal(nq)
    if robot.floating_base:
        q[3:7] /= np.linalg.norm(q[3:7])
    return q, rng.standard_normal(nv), rng.standard_normal(nv)


def _fr3_path():
    for case in iter_robot_cases(MANIFEST_PATH, base_mode="fixed"):
        if case["spec"].robot_id == "fr3":
            return resolve_robot_spec(case["spec"]).urdf_path
    pytest.skip("fr3 not in the manifest")


@pytest.mark.parametrize("floating", [False, True], ids=["fixed", "floating"])
def test_linear_mimic_projection_equals_reduced_reference(floating):
    robot = _parse(_fr3_path(), floating)
    ref = RBDReference(robot)
    assert ref._embedding is None            # the constant fold stays on the recursions
    emb = EmbeddedReference(robot, RBDReference(robot.full_tree_copy()))
    assert emb.nv_full == robot.get_num_vel() + 1
    rng = np.random.default_rng(0)
    leaf = robot.get_joint_by_id(int(robot.get_leaf_nodes()[0])).get_name()
    for _ in range(3):
        q, qd, qdd = _random_state(robot, rng)
        c_r = ref.inverse_dynamics(q, qd, qdd)[0]
        c_e = emb.inverse_dynamics(q, qd, qdd)[0]
        np.testing.assert_allclose(c_e, c_r, rtol=0, atol=1e-9)
        np.testing.assert_allclose(emb.crba(q), ref.crba(q), rtol=0, atol=1e-9)
        np.testing.assert_allclose(emb.minv(q), ref.minv(q), rtol=0, atol=1e-7)
        u = rng.standard_normal(robot.get_num_vel())
        np.testing.assert_allclose(emb.forward_dynamics(q, qd, u), ref.forward_dynamics(q, qd, u), rtol=0, atol=1e-7)
        np.testing.assert_allclose(emb.aba(q, qd, u), ref.aba(q, qd, u), rtol=0, atol=1e-7)
        np.testing.assert_allclose(emb.inverse_dynamics_gradient(q, qd, qdd), ref.inverse_dynamics_gradient(q, qd, qdd), rtol=0, atol=1e-8)
        ga, gb = emb.forward_dynamics_gradient(q, qd, u), ref.forward_dynamics_gradient(q, qd, u)
        np.testing.assert_allclose(ga[0], gb[0], rtol=0, atol=1e-6)
        np.testing.assert_allclose(ga[1], gb[1], rtol=0, atol=1e-6)
        for rf in ("LOCAL", "WORLD", "LOCAL_WORLD_ALIGNED"):
            np.testing.assert_allclose(emb.frame_jacobian(q, leaf, rf), ref.frame_jacobian(q, leaf, rf), rtol=0, atol=1e-10)
            np.testing.assert_allclose(emb.frame_velocity(q, qd, leaf, rf), ref.frame_velocity(q, qd, leaf, rf), rtol=0, atol=1e-10)
            np.testing.assert_allclose(emb.frame_acceleration(q, qd, qdd, leaf, rf), ref.frame_acceleration(q, qd, qdd, leaf, rf), rtol=0, atol=1e-9)
            np.testing.assert_allclose(emb.frame_jacobian_dot(q, qd, leaf, rf), ref.frame_jacobian_dot(q, qd, leaf, rf), rtol=0, atol=1e-9)
            for a, b in zip(emb.frame_velocity_derivatives(q, qd, leaf, rf), ref.frame_velocity_derivatives(q, qd, leaf, rf)):
                np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)
            for a, b in zip(emb.frame_acceleration_derivatives(q, qd, qdd, leaf, rf), ref.frame_acceleration_derivatives(q, qd, qdd, leaf, rf)):
                np.testing.assert_allclose(a, b, rtol=0, atol=1e-8)
        # regressors, centroidal and energy quantities through the projection
        np.testing.assert_allclose(emb.inverse_dynamics_regressor(q, qd, qdd), ref.inverse_dynamics_regressor(q, qd, qdd), rtol=0, atol=1e-9)
        np.testing.assert_allclose(emb.kinetic_energy_regressor(q, qd), ref.kinetic_energy_regressor(q, qd), rtol=0, atol=1e-9)
        np.testing.assert_allclose(emb.potential_energy_regressor(q), ref.potential_energy_regressor(q), rtol=0, atol=1e-9)
        np.testing.assert_allclose(emb.com(q), ref.com(q), rtol=0, atol=1e-10)
        np.testing.assert_allclose(emb.jacobian_com(q), ref.jacobian_com(q), rtol=0, atol=1e-10)
        for a, b in zip(emb.ccrba(q, qd), ref.ccrba(q, qd)):
            np.testing.assert_allclose(a, b, rtol=0, atol=1e-9)
        np.testing.assert_allclose(emb.generalized_gravity(q), ref.generalized_gravity(q), rtol=0, atol=1e-9)
        np.testing.assert_allclose(emb.coriolis_matrix(q, qd), ref.coriolis_matrix(q, qd), rtol=0, atol=1e-9)
        np.testing.assert_allclose(emb.dccrba(q), ref.dccrba(q), rtol=0, atol=1e-9)
        np.testing.assert_allclose(emb.cmm_time_variation(q, qd), ref.cmm_time_variation(q, qd), rtol=0, atol=1e-9)
        # the reduced reference's hdot sources its bias from finite differences; the analytic pieces agree exactly
        np.testing.assert_allclose(emb.centroidal_momentum_time_variation(q, qd, qdd),
                                   np.asarray(ref.ccrba(q, qd)[0]) @ qdd + ref.cmm_time_variation(q, qd) @ qd, rtol=0, atol=1e-9)
        np.testing.assert_allclose(emb.centroidal_momentum_time_variation(q, qd, qdd), ref.centroidal_momentum_time_variation(q, qd, qdd), rtol=0, atol=1e-6)
        np.testing.assert_allclose(emb.nonlinear_effects(q, qd), ref.nonlinear_effects(q, qd), rtol=0, atol=1e-9)
        assert abs(emb.kinetic_energy(q, qd) - ref.kinetic_energy(q, qd)) < 1e-9
        assert abs(emb.potential_energy(q) - ref.potential_energy(q)) < 1e-9


_INERTIAL = ('<inertial><origin xyz="0.05 0.02 0.1"/><mass value="1.3"/>'
             '<inertia ixx="0.04" ixy="0.001" ixz="0.002" iyy="0.05" iyz="0.003" izz="0.03"/></inertial>')


def _joint(name, parent, child, axis, xyz, rpy, mimic_xml=""):
    return (f'<joint name="{name}" type="revolute"><parent link="{parent}"/><child link="{child}"/>'
            f'<axis xyz="{axis}"/><origin xyz="{xyz}" rpy="{rpy}"/>'
            '<limit lower="-3" upper="3" effort="10" velocity="10"/>' + mimic_xml + '</joint>')


def _poly_urdf(tmp_path):
    links = '<link name="base"/>' + "".join(f'<link name="l{k}">{_INERTIAL}</link>' for k in range(1, 5))
    joints = (_joint("j1", "base", "l1", "0 0 1", "0 0 0.1", "0 0 0")
              + _joint("j2", "l1", "l2", "0 1 0", "0.2 0 0.05", "0.1 0 0",
                       '<mimic joint="j1" polycoef="0.1 0.6 0.25 -0.1 0.04" q0="0.2"/>')
              + _joint("j3", "l2", "l3", "1 0 0", "0.15 0.05 0", "0 0.2 0")
              + _joint("j4", "l3", "l4", "0 1 0", "0.1 0 0.1", "0 0 0.3",
                       '<mimic joint="j3" polycoef="0 1.5 0 0.3"/>'))
    path = tmp_path / "poly_chain.urdf"
    path.write_text(f'<robot name="poly_chain">{links}{joints}</robot>')
    return path


def _plain_fd(f, x, h=1e-6):
    cols = []
    for k in range(len(x)):
        e = np.zeros(len(x)); e[k] = h
        cols.append((f(x + e) - f(x - e)) / (2 * h))
    return np.array(cols).T


def _chart_fd(ref, f, q, nv, h=1e-6):
    cols = []
    for k in range(nv):
        e = np.zeros(nv); e[k] = h
        cols.append((f(ref.integrate(q, e)) - f(ref.integrate(q, -e))) / (2 * h))
    return np.array(cols).T


@pytest.mark.parametrize("floating", [False, True], ids=["fixed", "floating"])
def test_polynomial_coupling_projection(tmp_path, floating):
    robot = _parse(_poly_urdf(tmp_path), floating)
    ref = RBDReference(robot)
    assert ref._embedding is not None
    emb = ref._embedding
    nv, nq = robot.get_num_vel(), robot.get_num_pos()
    assert nv == (2 if not floating else 8) and emb.nv_full == nv + 2
    rng = np.random.default_rng(3)
    for _ in range(3):
        y, yd, ydd = _random_state(robot, rng)
        # the map: G = d expand_q / dy in the chart, Gdot yd = d/dt (G yd) at fixed yd
        G = emb.G(y)
        for joint, fqi, fvi, lfq, lrq, lrv in emb.followers:
            ql = float(y[lrq])
            fd_alpha = (joint.coupling_value(ql + 1e-6) - joint.coupling_value(ql - 1e-6)) / 2e-6
            assert abs(G[fvi, lrv] - fd_alpha) < 1e-8
        # dG/dy yd checked through qd_full(y) = G(y) yd: d/dy (G yd) = (dG/dy) yd
        d_qd = _chart_fd(ref, lambda yy: emb.expand_qd(yy, yd), y, nv)
        expect = np.zeros((emb.nv_full, nv))
        for fvi, lrv, _, ald, _ in emb._alphas(y):
            expect[fvi, lrv] = ald * yd[lrv]
        np.testing.assert_allclose(d_qd, expect, rtol=0, atol=1e-6)
        # the velocity map is the derivative of the position map along the flow
        h = 1e-6
        qp = emb.expand_q(ref.integrate(y, h * yd)); qm = emb.expand_q(ref.integrate(y, -h * yd))
        qd_full = emb.expand_qd(y, yd)
        # compare the follower rows (vector-space joints), where expand_q is plain
        for _, fqi, fvi, _, _, _ in emb.followers:
            assert abs((qp[fqi] - qm[fqi]) / (2 * h) - qd_full[fvi]) < 1e-6
        # Gdot yd = d/dt (G yd) along the flow at fixed yd
        gp = emb.G(ref.integrate(y, h * yd)) @ yd; gm = emb.G(ref.integrate(y, -h * yd)) @ yd
        np.testing.assert_allclose((gp - gm) / (2 * h), emb.gdot_yd(y, yd), rtol=0, atol=1e-6)
        # dynamics: ID and FD are inverses, ABA = FD, M symmetric and the qdd-derivative of ID
        tau = ref.inverse_dynamics(y, yd, ydd)[0]
        np.testing.assert_allclose(ref.forward_dynamics(y, yd, tau), ydd, rtol=0, atol=1e-8)
        np.testing.assert_allclose(ref.aba(y, yd, tau), ydd, rtol=0, atol=1e-8)
        M = ref.crba(y)
        np.testing.assert_allclose(M, M.T, rtol=0, atol=1e-12)
        assert np.all(np.linalg.eigvalsh(M) > 0)
        np.testing.assert_allclose(_plain_fd(lambda a: ref.inverse_dynamics(y, yd, a)[0], ydd), M, rtol=0, atol=1e-6)
        np.testing.assert_allclose(ref.minv(y) @ M, np.eye(nv), rtol=0, atol=1e-9)
        # the Coriolis matrix: C yd + g is the bias and C + C^T is the time derivative of M
        C = ref.coriolis_matrix(y, yd)
        np.testing.assert_allclose(C @ yd + ref.generalized_gravity(y), ref.nonlinear_effects(y, yd), rtol=0, atol=1e-9)
        Mp = ref.crba(ref.integrate(y, h * yd)); Mm = ref.crba(ref.integrate(y, -h * yd))
        np.testing.assert_allclose(C + C.T, (Mp - Mm) / (2 * h), rtol=0, atol=2e-5)
        # the momentum matrix derivatives: the tensor against the chart, Adot along the flow, hdot = A ydd + Adot yd
        A = np.asarray(ref.ccrba(y, yd)[0], dtype=np.float64)
        dA = ref.dccrba(y)
        for i in range(nv):
            e = np.zeros(nv); e[i] = h
            fd = (np.asarray(ref.ccrba(ref.integrate(y, e), yd)[0]) - np.asarray(ref.ccrba(ref.integrate(y, -e), yd)[0])) / (2 * h)
            np.testing.assert_allclose(dA[:, :, i], fd, rtol=0, atol=2e-5)
        Ap = np.asarray(ref.ccrba(ref.integrate(y, h * yd), yd)[0]); Am = np.asarray(ref.ccrba(ref.integrate(y, -h * yd), yd)[0])
        Adot = ref.cmm_time_variation(y, yd)
        np.testing.assert_allclose(Adot, (Ap - Am) / (2 * h), rtol=0, atol=2e-5)
        np.testing.assert_allclose(np.einsum("abi,i->ab", dA, yd), Adot, rtol=0, atol=1e-9)
        hp = np.asarray(ref.ccrba(ref.integrate(y, h * yd), yd + h * ydd)[1]); hm = np.asarray(ref.ccrba(ref.integrate(y, -h * yd), yd - h * ydd)[1])
        np.testing.assert_allclose(ref.centroidal_momentum_time_variation(y, yd, ydd), (hp - hm) / (2 * h), rtol=0, atol=2e-5)
        # gradients against chart finite differences of the projected values
        dc = ref.inverse_dynamics_gradient(y, yd, ydd)
        np.testing.assert_allclose(dc[:, :nv], _chart_fd(ref, lambda yy: ref.inverse_dynamics(yy, yd, ydd)[0], y, nv), rtol=0, atol=2e-5)
        np.testing.assert_allclose(dc[:, nv:], _plain_fd(lambda v: ref.inverse_dynamics(y, v, ydd)[0], yd), rtol=0, atol=2e-5)
        u = rng.standard_normal(nv)
        dq, dqd = ref.forward_dynamics_gradient(y, yd, u)
        np.testing.assert_allclose(dq, _chart_fd(ref, lambda yy: ref.forward_dynamics(yy, yd, u), y, nv), rtol=0, atol=2e-5)
        np.testing.assert_allclose(dqd, _plain_fd(lambda v: ref.forward_dynamics(y, v, u), yd), rtol=0, atol=2e-5)
        # kinematics: the pose gradient against chart finite differences of the pose
        leaf = robot.get_joint_by_id(int(robot.get_leaf_nodes()[0])).get_name()
        dpose = np.asarray(ref.end_effector_pose_gradient(y, leaf), dtype=np.float64).reshape(-1, nv)
        fd = _chart_fd(ref, lambda yy: np.asarray(ref.end_effector_pose(yy, leaf), dtype=np.float64).reshape(-1), y, nv)
        np.testing.assert_allclose(dpose, fd, rtol=0, atol=2e-6)
        # v = J yd; a = d/dt v along the flow (classical in LWA)
        for rf in ("LOCAL", "WORLD", "LOCAL_WORLD_ALIGNED"):
            J = ref.frame_jacobian(y, leaf, rf)
            assert J.shape == (6, nv)
            np.testing.assert_allclose(ref.frame_velocity(y, yd, leaf, rf), J @ yd, rtol=0, atol=1e-10)
            vp = ref.frame_velocity(ref.integrate(y, h * yd), yd + h * ydd, leaf, rf)
            vm = ref.frame_velocity(ref.integrate(y, -h * yd), yd - h * ydd, leaf, rf)
            np.testing.assert_allclose((vp - vm) / (2 * h), ref.frame_acceleration(y, yd, ydd, leaf, rf), rtol=0, atol=2e-5)
            # Jdot = d/dt J along the flow; the kinematics derivatives against the chart
            Jp = ref.frame_jacobian(ref.integrate(y, h * yd), leaf, rf); Jm = ref.frame_jacobian(ref.integrate(y, -h * yd), leaf, rf)
            np.testing.assert_allclose((Jp - Jm) / (2 * h), ref.frame_jacobian_dot(y, yd, leaf, rf), rtol=0, atol=2e-5)
            dv_dq, dv_dqd = ref.frame_velocity_derivatives(y, yd, leaf, rf)
            np.testing.assert_allclose(dv_dq, _chart_fd(ref, lambda yy: ref.frame_velocity(yy, yd, leaf, rf), y, nv), rtol=0, atol=2e-5)
            np.testing.assert_allclose(dv_dqd, _plain_fd(lambda v: ref.frame_velocity(y, v, leaf, rf), yd), rtol=0, atol=2e-5)
            dv2, da_dq, da_dqd, da_dqdd = ref.frame_acceleration_derivatives(y, yd, ydd, leaf, rf)
            np.testing.assert_allclose(dv2, dv_dq, rtol=0, atol=1e-12)
            np.testing.assert_allclose(da_dq, _chart_fd(ref, lambda yy: ref.frame_acceleration(yy, yd, ydd, leaf, rf), y, nv), rtol=0, atol=5e-5)
            np.testing.assert_allclose(da_dqd, _plain_fd(lambda v: ref.frame_acceleration(y, v, ydd, leaf, rf), yd), rtol=0, atol=2e-5)
            np.testing.assert_allclose(da_dqdd, _plain_fd(lambda a: ref.frame_acceleration(y, yd, a, leaf, rf), ydd), rtol=0, atol=2e-5)
            if nv >= 6:   # the task inertia exists only with six independent coordinates upstream
                L = ref.osc_inertia(y, leaf, rf)
                np.testing.assert_allclose(L @ (J @ ref.minv(y) @ J.T), np.eye(6), rtol=0, atol=1e-8)
        # regressors: tau = Y pi with the robot's own inertial parameters; KE and PE likewise
        from RBDReference.tests.test_regressor_equivalence import _project_pi
        pi = _project_pi(type("M", (), {"robot": robot})())
        tau = ref.inverse_dynamics(y, yd, ydd)[0]
        np.testing.assert_allclose(np.asarray(ref.inverse_dynamics_regressor(y, yd, ydd)) @ pi, tau, rtol=0, atol=1e-8)
        np.testing.assert_allclose(np.asarray(ref.kinetic_energy_regressor(y, yd)).reshape(-1) @ pi, 0.5 * yd @ ref.crba(y) @ yd, rtol=0, atol=1e-8)
        np.testing.assert_allclose(np.asarray(ref.potential_energy_regressor(y)).reshape(-1) @ pi, ref.potential_energy(y), rtol=0, atol=1e-8)
        np.testing.assert_allclose(ref.forward_dynamics_parameter_gradient(y, yd, u), -ref.minv(y) @ np.asarray(ref.inverse_dynamics_regressor(y, yd, ref.forward_dynamics(y, yd, u))), rtol=0, atol=1e-8)
        # centroidal: the CoM Jacobian against the chart, h = A yd, gravity and bias against the inverse dynamics
        np.testing.assert_allclose(ref.jacobian_com(y), _chart_fd(ref, lambda yy: np.asarray(ref.com(yy)).reshape(-1), y, nv), rtol=0, atol=2e-6)
        A, hvec = ref.ccrba(y, yd)
        np.testing.assert_allclose(np.asarray(A) @ yd, np.asarray(hvec).reshape(-1), rtol=0, atol=1e-10)
        np.testing.assert_allclose(ref.generalized_gravity(y), ref.inverse_dynamics(y, np.zeros(nv), np.zeros(nv))[0], rtol=0, atol=1e-10)
        np.testing.assert_allclose(ref.nonlinear_effects(y, yd), ref.inverse_dynamics(y, yd, np.zeros(nv))[0], rtol=0, atol=1e-10)
        assert abs(ref.kinetic_energy(y, yd) - 0.5 * yd @ ref.crba(y) @ yd) < 1e-9
        assert abs(ref.mechanical_energy(y, yd) - ref.kinetic_energy(y, yd) - ref.potential_energy(y)) < 1e-9
