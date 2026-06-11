"""End-to-end verification of SPHERICAL (ball) joints vs Pinocchio.

A URDF ``spherical`` joint is a 3-DOF rotation-only joint parameterized by a
UNIT QUATERNION: NV=3 (body-frame angular velocity), NQ=4 (xyzw quaternion).
It is the only genuinely-new MANIFOLD joint type -- its config-space retract is
an SO(3) quaternion exponential, NOT a vector add. Pinocchio's native oracle is
``JointModelSpherical`` (the quaternion variant; NOT ``JointModelSphericalZYX``).

Conventions matched to Pinocchio:
  - quaternion order: xyzw (w last); neutral = [0,0,0,1].
  - spherical velocity: body-frame angular velocity omega.

urdfdom cannot parse ``type="spherical"``, so the Pinocchio oracle models are
built PROGRAMMATICALLY (JointModelSpherical / JointModelRX/RZ) mirroring the
exact inertias and joint placements of the GRiD fixtures.

This module pins, to ~1e-9 over random unit-quaternion configs:
  - inverse_dynamics (RNEA), crba (M), aba (forward dynamics) vs Pinocchio;
  - integrate / dIntegrate(ARG0/q, ARG1/v) vs pin.integrate / pin.dIntegrate;
  - unit-norm preservation of the integrated quaternion;
  - the mixed-chain NQ!=NV q/v index maps (revolute + spherical + revolute).
"""
import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")

from URDFParser import URDFParser
from RBDReference import RBDReference
from RBDReference.tests.comparators import assert_close

FIX = "URDFParser/tests/fixtures/"

pytestmark = [
    pytest.mark.pinocchio_equivalence,
    pytest.mark.developer_only,
]

# Per-link inertials, copied from the fixtures (mass, com xyz, Ixx..Izz @ com).
# Every inertial/joint origin in the fixtures has identity rpy, so com == the
# inertial <origin> xyz and joint placement == a pure translation.
_LINKS = {
    "link1": (1.3, (0.05, 0.02, 0.10), (0.05, 0.01, 0.005, 0.06, 0.002, 0.04)),
    "link2": (0.9, (0.04, -0.03, 0.08), (0.04, 0.003, 0.001, 0.05, 0.004, 0.03)),
    "link3": (0.6, (0.0, 0.0, 0.06), (0.02, 0.0, 0.0, 0.02, 0.001, 0.015)),
}


def _inertia(name):
    m, com, (ixx, ixy, ixz, iyy, iyz, izz) = _LINKS[name]
    I = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    return pin.Inertia(m, np.array(com, dtype=float), I)


def _placement(xyz):
    return pin.SE3(np.eye(3), np.array(xyz, dtype=float))


def _build_pin_spherical():
    """spherical_arm.urdf: base -> [spherical] link1 -> [revolute Z] link2."""
    model = pin.Model()
    j1 = model.addJoint(0, pin.JointModelSpherical(), _placement((0, 0, 0.15)), "joint_1")
    model.appendBodyToJoint(j1, _inertia("link1"), pin.SE3.Identity())
    j2 = model.addJoint(j1, pin.JointModelRZ(), _placement((0, 0, 0.20)), "joint_2")
    model.appendBodyToJoint(j2, _inertia("link2"), pin.SE3.Identity())
    return model


def _build_pin_mixed():
    """mixed_spherical_arm.urdf: base -> [rev Z] link1 -> [spherical] link2 ->
    [rev X] link3."""
    model = pin.Model()
    j1 = model.addJoint(0, pin.JointModelRZ(), _placement((0, 0, 0.15)), "joint_1")
    model.appendBodyToJoint(j1, _inertia("link1"), pin.SE3.Identity())
    j2 = model.addJoint(j1, pin.JointModelSpherical(), _placement((0, 0, 0.20)), "joint_2")
    model.appendBodyToJoint(j2, _inertia("link2"), pin.SE3.Identity())
    j3 = model.addJoint(j2, pin.JointModelRX(), _placement((0, 0, 0.18)), "joint_3")
    model.appendBodyToJoint(j3, _inertia("link3"), pin.SE3.Identity())
    return model


def _grid(fixture):
    return URDFParser().parse(FIX + fixture, floating_base=False)


def _idx_list(index):
    return list(index) if isinstance(index, (list, tuple, np.ndarray)) else [index]


def _q_spherical(rng):
    quat = rng.standard_normal(4); quat /= np.linalg.norm(quat)
    return np.concatenate([quat, rng.uniform(-1, 1, 1)])  # [quat(4), rev(1)]


def _q_mixed(rng):
    quat = rng.standard_normal(4); quat /= np.linalg.norm(quat)
    # GRiD == pin order: rev_z(1), spherical quat(4), rev_x(1)
    return np.concatenate([rng.uniform(-1, 1, 1), quat, rng.uniform(-1, 1, 1)])


CASES = [
    pytest.param("spherical_arm.urdf", _build_pin_spherical, _q_spherical, 5, 4,
                 id="spherical_arm"),
    pytest.param("mixed_spherical_arm.urdf", _build_pin_mixed, _q_mixed, 6, 5,
                 id="mixed_spherical_arm"),
]


# ---------------------------------------------------------------------------
# FLOATING ROOT + spherical mid-chain (Tier-C generalization).
# mixed_spherical_arm.urdf parsed floating_base=True is a free-flyer ROOT
# (6-DOF, NQ=7) over [rev Z](1) -> [spherical](3, NQ=4) -> [rev X](1):
#   total NQ=13, NV=11, with TWO NQ!=NV joints (the root quat + the spherical
#   quat). Validates that the unified block-CRBA handles a floating root AND a
#   multi-DOF mid-chain joint simultaneously vs pin free-flyer+JointModelSpherical.
# ---------------------------------------------------------------------------
def _build_pin_floating_mixed():
    """free-flyer ROOT -> [rev Z] link1 -> [spherical] link2 -> [rev X] link3.

    The free-flyer body carries the GRiD `base` link inertia (mass 1.0, com at
    origin, rotational diag 0.1 -- mirrors mixed_spherical_arm.urdf's base).
    """
    model = pin.Model()
    j0 = model.addJoint(0, pin.JointModelFreeFlyer(), pin.SE3.Identity(), "root")
    model.appendBodyToJoint(
        j0, pin.Inertia(1.0, np.zeros(3), np.diag([0.1, 0.1, 0.1])), pin.SE3.Identity())
    j1 = model.addJoint(j0, pin.JointModelRZ(), _placement((0, 0, 0.15)), "joint_1")
    model.appendBodyToJoint(j1, _inertia("link1"), pin.SE3.Identity())
    j2 = model.addJoint(j1, pin.JointModelSpherical(), _placement((0, 0, 0.20)), "joint_2")
    model.appendBodyToJoint(j2, _inertia("link2"), pin.SE3.Identity())
    j3 = model.addJoint(j2, pin.JointModelRX(), _placement((0, 0, 0.18)), "joint_3")
    model.appendBodyToJoint(j3, _inertia("link3"), pin.SE3.Identity())
    return model


def _q_floating_mixed(rng):
    """GRiD == pin floating order: pos(3), root_quat(4), rev_z(1),
    spherical_quat(4), rev_x(1)."""
    pos = rng.uniform(-1, 1, 3)
    rquat = rng.standard_normal(4); rquat /= np.linalg.norm(rquat)
    squat = rng.standard_normal(4); squat /= np.linalg.norm(squat)
    return np.concatenate(
        [pos, rquat, rng.uniform(-1, 1, 1), squat, rng.uniform(-1, 1, 1)])


def _grid_floating(fixture):
    return URDFParser().parse(FIX + fixture, floating_base=True)


def test_floating_spherical_index_maps():
    """free-flyer root + spherical mid-chain: total NQ=13/NV=11, the root owns
    7q/6v and the spherical owns 4q/3v (TWO NQ!=NV joints in one chain)."""
    robot = _grid_floating("mixed_spherical_arm.urdf")
    pmodel = _build_pin_floating_mixed()
    assert robot.get_num_pos() == pmodel.nq == 13
    assert robot.get_num_vel() == pmodel.nv == 11
    assert _idx_list(robot.get_joint_index_q(0)) == [0, 1, 2, 3, 4, 5, 6]
    assert _idx_list(robot.get_joint_index_v(0)) == [0, 1, 2, 3, 4, 5]
    for jid in range(robot.get_num_bodies()):
        if getattr(robot.get_joint_by_id(jid), "jtype", None) == "spherical":
            assert len(_idx_list(robot.get_joint_index_q(jid))) == 4
            assert len(_idx_list(robot.get_joint_index_v(jid))) == 3


def test_floating_spherical_dynamics_match_pinocchio():
    """RNEA / CRBA / ABA for a free-flyer root + spherical mid-chain match
    Pinocchio's free-flyer + JointModelSpherical model to ~1e-9."""
    robot = _grid_floating("mixed_spherical_arm.urdf")
    ref = RBDReference(robot)
    pmodel = _build_pin_floating_mixed()
    pdata = pmodel.createData()
    nv = pmodel.nv
    rng = np.random.default_rng(3)
    for _ in range(25):
        q = _q_floating_mixed(rng)
        qd = rng.uniform(-1, 1, nv)
        qdd = rng.uniform(-1, 1, nv)
        tau = rng.uniform(-1, 1, nv)

        assert_close(ref.inverse_dynamics(q, qd, qdd, normalize_input=False)[0],
                     pin.rnea(pmodel, pdata, q, qd, qdd),
                     algorithm="inverse_dynamics", robot_id="floating_mixed_spherical")

        M_p = pin.crba(pmodel, pdata, q)
        M_p = np.triu(M_p) + np.triu(M_p, 1).T
        assert_close(ref.crba(q, normalize_input=False), M_p,
                     algorithm="minv", robot_id="floating_mixed_spherical")

        assert_close(ref.aba(q, qd, tau, normalize_input=False),
                     pin.aba(pmodel, pdata, q, qd, tau),
                     algorithm="forward_dynamics", robot_id="floating_mixed_spherical")


def test_floating_spherical_integrate_match_pinocchio():
    """integrate / dIntegrate(ARG0, ARG1) for the free-flyer root + spherical
    mid-chain match pin; both manifold quaternions stay unit-norm."""
    robot = _grid_floating("mixed_spherical_arm.urdf")
    ref = RBDReference(robot)
    pmodel = _build_pin_floating_mixed()
    nv = pmodel.nv
    rng = np.random.default_rng(5)
    for _ in range(25):
        q = _q_floating_mixed(rng)
        v_dt = rng.uniform(-0.4, 0.4, nv)

        qn_g = ref.integrate(q, v_dt)
        np.testing.assert_allclose(qn_g, pin.integrate(pmodel, q, v_dt), atol=1e-10)
        # root quat (q[3:7]) and the spherical quat both unit-norm.
        assert abs(np.linalg.norm(qn_g[3:7]) - 1.0) < 1e-12
        for jid in range(robot.get_num_bodies()):
            if getattr(robot.get_joint_by_id(jid), "jtype", None) == "spherical":
                iq = _idx_list(robot.get_joint_index_q(jid))
                assert abs(np.linalg.norm(qn_g[iq]) - 1.0) < 1e-12

        np.testing.assert_allclose(
            ref.dIntegrate(q, v_dt, "q"),
            pin.dIntegrate(pmodel, q, v_dt, pin.ArgumentPosition.ARG0), atol=1e-10)
        np.testing.assert_allclose(
            ref.dIntegrate(q, v_dt, "v"),
            pin.dIntegrate(pmodel, q, v_dt, pin.ArgumentPosition.ARG1), atol=1e-10)


@pytest.mark.parametrize(("fixture", "build_pin", "random_q", "nq", "nv"), CASES)
def test_spherical_index_maps(fixture, build_pin, random_q, nq, nv):
    """NQ==4/NV==3 per spherical joint; total NQ/NV and the q/v block maps match
    the Pinocchio model (the NQ!=NV bookkeeping across the mixed chain)."""
    robot = _grid(fixture)
    pmodel = build_pin()
    assert robot.get_num_pos() == pmodel.nq == nq
    assert robot.get_num_vel() == pmodel.nv == nv
    for jid in range(robot.get_num_bodies()):
        j = robot.get_joint_by_id(jid)
        if getattr(j, "jtype", None) == "spherical":
            assert j.get_num_dof() == 3
            assert len(_idx_list(robot.get_joint_index_q(jid))) == 4
            assert len(_idx_list(robot.get_joint_index_v(jid))) == 3


@pytest.mark.parametrize(("fixture", "build_pin", "random_q", "nq", "nv"), CASES)
def test_spherical_dynamics_match_pinocchio(fixture, build_pin, random_q, nq, nv):
    """RNEA / CRBA / ABA match Pinocchio's native JointModelSpherical to ~1e-9
    over random unit-quaternion configs + random velocities."""
    robot = _grid(fixture)
    ref = RBDReference(robot)
    pmodel = build_pin()
    pdata = pmodel.createData()
    rng = np.random.default_rng(0)
    for _ in range(25):
        q = random_q(rng)
        qd = rng.uniform(-1, 1, nv)
        qdd = rng.uniform(-1, 1, nv)
        tau = rng.uniform(-1, 1, nv)

        assert_close(ref.inverse_dynamics(q, qd, qdd, normalize_input=False)[0],
                     pin.rnea(pmodel, pdata, q, qd, qdd),
                     algorithm="inverse_dynamics", robot_id=fixture)

        M_p = pin.crba(pmodel, pdata, q)
        M_p = np.triu(M_p) + np.triu(M_p, 1).T  # pin fills upper triangle only
        assert_close(ref.crba(q, normalize_input=False), M_p,
                     algorithm="minv", robot_id=fixture)

        assert_close(ref.aba(q, qd, tau, normalize_input=False),
                     pin.aba(pmodel, pdata, q, qd, tau),
                     algorithm="forward_dynamics", robot_id=fixture)


def _split_id_gradient(out, nv):
    """RBDReference.inverse_dynamics_gradient returns hstack([dc_dq | dc_dqd])
    (nv x 2nv); split it into the two nv x nv tangent-space blocks."""
    return out[:, :nv], out[:, nv:]


@pytest.mark.parametrize(("fixture", "build_pin", "random_q", "nq", "nv"), CASES)
def test_spherical_id_gradient_match_pinocchio(fixture, build_pin, random_q, nq, nv):
    """inverse_dynamics_gradient (dRNEA/dq, dRNEA/dqd) matches Pinocchio's
    computeRNEADerivatives (dtau_dq, dtau_dv) for a JointModelSpherical model,
    covering BOTH the root-spherical (spherical_arm) and the mid-chain-spherical
    (mixed_spherical_arm, the real NQ!=NV multi-column test) fixtures to ~1e-9.

    dtau_dq / dtau_dv live in the nv-tangent (the SO(3) tangent for the spherical
    block), so they compare directly against GRiD's nv x nv dc_dq / dc_dqd."""
    robot = _grid(fixture)
    ref = RBDReference(robot)
    pmodel = build_pin()
    pdata = pmodel.createData()
    rng = np.random.default_rng(0)
    for _ in range(25):
        q = random_q(rng)
        qd = rng.uniform(-1, 1, nv)
        qdd = rng.uniform(-1, 1, nv)

        dc_dq, dc_dqd = _split_id_gradient(
            ref.inverse_dynamics_gradient(q, qd, qdd, normalize_input=False), nv)
        pin.computeRNEADerivatives(pmodel, pdata, q, qd, qdd)

        assert_close(dc_dq, pdata.dtau_dq,
                     algorithm="inverse_dynamics", robot_id=fixture)
        assert_close(dc_dqd, pdata.dtau_dv,
                     algorithm="inverse_dynamics", robot_id=fixture)


def test_floating_spherical_id_gradient_match_pinocchio():
    """inverse_dynamics_gradient for a free-flyer ROOT + spherical mid-chain
    (nv=11, TWO NQ!=NV joints: the root quat AND the spherical quat) matches
    Pinocchio's free-flyer + JointModelSpherical computeRNEADerivatives. This
    exercises a multi-column da/df term on a non-root body simultaneously with
    the floating root's own k=6 column block."""
    robot = _grid_floating("mixed_spherical_arm.urdf")
    ref = RBDReference(robot)
    pmodel = _build_pin_floating_mixed()
    pdata = pmodel.createData()
    nv = pmodel.nv
    rng = np.random.default_rng(3)
    for _ in range(25):
        q = _q_floating_mixed(rng)
        qd = rng.uniform(-1, 1, nv)
        qdd = rng.uniform(-1, 1, nv)

        dc_dq, dc_dqd = _split_id_gradient(
            ref.inverse_dynamics_gradient(q, qd, qdd, normalize_input=False), nv)
        pin.computeRNEADerivatives(pmodel, pdata, q, qd, qdd)

        assert_close(dc_dq, pdata.dtau_dq,
                     algorithm="inverse_dynamics", robot_id="floating_mixed_spherical")
        assert_close(dc_dqd, pdata.dtau_dv,
                     algorithm="inverse_dynamics", robot_id="floating_mixed_spherical")


@pytest.mark.parametrize(("fixture", "build_pin", "random_q", "nq", "nv"), CASES)
def test_spherical_integrate_match_pinocchio(fixture, build_pin, random_q, nq, nv):
    """integrate / dIntegrate(ARG0, ARG1) match pin.integrate / pin.dIntegrate;
    the integrated spherical quaternion stays unit-norm."""
    robot = _grid(fixture)
    ref = RBDReference(robot)
    pmodel = build_pin()
    rng = np.random.default_rng(7)
    for _ in range(25):
        q = random_q(rng)
        v_dt = rng.uniform(-0.4, 0.4, nv)

        qn_g = ref.integrate(q, v_dt)
        np.testing.assert_allclose(qn_g, pin.integrate(pmodel, q, v_dt), atol=1e-10)

        for jid in range(robot.get_num_bodies()):
            if getattr(robot.get_joint_by_id(jid), "jtype", None) == "spherical":
                iq = _idx_list(robot.get_joint_index_q(jid))
                assert abs(np.linalg.norm(qn_g[iq]) - 1.0) < 1e-12

        np.testing.assert_allclose(
            ref.dIntegrate(q, v_dt, "q"),
            pin.dIntegrate(pmodel, q, v_dt, pin.ArgumentPosition.ARG0), atol=1e-10)
        np.testing.assert_allclose(
            ref.dIntegrate(q, v_dt, "v"),
            pin.dIntegrate(pmodel, q, v_dt, pin.ArgumentPosition.ARG1), atol=1e-10)


@pytest.mark.parametrize(("fixture", "build_pin", "random_q", "nq", "nv"), CASES)
def test_spherical_dintegrate_matches_fd(fixture, build_pin, random_q, nq, nv):
    """dIntegrate(...,'v') matches a finite-difference of integrate in the
    SO(3)/scalar tangent (independent of the Pinocchio closed form)."""
    robot = _grid(fixture)
    ref = RBDReference(robot)
    rng = np.random.default_rng(11)
    eps = 1e-6
    for _ in range(10):
        q = random_q(rng)
        v_dt = rng.uniform(-0.3, 0.3, nv)
        J = ref.dIntegrate(q, v_dt, "v")
        q0 = ref.integrate(q, v_dt)
        Jfd = np.zeros((nv, nv))
        for k in range(nv):
            dv = np.zeros(nv); dv[k] = eps
            Jfd[:, k] = _tangent_diff(robot, q0, ref.integrate(q, v_dt + dv)) / eps
        np.testing.assert_allclose(J, Jfd, atol=1e-5)


def _tangent_diff(robot, q0, qp):
    """log(q0^-1 ⊕ qp) in the nv tangent (revolute = scalar diff; spherical =
    SO(3) log of the relative xyzw quaternion)."""
    out = []
    for jid in range(robot.get_num_bodies()):
        j = robot.get_joint_by_id(jid)
        iq = _idx_list(robot.get_joint_index_q(jid))
        if getattr(j, "jtype", None) == "spherical":
            a, b = q0[iq], qp[iq]
            ainv = np.array([-a[0], -a[1], -a[2], a[3]])
            out.extend(_so3_log_xyzw(RBDReference._quat_mul_xyzw(ainv, b)))
        else:
            out.append(float(qp[iq[0]] - q0[iq[0]]))
    return np.array(out)


def _so3_log_xyzw(qr):
    qr = qr / np.linalg.norm(qr)
    if qr[3] < 0:
        qr = -qr
    v, w = qr[:3], qr[3]
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return 2.0 * v
    return (2.0 * np.arctan2(nv, w) / nv) * v
