"""End-to-end verification of HELICAL / SCREW joints vs Pinocchio.

A helical (screw) joint is 1-DOF (NQ=NV=1) coupling rotation about an axis with
translation ALONG it by a `pitch` (meters / radian): a single scalar theta
drives both. Its motion subspace is a SINGLE column with a COUPLED linear part:

    S = [axis_unit ; pitch * axis_unit]

Even a cardinal axis gives >=2 nonzero entries, so a helical joint is
intrinsically NON-cardinal (Tier B): `S_is_cardinal_by_id` returns False and
`robot_has_skew_axis()` trips, routing every algorithm through the dense
6-vector Tier-B path that joint-1b built/validated. There is NO new algorithm
code -- the only new thing is the coupled linear rows of S and the screw
transform exp(theta * S^).

Pitch convention (extension, since URDF has no native helical type):
  <axis xyz="..." pitch="P"/>  with translation = P * angle, matching
  pinocchio's JointModelHelical pitch (meters / radian).

urdfdom has no `helical`/`screw` type, so the Pinocchio oracle is built
PROGRAMMATICALLY with JointModelHelicalUnaligned(axis, pitch) (and JointModelRX
for the plain revolute), mirroring the helical_arm.urdf fixture exactly.

This module pins, to ~1e-9 over random configs:
  - inverse_dynamics (RNEA), crba (M), forward_dynamics / aba vs Pinocchio
    native JointModelHelicalUnaligned (cardinal-axis AND skew-axis screws);
  - the same three quantities vs the RBDReference numpy path consuming the
    dense coupled 6-vector S (self-consistency of the Tier-B machinery);
  - forward kinematics (per-joint homogeneous transform chain) vs pin.oMi.
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

# Per-link inertials, copied verbatim from helical_arm.urdf (identity rpy on
# every origin, so com == inertial <origin> xyz and placement == translation).
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


def _build_pin_helical():
    """helical_arm.urdf: base -> [helical Z, pitch 0.05] link1
    -> [revolute X] link2 -> [screw (1,1,1), pitch -0.03] link3."""
    model = pin.Model()
    ax1 = np.array([0.0, 0.0, 1.0])
    j1 = model.addJoint(0, pin.JointModelHelicalUnaligned(ax1, 0.05),
                        _placement((0, 0, 0.15)), "joint_1")
    model.appendBodyToJoint(j1, _inertia("link1"), pin.SE3.Identity())
    j2 = model.addJoint(j1, pin.JointModelRX(), _placement((0, 0, 0.20)), "joint_2")
    model.appendBodyToJoint(j2, _inertia("link2"), pin.SE3.Identity())
    ax3 = np.array([1.0, 1.0, 1.0]); ax3 /= np.linalg.norm(ax3)
    j3 = model.addJoint(j2, pin.JointModelHelicalUnaligned(ax3, -0.03),
                        _placement((0, 0, 0.18)), "joint_3")
    model.appendBodyToJoint(j3, _inertia("link3"), pin.SE3.Identity())
    return model


def _grid():
    return URDFParser().parse(FIX + "helical_arm.urdf", floating_base=False)


def test_helical_S_is_coupled_and_non_cardinal():
    """Each helical joint exposes the COUPLED single-column S = [axis; pitch*axis]
    and is classified NON-cardinal so the Tier-B dense path is taken."""
    robot = _grid()
    assert robot.get_num_joints() == 3
    assert robot.get_num_pos() == robot.get_num_vel() == 3  # NQ==NV==1 per joint

    z = 1.0
    np.testing.assert_allclose(robot.get_S_by_id(0),
                               [0, 0, z, 0, 0, 0.05 * z], atol=1e-12)  # helical Z
    np.testing.assert_allclose(robot.get_S_by_id(1),
                               [1, 0, 0, 0, 0, 0], atol=1e-12)         # revolute X
    u = 1.0 / np.sqrt(3.0)
    np.testing.assert_allclose(robot.get_S_by_id(2),
                               [u, u, u, -0.03 * u, -0.03 * u, -0.03 * u], atol=1e-12)

    assert not robot.S_is_cardinal_by_id(0)
    assert robot.S_is_cardinal_by_id(1)        # the plain revolute stays Tier A
    assert not robot.S_is_cardinal_by_id(2)
    assert robot.robot_has_skew_axis() is True  # -> codegen emits Tier-B machinery


def test_helical_dynamics_match_pinocchio():
    """RNEA / CRBA / ABA match Pinocchio's native JointModelHelicalUnaligned to
    ~1e-9 over random configs (cardinal-axis AND skew-axis screws)."""
    robot = _grid()
    ref = RBDReference(robot)
    pmodel = _build_pin_helical()
    pdata = pmodel.createData()
    rng = np.random.default_rng(0)
    for _ in range(40):
        q = rng.uniform(-2, 2, 3)
        qd = rng.uniform(-1, 1, 3)
        qdd = rng.uniform(-1, 1, 3)
        tau = rng.uniform(-1, 1, 3)

        assert_close(ref.inverse_dynamics(q, qd, qdd, normalize_input=False)[0],
                     pin.rnea(pmodel, pdata, q, qd, qdd),
                     algorithm="inverse_dynamics", robot_id="helical_arm")

        M_p = pin.crba(pmodel, pdata, q)
        M_p = np.triu(M_p) + np.triu(M_p, 1).T  # pin fills upper triangle only
        assert_close(ref.crba(q, normalize_input=False), M_p,
                     algorithm="minv", robot_id="helical_arm")

        assert_close(ref.aba(q, qd, tau, normalize_input=False),
                     pin.aba(pmodel, pdata, q, qd, tau),
                     algorithm="forward_dynamics", robot_id="helical_arm")


def test_helical_round_trip_id_fd():
    """forward_dynamics(inverse_dynamics) is the identity: the screw transform +
    coupled S are self-consistent through the RBDReference numpy Tier-B path."""
    robot = _grid()
    ref = RBDReference(robot)
    rng = np.random.default_rng(5)
    for _ in range(20):
        q = rng.uniform(-2, 2, 3)
        qd = rng.uniform(-1, 1, 3)
        qdd = rng.uniform(-1, 1, 3)
        tau = ref.inverse_dynamics(q, qd, qdd, normalize_input=False)[0]
        qdd_back = ref.aba(q, qd, tau, normalize_input=False)
        np.testing.assert_allclose(qdd_back, qdd, atol=1e-9)


def test_helical_forward_kinematics_match_pinocchio():
    """The screw homogeneous transform chain reproduces pin.oMi (body->world),
    confirming the EE/world kinematics of the coupled rotation+translation."""
    robot = _grid()
    pmodel = _build_pin_helical()
    pdata = pmodel.createData()
    rng = np.random.default_rng(3)
    for _ in range(20):
        q = rng.uniform(-2, 2, 3)
        pin.forwardKinematics(pmodel, pdata, q)
        T = np.eye(4)
        for jid in range(robot.get_num_joints()):
            f = robot.get_joint_by_id(jid).get_transformation_matrix_hom_function()
            T = T @ np.array(f(q[jid]), dtype=float)
            np.testing.assert_allclose(T, pdata.oMi[jid + 1].homogeneous, atol=1e-10)
