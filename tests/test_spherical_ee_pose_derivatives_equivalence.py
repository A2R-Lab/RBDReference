"""SPHERICAL (ball) joint EE-pose KINEMATICS + tangent-space derivatives.

Companion to test_spherical_joint_equivalence.py (which pins the DYNAMICS /
integrate side): this module pins the HOMOGENEOUS-transform kinematics chain and
its first/second tangent-space derivatives for the spherical fixtures.

Background (2026-08-02): the spherical (and floating/planar) hom transform used
to compose ``exp(q) @ X_origin`` — the ball rotated about its PARENT's origin
instead of its own anchor — so the EE world POSITION diverged from Pinocchio
whenever the ball joint's <origin xyz> was nonzero (rotations agreed). The
spatial ``Xmat_sp`` (Featherstone ``X_free @ X_origin``, parent->child) was
always correct, which is why the dynamics suite matched Pinocchio while the
hom-consuming kinematics did not. Fixed to the forward URDF/pinocchio
convention ``X_origin @ exp(q)`` (placement-then-motion); this module is the
regression gate.

Conventions pinned here (they define the CUDA equivalence contract too):
  - spherical tangent = BODY-frame angular velocity omega (pinocchio
    JointModelSpherical v-ordering); retract q_new = q ⊗ exp(½·omega_dt).
  - a spherical joint contributes THREE consecutive gradient/hessian columns at
    its ``get_joint_index_v`` slots: column k is d/dv_k with world axis
    R_world(joint) @ e_k (R_world INCLUDES the joint's own quaternion rotation
    — the local/right-tangent convention) and lever arm p_ee - p_joint.
  - intra-joint second derivative uses the SYMMETRIZED SO(3) exp curvature
    d²R/dv_k dv_l = ½([e_k]x [e_l]x + [e_l]x [e_k]x) in the joint frame,
    matching ``B_for_dofpair`` and the CUDA same-joint rev-rev pair block.

Gates:
  - end_effector_pose (position + world R) vs Pinocchio forwardKinematics;
  - frame_jacobian (LWA / WORLD / LOCAL) vs pin.getJointJacobian;
  - end_effector_pose_gradient vs pin (rows 0..2 = LWA linear Jacobian; rows
    3..5 = E(rpy)^-1 @ LWA angular Jacobian) AND vs central differences ON THE
    QUATERNION MANIFOLD (perturb via ref.integrate = q ⊗ exp(½·eps·e_k));
  - end_effector_pose_hessian_analytic vs the FD-of-gradient-on-manifold oracle
    (end_effector_pose_hessian) and vs direct second differences of the pose.
"""
import numpy as np
import pytest

pin = pytest.importorskip("pinocchio")

from URDFParser import URDFParser
from RBDReference import RBDReference

import URDFParser as _urdfparser_pkg  # the package dir is the repo root
FIX = str(__import__("pathlib").Path(_urdfparser_pkg.__file__).resolve().parent / "tests" / "fixtures") + "/"

pytestmark = [
    pytest.mark.pinocchio_equivalence,
    pytest.mark.developer_only,
]


def _placement(xyz):
    return pin.SE3(np.eye(3), np.array(xyz, dtype=float))


def _build_pin_spherical():
    model = pin.Model()
    j1 = model.addJoint(0, pin.JointModelSpherical(), _placement((0, 0, 0.15)), "joint_1")
    j2 = model.addJoint(j1, pin.JointModelRZ(), _placement((0, 0, 0.20)), "joint_2")
    return model, j2


def _build_pin_mixed():
    model = pin.Model()
    j1 = model.addJoint(0, pin.JointModelRZ(), _placement((0, 0, 0.15)), "joint_1")
    j2 = model.addJoint(j1, pin.JointModelSpherical(), _placement((0, 0, 0.20)), "joint_2")
    j3 = model.addJoint(j2, pin.JointModelRX(), _placement((0, 0, 0.18)), "joint_3")
    return model, j3


def _grid(fixture):
    return URDFParser().parse(FIX + fixture, floating_base=False)


def _q_spherical(rng):
    quat = rng.standard_normal(4); quat /= np.linalg.norm(quat)
    return np.concatenate([quat, rng.uniform(-1, 1, 1)])


def _q_mixed(rng):
    quat = rng.standard_normal(4); quat /= np.linalg.norm(quat)
    return np.concatenate([rng.uniform(-1, 1, 1), quat, rng.uniform(-1, 1, 1)])


CASES = [
    pytest.param("spherical_arm.urdf", _build_pin_spherical, _q_spherical, 5, 4,
                 id="spherical_arm"),
    pytest.param("mixed_spherical_arm.urdf", _build_pin_mixed, _q_mixed, 6, 5,
                 id="mixed_spherical_arm"),
]


def _near_gimbal(R):
    """rpy is ill-conditioned as |pitch| -> pi/2 (sqrt(R22^2+R21^2) -> 0)."""
    return np.sqrt(R[2, 2] ** 2 + R[2, 1] ** 2) < 0.2


def _Einv_from_R(R):
    roll = np.arctan2(R[2, 1], R[2, 2])
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[2, 2] ** 2 + R[2, 1] ** 2))
    yaw = np.arctan2(R[1, 0], R[0, 0])
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    E = np.array([[cy * cp, -sy, 0.0],
                  [sy * cp,  cy, 0.0],
                  [-sp,     0.0, 1.0]])
    return np.linalg.inv(E)


@pytest.mark.parametrize("fixture,build_pin,random_q,nq,nv", CASES)
def test_spherical_ee_pose_matches_pinocchio(fixture, build_pin, random_q, nq, nv):
    """World placement of the leaf joint matches pin.forwardKinematics exactly
    (position + rotation; rpy compared through the rotation, not the angles)."""
    robot = _grid(fixture)
    ref = RBDReference(robot)
    model, leaf = build_pin()
    data = model.createData()
    rng = np.random.default_rng(3)
    leaf_jid = robot.get_leaf_nodes()[0]
    for _ in range(10):
        q = random_q(rng)
        pin.forwardKinematics(model, data, q)
        # world transform via the oracle's own hom chain
        Xw = np.eye(4)
        chain = sorted(robot.get_ancestors_by_id(leaf_jid)) + [leaf_jid]
        for jid in chain:
            iq = robot.get_joint_index_q(jid)
            iq = list(iq) if isinstance(iq, (list, tuple, np.ndarray)) else iq
            # multi-DOF joints take their q-block; single-DOF a SCALAR theta
            Xw = Xw @ np.asarray(robot.get_Xmat_hom_Func_by_id(jid)(q[iq]))
        np.testing.assert_allclose(Xw[:3, :3], data.oMi[leaf].rotation, atol=1e-12)
        np.testing.assert_allclose(Xw[:3, 3], data.oMi[leaf].translation, atol=1e-12)
        # and the [xyz] rows of end_effector_pose agree
        pose = np.asarray(ref.end_effector_pose(q)[0]).reshape(-1)
        np.testing.assert_allclose(pose[:3], data.oMi[leaf].translation, atol=1e-12)


@pytest.mark.parametrize("fixture,build_pin,random_q,nq,nv", CASES)
def test_spherical_frame_jacobian_matches_pinocchio(fixture, build_pin, random_q, nq, nv):
    """frame_jacobian (all three reference frames) matches pin.getJointJacobian.
    Pin returns [linear; angular] in v-ordering — spherical columns are the
    body-frame omega tangent, the exact convention the oracle must reproduce."""
    robot = _grid(fixture)
    ref = RBDReference(robot)
    model, leaf = build_pin()
    data = model.createData()
    rng = np.random.default_rng(5)
    leaf_jid = robot.get_leaf_nodes()[0]
    leaf_name = robot.get_joint_by_id(leaf_jid).get_name()
    frames = {"LOCAL_WORLD_ALIGNED": pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
              "WORLD": pin.ReferenceFrame.WORLD,
              "LOCAL": pin.ReferenceFrame.LOCAL}
    for _ in range(10):
        q = random_q(rng)
        pin.computeJointJacobians(model, data, q)
        for name, rf in frames.items():
            J_ref = ref.frame_jacobian(q, leaf_name, reference_frame=name)
            J_pin = pin.getJointJacobian(model, data, leaf, rf)
            np.testing.assert_allclose(J_ref, J_pin, atol=1e-10,
                                       err_msg=f"frame_jacobian mismatch in {name}")


@pytest.mark.parametrize("fixture,build_pin,random_q,nq,nv", CASES)
def test_spherical_ee_pose_gradient_matches_pinocchio_and_manifold_fd(
    fixture, build_pin, random_q, nq, nv
):
    """d(pose)/dv: rows 0..2 == pin LWA linear Jacobian; rows 3..5 == E^-1 J_w
    from pin's LWA angular Jacobian; the WHOLE 6 x nv matrix == central
    differences of end_effector_pose on the quaternion manifold
    (q ⊗ exp(½·eps·e_k) via ref.integrate)."""
    robot = _grid(fixture)
    ref = RBDReference(robot)
    model, leaf = build_pin()
    data = model.createData()
    rng = np.random.default_rng(7)
    h = 1e-6
    checked = 0
    while checked < 8:
        q = random_q(rng)
        pin.forwardKinematics(model, data, q)
        if _near_gimbal(data.oMi[leaf].rotation):
            continue  # rpy rows ill-conditioned; resample (pose param choice, not a bug)
        checked += 1
        J = np.asarray(ref.end_effector_pose_gradient(q)[0])
        assert J.shape == (6, nv)
        # --- vs pinocchio (independent oracle) ---
        pin.computeJointJacobians(model, data, q)
        J_pin = pin.getJointJacobian(model, data, leaf, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)
        np.testing.assert_allclose(J[:3], J_pin[:3], atol=1e-10)
        Einv = _Einv_from_R(data.oMi[leaf].rotation)
        np.testing.assert_allclose(J[3:], Einv @ J_pin[3:], atol=1e-9)
        # --- vs FD on the quaternion manifold ---
        J_fd = np.zeros((6, nv))
        for k in range(nv):
            v = np.zeros(nv); v[k] = h
            pp = np.asarray(ref.end_effector_pose(ref.integrate(q, v))[0]).reshape(-1)
            pm = np.asarray(ref.end_effector_pose(ref.integrate(q, -v))[0]).reshape(-1)
            J_fd[:, k] = (pp - pm) / (2.0 * h)
        np.testing.assert_allclose(J, J_fd, atol=5e-8)


@pytest.mark.parametrize("fixture,build_pin,random_q,nq,nv", CASES)
def test_spherical_ee_pose_hessian_analytic_matches_manifold_fd(
    fixture, build_pin, random_q, nq, nv
):
    """d²(pose)/dv²: the analytic chain-composition Hessian (the CUDA-mirrored
    path, incl. the symmetrized intra-ball ½(S_k S_l + S_l S_k) curvature)
    matches (a) the FD-of-gradient-on-manifold oracle end_effector_pose_hessian
    and (b) direct second differences of the pose on the manifold."""
    robot = _grid(fixture)
    ref = RBDReference(robot)
    model, leaf = build_pin()
    data = model.createData()
    rng = np.random.default_rng(11)
    checked = 0
    while checked < 4:
        q = random_q(rng)
        pin.forwardKinematics(model, data, q)
        if _near_gimbal(data.oMi[leaf].rotation):
            continue
        checked += 1
        Ha = np.asarray(ref.end_effector_pose_hessian_analytic(q)[0])
        assert Ha.shape == (6, nv, nv)
        # (a) vs the FD-of-gradient oracle (integrate-based)
        Hf = np.asarray(ref.end_effector_pose_hessian(q)[0])
        np.testing.assert_allclose(Ha, Hf, atol=5e-7)
        # symmetry in the (v_k, v_l) pair axes
        np.testing.assert_allclose(Ha, np.transpose(Ha, (0, 2, 1)), atol=1e-12)
        # (b) direct 4-point second difference of the POSE on the manifold:
        #   d²f/dv_k dv_l ≈ [f(+k+l) - f(+k-l) - f(-k+l) + f(-k-l)] / (4 h²)
        # (both increments applied in ONE retract; agrees with the symmetrized
        # curvature convention to O(h²)).
        h2 = 1e-4
        for k in range(nv):
            for l in range(nv):
                vk = np.zeros(nv); vk[k] = h2
                vl = np.zeros(nv); vl[l] = h2
                f = lambda v: np.asarray(
                    ref.end_effector_pose(ref.integrate(q, v))[0]).reshape(-1)
                d2 = (f(vk + vl) - f(vk - vl) - f(-vk + vl) + f(-vk - vl)) / (4.0 * h2 * h2)
                np.testing.assert_allclose(Ha[:, k, l], d2, atol=5e-5,
                                           err_msg=f"pair ({k},{l})")
