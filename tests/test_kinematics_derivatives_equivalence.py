import numpy as np
import pytest

from RBDReference.tests.conftest import MANIFEST_PATH, build_case_params
from RBDReference.tests.comparators import assert_close
from RBDReference.tests.model_sources import iter_robot_cases
from RBDReference.tests.state_sampling import build_dynamics_samples


def select_derivative_target(project_model, pinocchio_model):
    mimic_joint_names = set(getattr(pinocchio_model, "mimic_joint_names", []))
    leaf_ids = project_model.robot.get_leaf_nodes()
    leaf_names = [
        project_model.robot.get_joint_by_id(leaf_id).get_name()
        for leaf_id in leaf_ids
    ]
    for joint_name in reversed(leaf_names):
        if joint_name in mimic_joint_names:
            continue
        if joint_name not in pinocchio_model.frame_names:
            continue
        return joint_name, np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return None


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params(base_mode="floating"))
def test_floating_root_derivative_interface_is_well_formed(spec, base_mode, project_model):
    root_joint = project_model.robot.get_joint_by_id(0)
    root_indices = project_model.robot.get_joint_index_q(0)
    assert root_joint.get_local_q_dim() == 7
    assert list(root_indices) == [0, 1, 2, 3, 4, 5, 6]

    sample = build_dynamics_samples(project_model)[1]
    q_block = np.asarray(sample.q[root_indices], dtype=np.float64)
    xmat = np.asarray(
        project_model.robot.get_Xmat_hom_Func_by_id(0)(q_block),
        dtype=np.float64,
    )
    assert xmat.shape == (4, 4)
    assert np.isfinite(xmat).all()

    for local_index in range(root_joint.get_local_q_dim()):
        dX = np.asarray(
            project_model.robot.get_dXmat_hom_local_Func_by_id(0, local_index)(q_block),
            dtype=np.float64,
        )
        assert dX.shape == (4, 4)
        assert np.isfinite(dX).all()

    d2X = np.asarray(
        project_model.robot.get_d2Xmat_hom_local_Func_by_id(0, 3, 6)(q_block),
        dtype=np.float64,
    )
    assert d2X.shape == (4, 4)
    assert np.isfinite(d2X).all()


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params(base_mode="fixed"))
def test_fixed_base_pose_gradient_matches_pinocchio(
    spec, base_mode, project_model, pinocchio_model
):
    selection = select_derivative_target(project_model, pinocchio_model)
    if selection is None:
        pytest.skip(f"No common articulated leaf target available for {spec.robot_id}")
    target_name, offset = selection
    sample = build_dynamics_samples(project_model)[1]
    actual = project_model.end_effector_pose_gradient(sample.q, target_name, offset=offset)
    expected = pinocchio_model.end_effector_pose_gradient(sample.q, target_name, offset=offset)
    assert_close(actual, expected, algorithm="pose_gradient", robot_id=spec.robot_id)


@pytest.mark.parametrize(("spec", "base_mode"), build_case_params(base_mode="floating"))
def test_floating_base_pose_gradient_matches_pinocchio(
    spec, base_mode, project_model, pinocchio_model
):
    selection = select_derivative_target(project_model, pinocchio_model)
    if selection is None:
        pytest.skip(f"No common articulated leaf target available for {spec.robot_id}")
    target_name, offset = selection
    sample = build_dynamics_samples(project_model)[1]
    actual = project_model.end_effector_pose_gradient(sample.q, target_name, offset=offset)
    expected = pinocchio_model.end_effector_pose_gradient(sample.q, target_name, offset=offset)
    assert_close(actual, expected, algorithm="pose_gradient", robot_id=spec.robot_id)


def build_hessian_case_params():
    # iiwa14 has been the canonical Hessian comparator since the analytic
    # path landed; fr3 is added as the explicit mimic-joint coverage point
    # (analytic d2ee on fr3 exposed a mimic-aware gap in URDFParser before
    # mimic support landed -- see Joint.set_mimic + RBDReference's mimic
    # multiplier handling in end_effector_pose_gradient /
    # end_effector_pose_hessian_analytic).
    hessian_robots = {"iiwa14", "fr3"}
    params = []
    for case in iter_robot_cases(MANIFEST_PATH):
        spec = case["spec"]
        base_mode = case["base_mode"]
        if spec.robot_id not in hessian_robots:
            continue
        # Only fixed-base for fr3 (floating-base d2ee for fr3 brings in the
        # base-rotation rpy branch noise on a manipulator + finger combo,
        # which isn't the failure mode this case is meant to cover).
        if spec.robot_id == "fr3" and base_mode != "fixed":
            continue
        marks = [
            pytest.mark.pinocchio_equivalence,
            pytest.mark.developer_only,
            getattr(pytest.mark, f"robot_{spec.tier}"),
        ]
        if base_mode == "floating":
            marks.append(pytest.mark.floating_base)
        params.append(pytest.param(spec, base_mode, id=f"{spec.robot_id}-{base_mode}", marks=marks))
    return params


@pytest.mark.parametrize(("spec", "base_mode"), build_hessian_case_params())
def test_pose_hessian_matches_pinocchio(spec, base_mode, project_model, pinocchio_model):
    selection = select_derivative_target(project_model, pinocchio_model)
    if selection is None:
        pytest.skip(f"No common articulated leaf target available for {spec.robot_id}")
    target_name, offset = selection
    sample = build_dynamics_samples(project_model)[1]
    actual = project_model.end_effector_pose_hessian(sample.q, target_name, offset=offset)
    expected = pinocchio_model.end_effector_pose_hessian(sample.q, target_name, offset=offset)
    assert_close(actual, expected, algorithm="pose_hessian", robot_id=spec.robot_id)


@pytest.mark.parametrize(("spec", "base_mode"), build_hessian_case_params())
def test_pose_hessian_analytic_matches_fd(spec, base_mode, project_model):
    """Sanity check: the analytic d2ee implementation agrees with the FD
    oracle on the project side. This is the failure mode the A.1 (Python)
    landing exposed for fr3 -- before URDFParser learned `<mimic>` joints
    the analytic d2ee Hessian disagreed with the FD oracle on the mimicked
    column (the FD-on-J oracle was masking it via antisymmetric
    cancellation while the analytic d^2(pose)/dv^2 surfaced it). With the
    mimic-aware Jacobian + per-block scale fold, both paths now match to
    the FD floor.
    """
    selection = None
    if hasattr(project_model, "robot"):
        leaves = project_model.robot.get_leaf_nodes()
        for leaf_jid in reversed(leaves):
            joint = project_model.robot.get_joint_by_id(leaf_jid)
            if getattr(joint, "is_mimic", False):
                continue
            selection = joint.get_name()
            break
    if selection is None:
        pytest.skip(f"No non-mimic leaf joint available for {spec.robot_id}")
    target_name = selection
    offset = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    sample = build_dynamics_samples(project_model)[1]
    H_analytic = project_model.reference.end_effector_pose_hessian_analytic(
        sample.q, ee_joint_names=target_name, offsets=[offset]
    )
    H_fd = project_model.reference.end_effector_pose_hessian(
        sample.q, ee_joint_names=target_name, offsets=[offset]
    )
    assert len(H_analytic) == len(H_fd)
    diff = np.max(np.abs(H_analytic[0] - H_fd[0]))
    scale = max(1.0, float(np.max(np.abs(H_fd[0]))))
    # FD truncation noise is ~1e-7 * scale on these robots; analytic-vs-FD
    # at 1e-5 * scale catches structural mimic mishandling without
    # tripping on FD noise.
    assert diff < 1e-5 * scale, (
        f"{spec.robot_id} analytic d2ee disagrees with FD oracle "
        f"(max |diff| = {diff:.3e}, scale = {scale:.3e})"
    )
