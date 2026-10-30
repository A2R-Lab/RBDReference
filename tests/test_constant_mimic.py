"""Constant dependent joints agree with independently locked rigid transforms."""
import numpy as np
import pytest
from URDFParser import URDFParser
from RBDReference import RBDReference


def model(tmp_path, locked=False, independent=True, chained=False, branched=False):
    inertial = ('<inertial><origin xyz="0.03 -0.02 0.07"/>'
                '<mass value="1"/><inertia ixx="0.3" ixy="0" ixz="0" '
                'iyy="0.4" iyz="0" izz="0.5"/></inertial>')
    links = ''.join(f'<link name="{name}">{inertial}</link>'
                    for name in ['base', 'mount', 'arm', 'follower', 'tip'])
    drive = ('<joint name="drive" type="revolute"><parent link="mount"/>'
             '<child link="arm"/><origin xyz="0.1 0 0.2"/><axis xyz="0 1 0"/>'
             '<limit lower="-3" upper="3" effort="10" velocity="10"/></joint>')
    if not independent:
        drive = ('<joint name="drive" type="fixed"><parent link="mount"/>'
                 '<child link="arm"/><origin xyz="0.1 0 0.2"/></joint>')
    follower = (
        '<joint name="constant" type="fixed"><parent link="arm"/>'
        '<child link="follower"/><origin xyz="0.2 0 0.3" rpy="0 0 0.17"/></joint>'
        if locked else
        '<joint name="constant" type="revolute"><parent link="arm"/>'
        '<child link="follower"/><origin xyz="0.2 0 0.3"/><axis xyz="0 0 1"/>'
        '<limit lower="-3" upper="3" effort="10" velocity="10"/>'
        '<mimic joint="mount_fixed" multiplier="-0.5" offset="0.17"/></joint>'
    )
    xml = (f'<robot name="constant">{links}'
           '<joint name="mount_fixed" type="fixed"><parent link="base"/>'
           '<child link="mount"/><origin xyz="0.05 0.02 0.1"/></joint>'
           f'{drive}{follower}'
           '<joint name="tool" type="fixed"><parent link="follower"/>'
           '<child link="tip"/><origin xyz="0.1 0.2 0.05"/></joint></robot>')
    if chained:
        xml = xml.replace('<joint name="tool" type="fixed">',
                          '<joint name="tool" type="revolute">'
                          '<axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="10" velocity="10"/>'
                          '<mimic joint="constant" multiplier="2" offset="-0.1"/>')
    if branched:
        xml = xml.replace('<parent link="arm"/>', '<parent link="mount"/>')
    path = tmp_path / f'constant-{locked}-{independent}-{chained}-{branched}.urdf'
    path.write_text(xml)
    return URDFParser().parse(str(path))


@pytest.mark.parametrize('independent', [True, False])
def test_constant_pose_and_jacobian(tmp_path, independent):
    robot = model(tmp_path, independent=independent)
    ref = RBDReference(robot)
    for value in [-.4, .35]:
        q = np.full(robot.get_num_pos(), value)
        # Independent homogeneous composition: fixed mount, Ry(q), Rz(.17), tool.
        c, s = np.cos(value if independent else 0), np.sin(value if independent else 0)
        mount = np.eye(4); mount[:3, 3] = [.15, .02, .3]
        rotation = np.eye(4); rotation[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        follower = np.eye(4); follower[:3, 3] = [.2, 0, .3]
        c, s = np.cos(.17), np.sin(.17)
        follower[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        tool = np.eye(4); tool[:3, 3] = [.1, .2, .05]
        expected = mount @ rotation @ follower @ tool
        pose = ref.end_effector_pose(q, 'tool')[0]
        np.testing.assert_allclose(pose.reshape(-1)[:3], expected[:3, 3], atol=1e-13, rtol=0)
        gradient = ref.end_effector_pose_gradient(q, 'tool')[0]
        hessian = ref.end_effector_pose_hessian_analytic(q, ee_joint_names='tool')[0]
        for frame in ['LOCAL', 'WORLD', 'LOCAL_WORLD_ALIGNED']:
            jac = ref.frame_jacobian(q, 'tool', reference_frame=frame)
            dot = ref.frame_jacobian_dot(q, np.zeros_like(q), 'tool', reference_frame=frame)
            assert jac.shape == dot.shape == (6, len(q))
            np.testing.assert_allclose(dot, 0, atol=1e-13, rtol=0)
        if independent:
            h = 1e-6
            diff = (ref.end_effector_pose(q+h, 'tool')[0] - ref.end_effector_pose(q-h, 'tool')[0])/(2*h)
            np.testing.assert_allclose(gradient, diff, atol=1e-9, rtol=1e-8)
            diff_gradient = (ref.end_effector_pose_gradient(q+h, 'tool')[0]
                             - ref.end_effector_pose_gradient(q-h, 'tool')[0])/(2*h)
            np.testing.assert_allclose(np.asarray(hessian).reshape(6, 1), diff_gradient,
                                       atol=1e-9, rtol=1e-8)
        else:
            assert gradient.shape == (6, 0)
            assert np.asarray(hessian).size == 0


def test_constant_dynamics_match_rigid_body(tmp_path):
    dependent = RBDReference(model(tmp_path))
    locked = RBDReference(model(tmp_path, locked=True))
    for q in [np.array([-.4]), np.array([.35])]:
        qd, qdd, tau = np.array([.23]), np.array([-.31]), np.array([.8])
        for method, args in [
            ('inverse_dynamics', (q, qd, qdd)), ('crba', (q,)),
            ('forward_dynamics', (q, qd, tau)), ('aba', (q, qd, tau)),
            ('inverse_dynamics_gradient', (q, qd, qdd)),
            ('forward_dynamics_gradient', (q, qd, tau)),
            ('idsva_so_body_frame', (q, qd, qdd)),
            ('idsva_so_world_frame', (q, qd, qdd)),
            ('fdsva_so', (q, qd, tau)),
            ('coriolis_matrix', (q, qd)),
            ('jacobian_com', (q,)), ('ccrba', (q, qd)), ('dccrba', (q,)),
        ]:
            actual = getattr(dependent, method)(*args)
            expected = getattr(locked, method)(*args)
            # RNEA also returns per-body intermediates; rigid-body merging
            # deliberately changes their number. Compare generalized forces.
            if method == 'inverse_dynamics':
                actual, expected = actual[0], expected[0]
            if isinstance(actual, tuple):
                assert len(actual) == len(expected)
                for a, b in zip(actual, expected):
                    np.testing.assert_allclose(a, b, rtol=1e-10, atol=1e-11, err_msg=method)
            else:
                np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11, err_msg=method)


@pytest.mark.parametrize('method,argc', [
    ('inverse_dynamics', 3), ('inverse_dynamics_gradient', 3),
    ('crba', 1), ('minv', 1), ('forward_dynamics', 3), ('aba', 3),
    ('forward_dynamics_gradient', 3), ('idsva_so', 3), ('fdsva_so', 3),
    ('coriolis_matrix', 2), ('inverse_dynamics_regressor', 3),
    ('inverse_dynamics_regressor_gradient', 3), ('jacobian_com', 1),
    ('ccrba', 2), ('dccrba', 1), ('f_ext_jacobian_transpose', 1),
    ('f_ext_jacobian_transpose_dq', 1),
])
@pytest.mark.parametrize('chained', [False, True])
def test_zero_dof_constant_dynamics(tmp_path, method, argc, chained):
    ref = RBDReference(model(tmp_path, independent=False, chained=chained))
    result = getattr(ref, method)(*[np.empty(0) for _ in range(argc)])
    parts = result if isinstance(result, tuple) else [result]
    for part in parts:
        assert np.isfinite(np.asarray(part)).all(), method
    nb = ref.robot.get_num_bodies()
    shapes = {
        'inverse_dynamics': [(0,), (6, nb), (6, nb), (6, nb)],
        'inverse_dynamics_gradient': [(0, 0)],
        'crba': [(0, 0)], 'minv': [(0, 0)],
        'forward_dynamics': [(0,)], 'aba': [(0,)],
        'forward_dynamics_gradient': [(0, 0), (0, 0)],
        'idsva_so': [(0, 0, 0)] * 4, 'fdsva_so': [(0, 0, 0)] * 4,
        'coriolis_matrix': [(0, 0)],
        'inverse_dynamics_regressor': [(0, 10 * nb)],
        'inverse_dynamics_regressor_gradient': [(0, 0, 10 * nb)] * 2,
        'jacobian_com': [(3, 0)], 'ccrba': [(6, 0), (6,)],
        'dccrba': [(6, 0, 0)],
        'f_ext_jacobian_transpose': [(0, 6 * nb)],
        'f_ext_jacobian_transpose_dq': [(0, 6 * nb, 0)],
    }
    assert [np.asarray(p).shape for p in parts] == shapes[method]
    if method == 'inverse_dynamics':
        # A fixed mechanism has no generalized force or velocity, but gravity
        # still accelerates the reference frame and produces body forces.
        np.testing.assert_array_equal(parts[1], 0)
        assert np.linalg.norm(parts[2]) > 0
        assert np.linalg.norm(parts[3]) > 0
    elif method == 'ccrba':
        np.testing.assert_array_equal(parts[1], 0)


def test_constant_branch_does_not_read_poisoned_unrelated_velocity(tmp_path):
    robot = model(tmp_path, branched=True)
    ref = RBDReference(robot)
    q = np.array([.2])
    clean = ref.inverse_dynamics_fpass(q, np.zeros(1), np.zeros(1))
    poisoned = ref.inverse_dynamics_fpass(q, np.full(1, np.nan), np.full(1, np.nan))
    constant = robot.get_joint_by_name('constant').jid
    drive = robot.get_joint_by_name('drive').jid
    for actual, expected in zip(poisoned, clean):
        np.testing.assert_array_equal(actual[:, constant], expected[:, constant])
        assert np.isfinite(actual[:, constant]).all()
    assert np.isnan(poisoned[0][:, drive]).any(), 'Do not globally sanitize real NaN inputs'
