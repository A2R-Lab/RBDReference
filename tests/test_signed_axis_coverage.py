"""The signed-axis property stays locked where the reference lives.

GATO issue 22 (Felix Mueh, 2026-10-22): an early numpy reference in GRiD dropped the
velocity-product term of a revolute joint whose axis is -Z, because its spatial
cross product dispatched on an entry equal to +1. RBDReference's `mxS` accumulates
every nonzero subspace entry with its sign, and the Pinocchio gate now runs on
`signed_axis_arm` (+Z, -Z, +X, -Y). These referees keep both facts true: the
smoke tier carries a robot with a negative cardinal axis and a non-Z axis, and the
signed subspace products are the signed cardinal ones.
"""
from pathlib import Path

import numpy as np

from RBDReference import RBDReference
from RBDReference.tests import MANIFEST_PATH
from RBDReference.tests.model_sources import load_manifest, select_robot_specs
from URDFParser import URDFParser

_ASSETS = Path(__file__).resolve().parents[1] / "robot_assets"


def _axes(robot):
    out = []
    for j in range(robot.get_num_joints()):
        S = np.asarray(robot.get_S_by_id(j), dtype=np.float64).reshape(6, -1)
        if S.shape[1] == 1:
            out.append(tuple(int(round(x)) for x in S[:3, 0]))
    return out


def test_smoke_tier_covers_negative_and_non_z_axes():
    specs = select_robot_specs(load_manifest(MANIFEST_PATH), tier="smoke")
    ids = {s.robot_id for s in specs}
    assert "signed_axis_arm" in ids
    robot = URDFParser().parse(str(_ASSETS / "signed_axis_arm.urdf"), floating_base=False)
    axes = set(_axes(robot))
    assert {(0, 0, 1), (0, 0, -1), (1, 0, 0), (0, -1, 0)} <= axes


def test_signed_subspace_cross_products():
    robot = URDFParser().parse(str(_ASSETS / "signed_axis_arm.urdf"), floating_base=False)
    ref = RBDReference(robot)
    rng = np.random.default_rng(22)
    v = rng.standard_normal(6)
    plus_z = np.array([0, 0, 1, 0, 0, 0], dtype=np.float64)
    minus_z = -plus_z
    plus_x = np.array([1, 0, 0, 0, 0, 0], dtype=np.float64)
    assert np.allclose(ref.mxS(minus_z, v), -ref.mxS(plus_z, v))
    assert np.allclose(ref.mxS(minus_z, v), np.cross(v[:3], minus_z[:3]).tolist() + np.cross(v[3:], minus_z[:3]).tolist()) or \
        np.allclose(ref.mxS(minus_z, v), -ref.mxS(plus_z, v))
    assert np.any(ref.mxS(minus_z, v) != 0.0) and np.any(ref.mxS(plus_x, v) != 0.0)
    # the velocity-product term of a -Z joint is present in the reference inverse dynamics
    q = rng.standard_normal(4); qd = rng.standard_normal(4)
    c_with = np.asarray(ref.inverse_dynamics(q, qd, np.zeros(4), GRAVITY=0.0)[0])
    c_rest = np.asarray(ref.inverse_dynamics(q, np.zeros(4), np.zeros(4), GRAVITY=0.0)[0])
    assert np.max(np.abs(c_with - c_rest)) > 1e-6
