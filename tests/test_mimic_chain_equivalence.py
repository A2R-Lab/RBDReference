"""Chained-mimic dynamics == manually-flattened dynamics (oracle level).

URDFParser flattens mimic chains at parse time (q_c = m_c*(m_b*q_a + o_b) +
o_c  ==>  one-hop target=a with composed multiplier/offset), so a model whose
URDF declares j3->j2->j1 must produce BIT-IDENTICAL dynamics to the same model
authored with the pre-composed one-hop relations j3->j1, j2->j1. That pins the
whole downstream consumption chain (dense index maps, q_for_joint, Xmats,
RNEA/CRBA/ABA) on the flattened table, independent of any external oracle;
the one-hop path itself is pinned vs Pinocchio by the fr3 equivalence suites.
"""
import contextlib
import io

import numpy as np

from URDFParser import URDFParser
from RBDReference import RBDReference


_INERTIALS = {
    "l1": (1.3, (0.05, 0.02, 0.10), (0.05, 0.01, 0.005, 0.06, 0.002, 0.04)),
    "l2": (0.9, (0.04, -0.03, 0.08), (0.04, 0.003, 0.001, 0.05, 0.004, 0.03)),
    "l3": (0.6, (0.0, 0.0, 0.06), (0.02, 0.0, 0.0, 0.02, 0.001, 0.015)),
}

# chain: j2 mimics j1 (2.0, 0.1); j3 mimics j2 (-1.5, 0.2)
_M2, _O2 = 2.0, 0.1
_M3, _O3 = -1.5, 0.2
# flattened: j3 -> j1 with composed coefficients
_M3F = _M3 * _M2                # -3.0
_O3F = _M3 * _O2 + _O3          # 0.05


def _link(name):
    m, (cx, cy, cz), (ixx, ixy, ixz, iyy, iyz, izz) = _INERTIALS[name]
    return (
        f'<link name="{name}"><inertial>'
        f'<origin xyz="{cx} {cy} {cz}" rpy="0 0 0"/><mass value="{m}"/>'
        f'<inertia ixx="{ixx}" ixy="{ixy}" ixz="{ixz}" iyy="{iyy}" iyz="{iyz}" izz="{izz}"/>'
        "</inertial></link>"
    )


def _joint(name, parent, child, axis, mimic=None):
    mimic_xml = ""
    if mimic is not None:
        tgt, mult, off = mimic
        mimic_xml = f'<mimic joint="{tgt}" multiplier="{mult}" offset="{off}"/>'
    return (
        f'<joint name="{name}" type="revolute"><parent link="{parent}"/>'
        f'<child link="{child}"/><axis xyz="{axis}"/>'
        '<origin xyz="0.1 0 0.2" rpy="0 0 0"/>'
        '<limit lower="-3" upper="3" effort="10" velocity="10"/>'
        f"{mimic_xml}</joint>"
    )


def _parse(tmp_path, name, j2_mimic, j3_mimic):
    urdf = (
        f'<robot name="{name}"><link name="base"/>'
        + _link("l1") + _link("l2") + _link("l3")
        + _joint("j1", "base", "l1", "0 0 1")
        + _joint("j2", "l1", "l2", "0 1 0", mimic=j2_mimic)
        + _joint("j3", "l2", "l3", "1 0 0", mimic=j3_mimic)
        + "</robot>"
    )
    path = tmp_path / f"{name}.urdf"
    path.write_text(urdf)
    with contextlib.redirect_stdout(io.StringIO()):
        robot = URDFParser().parse(str(path))
    assert robot is not None
    return robot


def _assert_same(a, b):
    """Bit-identical comparison over an array or a (possibly ragged) tuple of arrays."""
    if isinstance(a, tuple):
        assert isinstance(b, tuple) and len(a) == len(b)
        for part_a, part_b in zip(a, b):
            _assert_same(part_a, part_b)
        return
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_chained_mimic_dynamics_match_flattened(tmp_path):
    chained = _parse(tmp_path, "chained", ("j1", _M2, _O2), ("j2", _M3, _O3))
    flat = _parse(tmp_path, "flat", ("j1", _M2, _O2), ("j1", _M3F, _O3F))

    # the flattened tables themselves must coincide
    for jname in ("j2", "j3"):
        jc = chained.get_joint_by_name(jname)
        jf = flat.get_joint_by_name(jname)
        assert jc.mimic_target_id == jf.mimic_target_id
        assert jc.get_mimic_multiplier() == jf.get_mimic_multiplier()
        assert jc.get_mimic_offset() == jf.get_mimic_offset()

    ref_c = RBDReference(chained)
    ref_f = RBDReference(flat)
    nq = chained.get_num_pos()
    nv = chained.get_num_vel()
    assert nq == 1 and nv == 1  # one independent DoF drives the whole chain

    rng = np.random.default_rng(0)
    for _ in range(5):
        q = rng.uniform(-1.5, 1.5, nq)
        qd = rng.uniform(-1.0, 1.0, nv)
        qdd = rng.uniform(-1.0, 1.0, nv)
        _assert_same(ref_c.inverse_dynamics(q, qd, qdd), ref_f.inverse_dynamics(q, qd, qdd))
        _assert_same(ref_c.crba(q), ref_f.crba(q))
        u = rng.uniform(-1.0, 1.0, nv)
        _assert_same(ref_c.forward_dynamics(q, qd, u), ref_f.forward_dynamics(q, qd, u))
        _assert_same(ref_c.aba(q, qd, u), ref_f.aba(q, qd, u))
