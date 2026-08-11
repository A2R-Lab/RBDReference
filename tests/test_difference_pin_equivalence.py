"""Oracle gate for `RBDReference.difference` (Lie-group boxminus, GATO ASK 4).

Two independent checks:

1. **Exact-inverse round trip**: `difference(q, integrate(q, v)) == v` for
   random tangents on a floating quadruped (go2) and a fixed arm (iiwa14, where
   both collapse to plain vector ops). Uses only our own code, so it pins the
   internal consistency of the retract/difference pair.
2. **Pinocchio cross-check**: `difference(q1, q2) == pin.difference(model, q1,
   q2)` on random configuration pairs — the external anchor (canonical
   |phi| <= pi branch, floating tangent order [v_lin; omega]).
"""

from __future__ import annotations

import numpy as np
import pytest

from RBDReference.tests.conftest import MANIFEST_PATH
from RBDReference.tests.model_sources import iter_robot_cases, resolve_robot_spec
from RBDReference.equivalents.reference_backend import build_project_adapter
from RBDReference.equivalents.pinocchio_backend import build_pinocchio_adapter


def _case(robot_id, base_mode):
    for case in iter_robot_cases(MANIFEST_PATH, base_mode=base_mode):
        if case["spec"].robot_id == robot_id:
            return case["spec"]
    pytest.skip(f"{robot_id}-{base_mode} not in the robot manifest")


@pytest.mark.pinocchio_equivalence
@pytest.mark.parametrize("robot_id,base_mode", [("go2", "floating"), ("iiwa14", "fixed")])
def test_difference_round_trip_and_pinocchio(robot_id, base_mode):
    spec = _case(robot_id, base_mode)
    resolved = resolve_robot_spec(spec)
    project = build_project_adapter(spec, resolved, base_mode=base_mode)
    ref = project.reference
    robot = project.robot
    nq, nv = robot.get_num_pos(), robot.get_num_vel()
    floating = base_mode == "floating"

    rng = np.random.default_rng(42)

    def rand_q():
        q = rng.uniform(-1.0, 1.0, nq)
        if floating:
            q[3:7] /= np.linalg.norm(q[3:7])
        return q

    # 1. exact-inverse round trip
    for _ in range(25):
        q = rand_q()
        v = rng.uniform(-0.9, 0.9, nv)
        assert np.max(np.abs(ref.difference(q, ref.integrate(q, v)) - v)) < 1e-12

    # 2. pinocchio anchor
    pin_adapter = build_pinocchio_adapter(spec, resolved, base_mode=base_mode)
    import pinocchio as pin
    model = pin_adapter.model
    for _ in range(25):
        q1, q2 = rand_q(), rand_q()
        d_ours = ref.difference(q1, q2)
        d_pin = pin.difference(model, q1, q2)
        assert np.max(np.abs(d_ours - d_pin)) < 1e-10
