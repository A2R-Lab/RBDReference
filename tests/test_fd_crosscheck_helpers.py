"""Exercise the retained FD cross-check helpers against their analytic twins.

`_d2Integrate_fd` and `_frame_jacobian_dot_fd` are kept exactly to referee the
analytic `d2Integrate` / `frame_jacobian_dot` — but until this module nothing
called them, so they were unverified weight (2026-09-18 audit). One fixed-base
and one floating case each keeps them honest (and keeps them from bit-rotting).
"""
import numpy as np
import pytest

from RBDReference.tests.conftest import build_case_params
from RBDReference.tests.state_sampling import build_dynamics_samples


def _cases(base_mode):
    # build_case_params returns pytest.param objects; filter on their values.
    return [p for p in build_case_params(base_mode)
            if p.values[0].robot_id in ("iiwa14", "go2")]


@pytest.mark.parametrize(("spec", "base_mode"), _cases("fixed") + _cases("floating"))
def test_fd_helpers_match_analytic(spec, base_mode, project_model):
    ref = project_model.reference
    sample = build_dynamics_samples(project_model)[0]
    q, qd = sample.q, sample.qd

    # frame_jacobian_dot vs its central-FD twin (FD of the analytic J along the
    # integrator flow; O(step^2) truncation). Explicit leaf frame — the helper
    # does not default-resolve frame_name.
    leaf = ref.robot.get_joint_by_id(int(ref.robot.get_leaf_nodes()[0])).get_name()
    Jd = np.asarray(ref.frame_jacobian_dot(q, qd, leaf), dtype=np.float64)
    Jd_fd = np.asarray(ref._frame_jacobian_dot_fd(q, qd, leaf), dtype=np.float64)
    assert np.max(np.abs(Jd - Jd_fd)) < 5e-5

    # d2Integrate vs its 4th-order FD twin (nonzero only on the floating
    # base's dv x dv block; both return exact zeros on fixed base).
    v_dt = 0.1 * np.asarray(qd, dtype=np.float64)
    H = np.asarray(ref.d2Integrate(q, v_dt, "v", "v"), dtype=np.float64)
    H_fd = np.asarray(ref._d2Integrate_fd(q, v_dt, "v", "v"), dtype=np.float64)
    assert np.max(np.abs(H - H_fd)) < 5e-5
