"""The tolerance table is closed: every key a test passes must exist, and every
per-robot override must name a known algorithm. An unknown key used to fall back
to the default bucket, which hid a mismatched key behind a 1e-7 comparison."""
import pytest

from RBDReference.tests.tolerances import (
    ALGORITHM_TOLERANCES,
    DEFAULT_TOLERANCE,
    ROBOT_ALGORITHM_TOLERANCES,
    get_tolerance,
)


def test_unknown_algorithm_key_is_an_error():
    with pytest.raises(KeyError, match="unknown tolerance algorithm 'f_ext_grad'"):
        get_tolerance("f_ext_grad")
    with pytest.raises(KeyError, match="unknown tolerance algorithm 'fd'"):
        get_tolerance("fd", robot_id="gen3")


def test_every_override_names_a_known_algorithm():
    unknown = {algo for (_robot, algo) in ROBOT_ALGORITHM_TOLERANCES} - set(ALGORITHM_TOLERANCES)
    assert not unknown


def test_override_wins_over_the_algorithm_bucket():
    assert get_tolerance("inverse_dynamics") is ALGORITHM_TOLERANCES["inverse_dynamics"]
    assert get_tolerance("inverse_dynamics", robot_id="iiwa14") is ALGORITHM_TOLERANCES["inverse_dynamics"]
    assert get_tolerance("inverse_dynamics", robot_id="gen3") is ROBOT_ALGORITHM_TOLERANCES[("gen3", "inverse_dynamics")]


def test_forward_dynamics_mirrors_the_inverse_dynamics_widenings():
    assert get_tolerance("forward_dynamics") == DEFAULT_TOLERANCE.__class__(
        rtol=DEFAULT_TOLERANCE.rtol, atol=DEFAULT_TOLERANCE.atol, note=get_tolerance("forward_dynamics").note
    )
    for robot_id in ("gen3", "h1_2"):
        fd = get_tolerance("forward_dynamics", robot_id=robot_id)
        idn = get_tolerance("inverse_dynamics", robot_id=robot_id)
        assert (fd.rtol, fd.atol) == (idn.rtol, idn.atol), robot_id
