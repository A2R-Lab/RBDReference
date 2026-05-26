"""Shared equivalence interface for RBDReference.

Two interchangeable backends expose an identical adapter API (``rnea``, ``aba``,
``forward_dynamics``, ``minv``, ``crba``, the first/second-order derivatives,
``idsva_so_body_frame``, ``fdsva_so``, the integrators, and
``end_effector_pose``/``_gradient``/``_hessian``):

* ``reference`` — the pure-Python :class:`RBDReference` (numpy + sympy only).
* ``pinocchio`` — Pinocchio (C++) plus the ``pin_so_ext`` second-order binding,
  with results reordered into the GRiD/project convention via :mod:`conventions`.

:func:`build_adapter` is the one-line swap: choose the backend explicitly, or
leave it ``None`` to read the ``GRID_REFERENCE_BACKEND`` environment variable
(default ``reference``). Both backends return the same method surface, so a
consumer (e.g. the GRiD CUDA equivalence harness) switches which reference it
compares against by changing this single argument.

The builders import their backends lazily so this package stays importable with
only the base requirements installed; the pinocchio backend's heavy
dependencies (pinocchio, beautifulsoup4, the compiled ``pin_so_ext``) and the
reference backend's URDFParser dependency are pulled in only when actually
built. See ``requirements-dev.txt`` for the developer/equivalence extras.
"""
import os
from pathlib import Path

EQUIVALENTS_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = EQUIVALENTS_ROOT / "robot_manifest.json"
SOURCE_LOCK_PATH = EQUIVALENTS_ROOT / "ROBOT_SOURCE_LOCK.json"

DEFAULT_BACKEND = "reference"
SUPPORTED_BACKENDS = ("reference", "pinocchio")


def resolve_backend(backend=None):
    """Normalize a backend selector. ``None`` falls back to the
    ``GRID_REFERENCE_BACKEND`` environment variable, then to ``reference``."""
    if backend is None:
        backend = os.environ.get("GRID_REFERENCE_BACKEND", DEFAULT_BACKEND)
    backend = str(backend).lower()
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unknown reference backend {backend!r}; expected one of {SUPPORTED_BACKENDS}."
        )
    return backend


def build_reference_adapter(spec, resolved_model, base_mode, **kwargs):
    """Pure-Python RBDReference adapter (numpy + sympy; URDFParser at build)."""
    from .reference_backend import build_project_adapter
    return build_project_adapter(spec, resolved_model, base_mode, **kwargs)


def build_pinocchio_adapter(spec, resolved_model, base_mode, **kwargs):
    """Pinocchio (C++) adapter exposing the same surface as the reference one."""
    from .pinocchio_backend import build_pinocchio_adapter as _build_pinocchio
    return _build_pinocchio(spec, resolved_model, base_mode, **kwargs)


def build_adapter(spec, resolved_model, base_mode, backend=None, **kwargs):
    """Build an equivalence adapter for ``backend`` (``"reference"`` or
    ``"pinocchio"``). This is the one-line reference swap."""
    if resolve_backend(backend) == "pinocchio":
        return build_pinocchio_adapter(spec, resolved_model, base_mode, **kwargs)
    return build_reference_adapter(spec, resolved_model, base_mode, **kwargs)
