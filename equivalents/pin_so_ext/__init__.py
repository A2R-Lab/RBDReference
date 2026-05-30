"""Loader for the Pinocchio second-order RNEA Python extension.

The C++ source lives next to this file (`pin_so_ext.cpp`). The compiled extension
is built into this same directory via `python setup.py build_ext --inplace`. The
`load()` helper compiles on demand the first time it's called -- and rebuilds when
the source is newer than the compiled artifact -- so a fresh checkout can run the
tests without manual setup as long as pybind11 and g++ are available.

`pinocchio.pc` ships inside the active venv's `cmeel.prefix` (not on the system
pkg-config path), so the build prepends that directory to `PKG_CONFIG_PATH`
automatically; you no longer need `developer_install.sh` to have exported it.
"""
import importlib
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

_DIR = Path(__file__).resolve().parent
_SOURCES = ("pin_so_ext.cpp", "setup.py")


def _find_compiled():
    for entry in _DIR.iterdir():
        if entry.name.startswith("pin_so_ext.") and entry.name.endswith(".so"):
            return entry
    return None


def _is_stale(so):
    """True if the compiled extension is missing or older than its sources."""
    if so is None:
        return True
    so_mtime = so.stat().st_mtime
    return any(
        (_DIR / src).exists() and (_DIR / src).stat().st_mtime > so_mtime
        for src in _SOURCES
    )


def _cmeel_pkgconfig_dirs():
    """pkgconfig dir(s) holding pinocchio.pc inside the venv's cmeel.prefix."""
    dirs = []
    for key in ("purelib", "platlib"):
        base = sysconfig.get_paths().get(key)
        if not base:
            continue
        cand = Path(base) / "cmeel.prefix" / "lib" / "pkgconfig"
        if cand.is_dir() and cand not in dirs:
            dirs.append(cand)
    return dirs


def _build_env():
    """Environment for the build with the venv's cmeel pkgconfig prepended so
    pkg-config can find pinocchio.pc without a manual export."""
    env = os.environ.copy()
    extra = os.pathsep.join(str(d) for d in _cmeel_pkgconfig_dirs())
    if extra:
        existing = env.get("PKG_CONFIG_PATH", "")
        env["PKG_CONFIG_PATH"] = extra + (os.pathsep + existing if existing else "")
    return env


def _build():
    cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
    subprocess.check_call(cmd, cwd=_DIR, env=_build_env())


def load():
    """Return the compiled `pin_so_ext` module, building it on first use (or when
    the C++ source has changed since the last build)."""
    so = _find_compiled()
    if _is_stale(so):
        _build()
        so = _find_compiled()
        if so is None:
            raise RuntimeError(
                "Failed to build pin_so_ext. Ensure pybind11 is installed and "
                "Pinocchio development headers are available (pkg-config pinocchio); "
                "pinocchio.pc ships in the venv's cmeel.prefix/lib/pkgconfig."
            )
    if str(_DIR) not in sys.path:
        sys.path.insert(0, str(_DIR))
    return importlib.import_module("pin_so_ext")
