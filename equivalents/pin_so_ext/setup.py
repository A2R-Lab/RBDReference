"""Build script for the Pinocchio second-order RNEA binding.

Resolution order for the pinocchio C++ headers/libs:
  1. pkg-config (pin <= 3 wheels ship ``cmeel.prefix/lib/pkgconfig/pinocchio.pc``;
     the loader's ``_build_env`` prepends the venv's cmeel pkgconfig dirs, and
     ``eigen3.pc`` comes from a system libeigen3-dev or the cmeel-eigen wheel's
     ``share/pkgconfig``).
  2. direct cmeel.prefix probing (pin >= 4 wheels DROPPED pinocchio.pc but still
     ship ``include/pinocchio`` + ``include/eigen3`` + ``lib/libpinocchio_*.so``
     inside the prefix) — includes, libs and an rpath are derived from the
     prefix, no pkg-config needed. This is what a bare CI runner uses.
"""
import os
import subprocess
import sys
import sysconfig
from pathlib import Path

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext


def _pin_pkg_config():
    raw = subprocess.check_output(["pkg-config", "--cflags", "--libs", "pinocchio"], text=True)
    include_dirs, library_dirs, libraries, defines, extra_link = [], [], [], [], []
    for token in raw.split():
        if token.startswith("-I"):
            include_dirs.append(token[2:])
        elif token.startswith("-L"):
            library_dirs.append(token[2:])
        elif token.startswith("-l"):
            libraries.append(token[2:])
        elif token.startswith("-D"):
            key_val = token[2:].split("=", 1)
            defines.append((key_val[0], key_val[1] if len(key_val) > 1 else None))
        elif token.startswith("-Wl,-rpath,"):
            extra_link.append(token)
    return include_dirs, library_dirs, libraries, defines, extra_link


def _cmeel_prefixes():
    out = []
    for key in ("purelib", "platlib"):
        base = sysconfig.get_paths().get(key)
        if not base:
            continue
        cand = Path(base) / "cmeel.prefix"
        if cand.is_dir() and cand not in out:
            out.append(cand)
    return out


def _pin_from_cmeel_prefix():
    """pin >= 4 layout: headers + libs in cmeel.prefix, no pinocchio.pc."""
    for prefix in _cmeel_prefixes():
        inc = prefix / "include"
        lib = prefix / "lib"
        if not (inc / "pinocchio").is_dir():
            continue
        include_dirs = [str(inc)]
        eigen_inc = inc / "eigen3"
        if eigen_inc.is_dir():
            include_dirs.append(str(eigen_inc))
        # pin >= 4 moved model/data headers and ships the old paths as a
        # deprecated/ compatibility tree — put it on the include path so the
        # classic <pinocchio/multibody/data.hpp> spellings keep resolving.
        compat = inc / "pinocchio" / "deprecated"
        if compat.is_dir():
            include_dirs.append(str(compat))
        # pin 4 splits the library; the core algorithms live in
        # libpinocchio_default; older single-lib layouts keep libpinocchio.
        libraries = []
        for name in ("pinocchio_default", "pinocchio_parsers", "pinocchio"):
            if (lib / f"lib{name}.so").exists():
                libraries.append(name)
        if not libraries:
            continue
        extra_link = [f"-Wl,-rpath,{lib}"]
        return include_dirs, [str(lib)], libraries, [], extra_link
    raise RuntimeError(
        "pin_so_ext: could not resolve pinocchio via pkg-config OR a cmeel.prefix "
        "with include/pinocchio — is the `pin` wheel installed in this environment?")


try:
    include_dirs, library_dirs, libraries, defines, extra_link = _pin_pkg_config()
except (OSError, subprocess.CalledProcessError):
    include_dirs, library_dirs, libraries, defines, extra_link = _pin_from_cmeel_prefix()

ext = Pybind11Extension(
    "pin_so_ext",
    sources=["pin_so_ext.cpp"],
    include_dirs=include_dirs,
    library_dirs=library_dirs,
    libraries=libraries,
    define_macros=defines,
    extra_link_args=extra_link,
    cxx_std=17,
)

setup(
    name="pin_so_ext",
    version="0.1.0",
    ext_modules=[ext],
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
