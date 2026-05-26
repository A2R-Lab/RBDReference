"""RBDReference equivalence test suite + its shared test infrastructure.

This package holds the pytest suite (pure-Python ``RBDReference`` vs Pinocchio)
plus the test-harness helpers that both this suite and the parent GRiD CUDA
equivalence harness import: the robot manifest + source lock, robot resolution
(``model_sources``), state sampling, tolerances, and comparators. The reusable
equivalence *implementations* (the two backends + the convention/translation
layer + the ``pin_so_ext`` C++ binding) live in ``RBDReference.equivalents``.
"""
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = TESTS_ROOT / "robot_manifest.json"
SOURCE_LOCK_PATH = TESTS_ROOT / "ROBOT_SOURCE_LOCK.json"
