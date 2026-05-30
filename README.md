# RBDReference

A Python reference implementation of rigid body dynamics algorithms.

This package is designed to enable rapid prototyping and testing of new
algorithms and algorithmic optimizations. The CUDA / FPGA / accelerator
implementations in the parent GRiD-A2R repo use it as a golden CPU oracle
during testing (in turn grounded against Pinocchio's C++ implementation via
the in-package `equivalents/` layer; see "Equivalence testing" below).

If your favorite rigid body dynamics algorithm isn't yet implemented please
submit a PR with the implementation.

## Usage and API

This package relies on an already-parsed `robot` object from our
[URDFParser](https://github.com/robot-acceleration/URDFParser) package.

```python
from RBDReference import RBDReference
rbd = RBDReference(robot)
outputs = rbd.ALGORITHM(inputs)
```

### Core dynamics

| Algorithm | Signature |
|---|---|
| RNEA (inverse dynamics) | `(c, v, a, f) = rbd.rnea(q, qd, qdd=None, GRAVITY=-9.81)` |
| ABA (forward dynamics, articulated body) | `qdd = rbd.aba(q, qd, tau, f_ext=[], GRAVITY=-9.81)` |
| CRBA (composite-rigid-body mass matrix) | `M = rbd.crba(q)` |
| Minv (direct mass-matrix inverse) | `Minv = rbd.minv(q, output_dense=True)` |
| Forward dynamics (Minv·(τ−c)) | `qdd = rbd.forward_dynamics(q, qd, u, f_ext=None)` |
| Apply external forces (local-frame subtract) | `f_out = rbd.apply_external_forces(f_in, f_ext)` |

### Gradients

| Algorithm | Signature |
|---|---|
| ∂RNEA/∂(q, qd) | `dc_du = rbd.rnea_grad(q, qd, qdd=None, GRAVITY=-9.81)` returning `np.hstack((dc_dq, dc_dqd))` |
| ∂forward-dynamics/∂(q, qd) | `(dqdd_dq, dqdd_dqd) = rbd.forward_dynamics_grad(q, qd, u)` |

### Kinematics

| Algorithm | Signature |
|---|---|
| End-effector pose | `ee = rbd.end_effector_pose(q, ee_joint_names=None, ee_offsets=None)` |
| EE pose gradient (Jacobian) | `dee = rbd.end_effector_pose_gradient(q, ee_joint_names=None, ee_offsets=None)` |
| EE pose Hessian | `d2ee = rbd.end_effector_pose_hessian(q, offsets=None, ee_joint_names=None)` |

### Second-order

| Algorithm | Signature |
|---|---|
| IDSVA-SO (rank-3 ∂²τ tensors) | `(d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq) = rbd.idsva_so_body_frame(q, qd, qdd, GRAVITY=-9.81)` |
| IDSVA-SO world-frame (single-pass) | `(d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq) = rbd.idsva_so_world_frame(q, qd, qdd, GRAVITY=-9.81)` |
| FDSVA-SO (second-order forward dynamics) | `... = rbd.fdsva_so(q, qd, u, GRAVITY=-9.81)` |

The two IDSVA-SO variants are mathematically equivalent — they differ only
in **reference frame**:

| Variant | Reference frame | Best for |
|---|---|---|
| `idsva_so_body_frame` | Body-frame propagation, body-frame inertia, body-frame motion subspace. Multi-pass forward/backward sweeps. | **Fixed-base** robots — wins by a wide margin (e.g. iiwa14 fixed: 27 µs vs 827 µs on GPU). |
| `idsva_so_world_frame` | World-frame propagation, world-frame motion subspace, gravity baked into the main sweep. Single-pass reference (closer to the textbook spatial-vector-algebra derivation). | **Floating-base** robots — wins by 2–4× (e.g. iiwa14_floating 1.7×, g1_floating 3.6×, GPU). |

GPU benchmarks above are from `test/benchmarks/run_multi_version.py` on
sm_120 (RTX 5090). The crossover is purely a function of which kinematic
chain depth dominates: body-frame's subtree-broadcast pays off when the
chain is short and the tree is fixed; world-frame's single-pass cost is
flat in chain depth which wins as DOF grows under floating base.

### Per-pass helpers

Many algorithms also expose their internal passes (e.g. `rnea_fpass`,
`rnea_bpass`, `minv_bpass`, `minv_fpass`, `rnea_grad_fpass_dq` / `_dqd`,
`rnea_grad_bpass_dq` / `_dqd`) for unit-testing accelerator port pieces
independently. See `RBDReference/RBDReference.py` for full signatures.

## Installation

Two dependency tiers:

* **Base (runtime)** — the pure-Python reference. Only `numpy` + `sympy`:
  ```shell
  pip install -r requirements.txt
  ```
  Building a `robot` object also requires
  [URDFParser](https://github.com/robot-acceleration/URDFParser) (a sibling
  package, not on PyPI).

* **Developer / equivalence testing** — adds the Pinocchio backend and the
  test suite (`pin`, `robot_descriptions`, `beautifulsoup4`, `pybind11`,
  `pytest`):
  ```shell
  pip install -r requirements-dev.txt
  ```

## Equivalence testing

Every algorithm above is checked against Pinocchio (C++) as the golden oracle.
That machinery now lives **inside this package**:

* `equivalents/` — the reusable, shared-interface layer. Two interchangeable
  backends expose the *identical* adapter API:
  * `reference` — the pure-Python `RBDReference` (base deps only);
  * `pinocchio` — Pinocchio + the `pin_so_ext` second-order C++ binding,
    reordered into the project convention by `equivalents/conventions.py`.

  Pick one with the single swap point — no call-site changes:
  ```python
  from RBDReference.equivalents import build_adapter
  adapter = build_adapter(spec, resolved_model, base_mode, backend="pinocchio")
  # or leave backend=None and set GRID_REFERENCE_BACKEND=pinocchio in the env
  ```
  Because both backends share the surface, a consumer (e.g. the GRiD CUDA
  equivalence harness) switches which reference it compares against by flipping
  this one argument — turning the multi-hour pure-Python second-order
  references into millisecond C++ calls.

* `tests/` — this package's own suite, asserting the two backends agree. Run
  (from the directory containing `RBDReference`, e.g. the GRiD repo root):
  ```shell
  pytest RBDReference/tests/
  ```

The `pin_so_ext` binding wraps `pinocchio::ComputeRNEASecondOrderDerivatives`
(Pinocchio 3.x ships the C++ but does not expose it to Python); its loader
builds it on first use. The parent repo's `developer_install.sh` also builds it
ahead of time. See `equivalents/pin_so_ext/` and the parent
`test/benchmarks/README.md` for the Pinocchio install details.
