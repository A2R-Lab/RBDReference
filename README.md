# RBDReference

A Python reference implementation of rigid body dynamics algorithms.

This package is designed to enable rapid prototyping and testing of new
algorithms and algorithmic optimizations. The CUDA / FPGA / accelerator
implementations in the parent GRiD-A2R repo use it as a golden CPU oracle
during testing (in turn grounded against Pinocchio's C++ implementation in
`test/pinocchio_equivalents/`).

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
| Forward dynamics (Minv·(τ−c)) | `qdd = rbd.forward_dynamics(q, qd, u)` |
| Apply external forces (spatial xforms) | `f_out = rbd.apply_external_forces(q, f_in, f_ext)` |

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
| IDSVA-SO (rank-3 ∂²τ tensors) | `(d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq) = rbd.idsva_so(q, qd, qdd, GRAVITY=-9.81)` |
| IDSVA-SO spatial_v2 (single-pass) | `(d2tau_dq, d2tau_dqd, d2tau_cross, dM_dq) = rbd.idsva_so_spatial_v2(q, qd, qdd, GRAVITY=-9.81)` |
| FDSVA-SO (second-order forward dynamics) | `... = rbd.fdsva_so(q, qd, u, GRAVITY=-9.81)` |

The two IDSVA-SO variants are mathematically equivalent. `idsva_so` is the
original optimized formulation; `idsva_so_spatial_v2` is a cleaner
single-pass implementation closer to the textbook spatial_v2 derivation,
useful as a structural reference.

### Per-pass helpers

Many algorithms also expose their internal passes (e.g. `rnea_fpass`,
`rnea_bpass`, `minv_bpass`, `minv_fpass`, `rnea_grad_fpass_dq` / `_dqd`,
`rnea_grad_bpass_dq` / `_dqd`) for unit-testing accelerator port pieces
independently. See `RBDReference/RBDReference.py` for full signatures.

## Installation

The only external dependency is `numpy`:
```shell
pip install -r requirements.txt
```

This package also depends on
[URDFParser](https://github.com/robot-acceleration/URDFParser).

## Equivalence testing

Each algorithm above is checked against Pinocchio (C++) as the golden
oracle in the parent repo's `test/pinocchio_equivalents/` suite. Run:
```shell
pytest test/pinocchio_equivalents/
```
See `test/benchmarks/README.md` for the Pinocchio install steps needed to
build the `pin_so_ext/` binding.
