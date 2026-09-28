# RBDReference

A NumPy reference implementation of robot dynamics, kinematics, analytical
derivatives, and integration. Active development lives at
[A2R-Lab/RBDReference](https://github.com/A2R-Lab/RBDReference); the
[robot-acceleration repository](https://github.com/robot-acceleration/RBDReference)
preserves the original implementation.

This package is designed to enable rapid prototyping and testing of new
algorithms and algorithmic optimizations. The CUDA / FPGA / accelerator
implementations can use it as a CPU reference
during testing (in turn grounded against Pinocchio's C++ implementation via
the in-package `equivalents/` layer; see "Equivalence testing" below).

If your favorite rigid body dynamics algorithm isn't yet implemented please
submit a PR with the implementation.

## Usage and API

This package relies on an already-parsed `robot` object from our
[URDFParser](https://github.com/A2R-Lab/URDFParser) package.

```python
from pathlib import Path
import numpy as np
from URDFParser import URDFParser
from RBDReference import RBDReference

# Run from the common parent of the RBDReference and URDFParser checkouts.
robot = URDFParser().parse(Path("RBDReference/robot_assets/iiwa14.urdf"))
rbd = RBDReference(robot)
q = np.zeros(robot.get_num_pos())  # fixed-base scalar joints in this example
qd = np.zeros(robot.get_num_vel())
tau, spatial_velocity, spatial_acceleration, spatial_force = rbd.inverse_dynamics(q, qd)
M = rbd.crba(q)
qdd = rbd.forward_dynamics(q, qd, tau)
```

`q` has width **NQ**; velocities, accelerations, generalized forces, and
configuration tangent perturbations have width **NV**. Do not size arrays by
the number of joints or bodies. Use `integrate(q, delta)` to perturb a
configuration and `difference(q_from, q_to)` for a tangent-space error.
Quaternions must represent valid rotations; an all-zero floating/spherical
quaternion is not a neutral configuration.

### Core dynamics

| Algorithm | Signature |
|---|---|
| Inverse dynamics (RNEA / Recursive Newton-Euler Algorithm) | `(c, v, a, f) = rbd.inverse_dynamics(q, qd, qdd=None, GRAVITY=-9.81)` |
| ABA (forward dynamics, articulated body) | `qdd = rbd.aba(q, qd, tau, f_ext=[], GRAVITY=-9.81)` |
| CRBA (composite-rigid-body mass matrix) | `M = rbd.crba(q)` |
| Minv (direct mass-matrix inverse) | `Minv = rbd.minv(q, output_dense=True)` |
| Forward dynamics (Minv·(τ−c)) | `qdd = rbd.forward_dynamics(q, qd, u, f_ext=None)` |
| Apply external forces (local-frame subtract) | `f_out = rbd.apply_external_forces(f_in, f_ext)` |

### Gradients

| Algorithm | Signature |
|---|---|
| ∂(inverse dynamics)/∂(q, qd) | `dc_du = rbd.inverse_dynamics_gradient(q, qd, qdd=None, GRAVITY=-9.81)` returning `np.hstack((dc_dq, dc_dqd))` |
| ∂forward-dynamics/∂(q, qd) | `(dqdd_dq, dqdd_dqd) = rbd.forward_dynamics_gradient(q, qd, u)` |
| External-force gradients | `rbd.f_ext_gradient(q)` — analytic `∂τ/∂f_ext = −Jᵀ` and `∂q̈/∂f_ext = M⁻¹Jᵀ` per body |

### State space / integrators

Lie-group state operations handle floating and spherical joints. They collapse
to the familiar `+`/`−` only for Euclidean scalar-joint configurations:

| Algorithm | Signature |
|---|---|
| Retract (q ⊕ v·dt) | `q_new = rbd.integrate(q, v_dt)` (matches `pin.integrate`) |
| Tangent Jacobians of retract | `J = rbd.dIntegrate(q, v_dt, with_respect_to)` (`'q'`/`'v'`, nv×nv) |
| Second-order retract derivative | `H = rbd.d2Integrate(q, v_dt, arg1, arg2)` (nv×nv×nv tangent derivative of `dIntegrate`) |
| Boxminus (q_to ⊖ q_from) | `v = rbd.difference(q_from, q_to)` (matches `pin.difference`) |
| Tangent Jacobians of difference | `J = rbd.dDifference(q_from, q_to, with_respect_to)` (`'from'`/`'to'`) |
| One integration step | `x_kp1 = rbd.integrator(q, qd, u, dt, integrator_type="euler")` (`euler`/`semi_implicit_euler`/`constant_acceleration`/`trapezoidal`/`midpoint`/`rk4`) |
| Integrator Jacobian | `AB = rbd.integrator_gradient(q, qd, u, dt, ...)` — `[A | B]` of shape (2·nv, 3·nv), tangent-space `[d/dq | d/dqd | d/du]` |
| Tangent-space quadratic state cost | `rbd.quadratic_state_cost_tangent(x, x_des, Q)` — log-map error `[difference(q_des,q); qd−qd_des]`, diagonal `Q` of size 2·nv |

The single-evaluation update previously named `trapezoidal` is now
`constant_acceleration`; `si_euler` and `rk3` are removed without aliases.
The current `trapezoidal` is explicit two-stage Heun. Midpoint and Heun have
order two, and full-state RK4 has order four on Euclidean configurations.
On floating/spherical rotational manifolds the base-point retractions generally
give only order two, including RK4; this is not a Munthe-Kaas method.
Spherical multi-stage gradients and multi-stage step Hessians are unsupported.
Reference availability alone does not imply support on every GPU surface;
consult [GRiD's support documentation](https://a2r-lab.github.io/GRiD/docs/user_guide/tutorials/cuda_support_status.html)
for the generated kernels and bindings.

`momentum_cost` returns an exact full tangent-state gradient of shape `(2*nv,)`
and a Gauss–Newton Hessian of shape `(2*nv, 2*nv)`, including configuration and
cross blocks. These are not ambient `(nq+nv)` arrays. The GN Hessian uses the
full momentum residual Jacobian, not a frozen-configuration approximation.

### Kinematics


| Algorithm | Signature |
|---|---|
| End-effector pose | `ee = rbd.end_effector_pose(q, ee_joint_names=None, ee_offsets=None)` |
| EE pose gradient (Jacobian) | `dee = rbd.end_effector_pose_gradient(q, ee_joint_names=None, ee_offsets=None)` |
| EE pose Hessian | `d2ee = rbd.end_effector_pose_hessian(q, offsets=None, ee_joint_names=None)` |
| EE pose Hessian (analytic) | `d2ee = rbd.end_effector_pose_hessian_analytic(q, offsets=None, ee_joint_names=None)` — closed-form second derivatives |
| General-frame geometric Jacobian | `J = rbd.frame_jacobian(q, frame_name, reference_frame)` (`LOCAL`/`WORLD`/`LOCAL_WORLD_ALIGNED`) |
| Frame Jacobian time-variation (J̇) | `Jdot = rbd.frame_jacobian_dot(q, qd, frame_name, reference_frame)` |
| Operational-space (OSC) inertia | `Lambda = rbd.osc_inertia(q)` = `(J·M⁻¹·Jᵀ)⁻¹` |

### Energy / centroidal / regressors

| Algorithm | Signature |
|---|---|
| Generalized gravity / nonlinear effects | `g = rbd.generalized_gravity(q, GRAVITY=-9.81)`, `c = rbd.nonlinear_effects(q, qd, GRAVITY=-9.81)` |
| Kinetic / potential / mechanical energy | `rbd.kinetic_energy(q, qd)`, `rbd.potential_energy(q, GRAVITY=-9.81)`, `rbd.mechanical_energy(...)` |
| Coriolis matrix `C(q,q̇)` | `C = rbd.coriolis_matrix(q, qd)` (with `C·q̇ + g = nonlinear_effects`) |
| CoM + CoM Jacobian | `p_com = rbd.com(q)` returns `(3,)`; `J_com = rbd.jacobian_com(q)` returns `(3, nv)` |
| CCRBA / centroidal momentum | `(A, h) = rbd.ccrba(q, qd)`, `rbd.centroidal_momentum(q, qd)` |
| dCCRBA (∂A/∂q tensor) | `dA = rbd.dccrba(q)` (analytic; the finite-difference variants `dccrba_fd` / `cmm_time_variation_fd` are retained as cross-checks) |
| CMM time variation (Ȧ) | `Adot = rbd.cmm_time_variation(q, qd)` = `Σ_i (∂A/∂q_i)·q̇_i` |
| Centroidal-momentum rate (ḣ) | `hdot = rbd.centroidal_momentum_time_variation(q, qd, qdd)` = `A·q̈ + Ȧ·q̇` (matches `pin.computeCentroidalMomentumTimeVariation`) |
| Centroidal dynamics derivatives | `(dh_dq, dhdot_dq, dhdot_dv, dhdot_da) = rbd.centroidal_dynamics_derivatives(q, qd, qdd)` (matches `pin.computeCentroidalDynamicsDerivatives`) |
| Inverse-dynamics regressor | `Y = rbd.inverse_dynamics_regressor(q, qd, qdd=None)` (`τ = Y·π`) |
| Regressor gradient (dY/dx) | `dY_dx = rbd.inverse_dynamics_regressor_gradient(q, qd, qdd)` with `dY_dx[c]·π == ∂τ/∂x[:,c]` |
| ∂q̈/∂π (inertial-parameter gradient) | `dqdd_dpi = rbd.forward_dynamics_parameter_gradient(q, qd, u)` = `−M⁻¹·Y` |
| Kinetic / potential energy regressors | `rbd.kinetic_energy_regressor(q, qd)`, `rbd.potential_energy_regressor(q, GRAVITY=-9.81)` (`E = y·π`, length `10·NB`) |
| Plant / cost / barrier reference | `rbd.plant_step(...)` (+ gradient / hessian), quadratic state/input costs, `ee_pos_cost`, `com_cost`, `momentum_cost`, joint position/velocity/torque log-barriers |

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
| `idsva_so_body_frame` | Body-frame propagation, inertia, and motion subspaces. | Default selected by `idsva_so` for fixed-base robots. |
| `idsva_so_world_frame` | World-frame propagation, with gravity in the main sweep. | Default selected by `idsva_so` for floating-base robots. |

These are CPU reference implementations, not GPU performance claims. The
dispatch above is an implementation choice, not a universal speed ranking.
For measured accelerator performance, use GRiD's separately versioned
benchmark results and their stated hardware and timing boundaries.

### Per-pass helpers

Many algorithms also expose their internal passes (e.g. `inverse_dynamics_fpass`,
`inverse_dynamics_bpass`, `minv_bpass`, `minv_fpass`,
`inverse_dynamics_gradient_fpass_dq` / `_dqd`,
`inverse_dynamics_gradient_bpass_dq` / `_dqd`) for unit-testing accelerator port pieces
independently. See `RBDReference.py` and the `_plant.py`, `_centroidal.py`,
`_energy.py`, and `_regressor.py` mixins for full signatures and returns.

## Joint-type support

* **Scalar joints** — `revolute`, `continuous`, and `prismatic`; fixed joints
  are merged by the parser. Floating roots use a six-dimensional tangent.
* **Helical (screw) joints** — supported natively, following Pinocchio's
  `JointModelHelical` pitch convention.
* **Planar and translation joints** — decomposed at parse time into their
  cardinal sub-joints, so downstream algorithms only ever see cardinal joints.
* **Spherical joints** — a native 3-DoF quaternion joint (so `NQ != NV` for
  models containing one).
* **Skew (non-cardinal) axes** — handled through dense motion subspaces.
* **Mimic joints** — folded into their target joint's reduced coordinate;
  chained mimics are flattened at resolve time.

Representative cases are validated against Pinocchio in `tests/`
(`test_spherical_joint_equivalence`, `test_helical_joint_equivalence`,
`test_mimic_chain_equivalence`, ...).

## Floating base

The Pinocchio free-flyer convention is native at the API boundary:
`q = [x, y, z, qx, qy, qz, qw, ...]` (quaternion **xyzw**), and the base
tangent is ordered `[linear; angular]` (`[vx, vy, vz, wx, wy, wz]`). A
floating-base model without additional spherical joints has `NQ = NV + 1`;
each additional spherical joint contributes another quaternion coordinate.
Velocity and force inputs are tangent-width (`NV`). The parser also provides
a legacy convention; new integrations should use its default `pinocchio`
convention explicitly.

## Installation

Two dependency tiers:

* **Base (runtime)** — the pure-Python reference. Only `numpy` + `sympy`:
  ```shell
  pip install -r requirements.txt
  ```
  Building a `robot` object also requires
  [URDFParser](https://github.com/A2R-Lab/URDFParser) (a sibling
  package, not on PyPI).

* **Developer / equivalence testing** — adds the Pinocchio backend and the
  test suite (`pin`, `robot_descriptions`, `xacrodoc`, `beautifulsoup4`,
  `pybind11`, `scipy`, `pytest`):
  ```shell
  pip install -r requirements-dev.txt
  ```

## Sibling repos / standalone use

This package is consumed both as a GRiD submodule and standalone. Either way,
`RBDReference` and [URDFParser](https://github.com/A2R-Lab/URDFParser) are
**siblings**: the checkout directories must be named exactly `RBDReference`
and `URDFParser`, side by side under a common parent that is on `sys.path`
(the package's absolute imports are `RBDReference.*`; running pytest from
that parent provides this automatically). Install both source checkouts from
`A2R-Lab` using `main`. Installing the requirements does not install these
source packages into arbitrary Python environments; add their common parent
to `PYTHONPATH` when running elsewhere.

The Pinocchio pins in `requirements-dev.txt` are **load-bearing**:

* `pin<4` — pin 4.x drops `pinocchio.pc` **and** restructures the C++
  headers, which breaks the `pin_so_ext` build;
* `cmeel-eigen` — provides Eigen headers + `eigen3.pc` on boxes without a
  system `libeigen3-dev`;
* `cmeel-urdfdom<5` — pin 3.9's pywrap links `liburdfdom_*.so.4`; a newer
  urdfdom wheel makes `import pinocchio` fail.

## Equivalence testing

The suite combines Pinocchio equivalence, finite-difference cross-checks,
analytical solutions, and independent convergence tests. Coverage and supported
configurations are defined by the tests, not an assertion that every possible
combination has been verified. That machinery lives **inside this package**:

* `equivalents/` — the reusable, shared-interface layer. Two interchangeable
  backends expose the *identical* adapter API:
  * `reference` — the Python `RBDReference` with the sibling parser (the
    adapter also uses the XML-parsing developer dependencies);
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
  this one argument. Adapter methods normalize return layouts; the raw
  `RBDReference` class may return additional per-pass intermediates.

  `equivalents/` also contains `mujoco_convention.py` (documented in
  `mujoco_convention.md`) — the MuJoCo-convention adapter layer. It is a
  **convention mapping** over the existing backends, not a third backend:
  `SUPPORTED_BACKENDS` stays `("reference", "pinocchio")`.

* `tests/` — this package's own suite, asserting the two backends agree. Run
  (from the common parent of the two checkouts, `external/` inside GRiD):
  ```shell
  python -m pytest RBDReference/tests/ -q
  ```

The `pin_so_ext` binding wraps `pinocchio::ComputeRNEASecondOrderDerivatives`
(Pinocchio 3.x ships the C++ but does not expose it to Python); its loader
builds it on first use — see `equivalents/pin_so_ext/` and `tests/README.md`
for the build details. Benchmark harnesses and further install tooling live
in the parent GRiD repo (this repo is also consumed standalone).
