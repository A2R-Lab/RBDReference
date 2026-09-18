# RBDReference

A Python reference implementation of rigid body dynamics algorithms.

This package is designed to enable rapid prototyping and testing of new
algorithms and algorithmic optimizations. The CUDA / FPGA / accelerator
implementations in the parent GRiD repo use it as a golden CPU oracle
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

Lie-group state operations (floating-base `q` lives on SE(3)×ℝⁿ, so these are
not plain vector ops; for fixed-base they collapse to the familiar `+`/`−`):

| Algorithm | Signature |
|---|---|
| Retract (q ⊕ v·dt) | `q_new = rbd.integrate(q, v_dt)` (matches `pin.integrate`) |
| Tangent Jacobians of retract | `J = rbd.dIntegrate(q, v_dt, with_respect_to)` (`'q'`/`'v'`, nv×nv) |
| Second-order retract derivative | `H = rbd.d2Integrate(q, v_dt, arg1, arg2)` (nv×nv×nv tangent derivative of `dIntegrate`) |
| Boxminus (q_to ⊖ q_from) | `v = rbd.difference(q_from, q_to)` (matches `pin.difference`) |
| Tangent Jacobians of difference | `J = rbd.dDifference(q_from, q_to, with_respect_to)` (`'from'`/`'to'`) |
| One integration step | `x_kp1 = rbd.integrator(q, qd, u, dt, integrator_type="euler")` (`euler`/`semi_implicit_euler`/`trapezoidal`/`midpoint`/`rk3`/`rk4`) |
| Integrator Jacobian | `AB = rbd.integrator_gradient(q, qd, u, dt, ...)` — `[A | B]` of shape (2·nv, 3·nv), tangent-space `[d/dq | d/dqd | d/du]` |
| Tangent-space quadratic state cost | `rbd.quadratic_state_cost_tangent(x, x_des, Q)` — log-map error `[difference(q_des,q); qd−qd_des]`, diagonal `Q` of size 2·nv |

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
| CoM + CoM Jacobian | `(p_com, J_com) = rbd.com(q)`, `rbd.jacobian_com(q)` |
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
| `idsva_so_body_frame` | Body-frame propagation, body-frame inertia, body-frame motion subspace. Multi-pass forward/backward sweeps. | **Fixed-base** robots — wins by a wide margin (e.g. iiwa14 fixed: 27 µs vs 827 µs on GPU). |
| `idsva_so_world_frame` | World-frame propagation, world-frame motion subspace, gravity baked into the main sweep. Single-pass reference (closer to the textbook spatial-vector-algebra derivation). | **Floating-base** robots — wins by 2–4× (e.g. iiwa14_floating 1.7×, g1_floating 3.6×, GPU). |

GPU benchmarks above are from the **parent GRiD repo's** benchmark harness
(under its `test/benchmarks/`; this repo is also consumed standalone) on
sm_120 (RTX 5090). The crossover is purely a function of which kinematic
chain depth dominates: body-frame's subtree-broadcast pays off when the
chain is short and the tree is fixed; world-frame's single-pass cost is
flat in chain depth which wins as DOF grows under floating base.

### Per-pass helpers

Many algorithms also expose their internal passes (e.g. `inverse_dynamics_fpass`,
`inverse_dynamics_bpass`, `minv_bpass`, `minv_fpass`,
`inverse_dynamics_gradient_fpass_dq` / `_dqd`,
`inverse_dynamics_gradient_bpass_dq` / `_dqd`) for unit-testing accelerator port pieces
independently. See `RBDReference/RBDReference.py` for full signatures.

## Joint-type support

* **Tier-A cardinal joints** — `revolute`, `continuous`, `prismatic`, `fixed`,
  and `floating` on cardinal (±x/±y/±z) axes: the fully optimized core path.
* **Helical (screw) joints** — supported natively, following Pinocchio's
  `JointModelHelical` pitch convention.
* **Planar and translation joints** — decomposed at parse time into their
  cardinal sub-joints, so downstream algorithms only ever see cardinal joints.
* **Spherical joints** — a native 3-DoF quaternion joint (so `NQ != NV` for
  models containing one).
* **Skew (non-cardinal) axes** — handled through the dense-motion-subspace
  Tier-B path.
* **Mimic joints** — folded into their target joint's reduced coordinate;
  chained mimics are flattened at resolve time.

Each of these is validated against Pinocchio in `tests/`
(`test_spherical_joint_equivalence`, `test_helical_joint_equivalence`,
`test_mimic_chain_equivalence`, ...).

## Floating base

The Pinocchio free-flyer convention is native at the API boundary:
`q = [x, y, z, qx, qy, qz, qw, ...]` (quaternion **xyzw**), and the base
tangent is ordered `[linear; angular]` (`[vx, vy, vz, wx, wy, wz]`). A
floating-base model has `NQ = NV + 1`; velocity and force inputs are
tangent-width (`NV`).

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
  test suite (`pin`, `robot_descriptions`, `xacrodoc`, `beautifulsoup4`,
  `pybind11`, `pytest`):
  ```shell
  pip install -r requirements-dev.txt
  ```

## Sibling repos / standalone use

This package is consumed both as a GRiD submodule and standalone. Either way,
`RBDReference` and [URDFParser](https://github.com/A2R-Lab/URDFParser) are
**siblings**: the checkout directories must be named exactly `RBDReference`
and `URDFParser`, side by side under a common parent that is on `sys.path`
(the package's absolute imports are `RBDReference.*`; running pytest from
that parent provides this automatically). `RBDReference` currently needs
`URDFParser` on its `modernizing-tests` branch.

The Pinocchio pins in `requirements-dev.txt` are **load-bearing**:

* `pin<4` — pin 4.x drops `pinocchio.pc` **and** restructures the C++
  headers, which breaks the `pin_so_ext` build;
* `cmeel-eigen` — provides Eigen headers + `eigen3.pc` on boxes without a
  system `libeigen3-dev`;
* `cmeel-urdfdom<5` — pin 3.9's pywrap links `liburdfdom_*.so.4`; a newer
  urdfdom wheel makes `import pinocchio` fail.

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

  `equivalents/` also contains `mujoco_convention.py` (documented in
  `mujoco_convention.md`) — the MuJoCo-convention adapter layer. It is a
  **convention mapping** over the existing backends, not a third backend:
  `SUPPORTED_BACKENDS` stays `("reference", "pinocchio")`.

* `tests/` — this package's own suite, asserting the two backends agree. Run
  (from the directory containing `RBDReference`, e.g. the GRiD repo root):
  ```shell
  pytest RBDReference/tests/
  ```

The `pin_so_ext` binding wraps `pinocchio::ComputeRNEASecondOrderDerivatives`
(Pinocchio 3.x ships the C++ but does not expose it to Python); its loader
builds it on first use — see `equivalents/pin_so_ext/` and `tests/README.md`
for the build details. Benchmark harnesses and further install tooling live
in the parent GRiD repo (this repo is also consumed standalone).
