# MuJoCo / MJX convention transforms

This note describes the CPU reference transforms in `mujoco_convention.py`.
It is not a benchmark or a promise of support on every GRiD binding. GRiD's
generated MuJoCo-convention kernels and bindings have their own implementation
and validation; consult that project's current support documentation.

The helpers map quantities between the Pinocchio free-flyer convention and
the MuJoCo free-joint convention. They are a convention layer over the
`reference` and `pinocchio` adapters, not a third dynamics backend. Model
inertias, joints, frames, forces, and gravity must still agree for a numerical
comparison to be meaningful.

## State and value conventions

Pinocchio uses an **xyzw** base quaternion and base velocity
`[linear LOCAL; angular LOCAL]`. MuJoCo uses **wxyz** and
`[linear WORLD; angular LOCAL]`. With base rotation `R`, define
`G = diag(R, I3, I_internal)` on the velocity space.

| Quantity | Pinocchio → MuJoCo |
|---|---|
| Configuration | Reorder the free-flyer quaternion xyzw → wxyz |
| Velocity | `v_mj = G v_pin` |
| Generalized force | `tau_mj = G tau_pin` |
| Mass matrix | `M_mj = G M_pin G.T` |
| Inverse mass matrix | `Minv_mj = G Minv_pin G.T` |
| Acceleration | `a_mj = G a_pin + Gdot v_pin` |

The acceleration correction is essential: its base-linear block is
`R (a_linear + omega × v_linear)`. A rotation alone gives incorrect
velocity-dependent dynamics. The inverse input conversion must remove this
term too. Helpers take an explicit `FloatingRootLayout`; the no-floating-root
case leaves these free-flyer transforms inactive. This is not a general
conversion of arbitrary spherical-joint layouts or unmatched robot models.

## Derivatives

Configuration columns are tangent perturbations, not derivatives of the raw
quaternion entries. MuJoCo-frame derivatives hold the other MuJoCo-frame
inputs fixed. Rotating a Pinocchio gradient's output is therefore insufficient:
the input velocity/acceleration/force maps and the output map also contribute
configuration derivatives.

`id_gradient_pin_to_mjx` and `fd_gradient_pin_to_mjx` implement these chain
rules. They need the matching value and derivative data, including `M` or
`Minv`, as specified by their docstrings.

`second_order_id_pin_to_mjx` and `second_order_fd_pin_to_mjx` both exist.
They transform the four inverse-/forward-dynamics second-order tensor blocks.
These are derivatives of the first-order tangent gradients, with index order
defined by the function docstrings; do not assume a raw-coordinate symmetric
Hessian. The implementation uses complex-step differentiation of the transform
expressions together with the supplied first-/second-order dynamics tensors.
Complex-step avoids subtractive cancellation, but still has floating-point
and step-size error; it is not an exact-arithmetic guarantee.

The module also includes transforms for centroidal derivatives, EE-pose
Hessians, tracking costs, and selected integrator sensitivities. These helpers
have their own scope restrictions; their presence does not establish support
for every integration scheme. Read the corresponding function docstring and
tests before composing a new path.

## Validation and performance scope

`tests/test_mujoco_convention.py` contains finite-difference consistency checks
and optional direct MuJoCo checks on a matched model. The direct checks require
MuJoCo; they skip when it is not installed. The normal CI dependency set does
not install it. No broad MuJoCo parity claim follows from a run that skipped
those checks.

The value maps operate on the base block, but reference matrix/derivative
implementations may still allocate or multiply full arrays. Their overhead
depends on the implementation, state size, and calling boundary. This note
makes no measured percentage-overhead claim and does not predict whether a
host transform, fused device transform, or native-convention kernel is fastest.
