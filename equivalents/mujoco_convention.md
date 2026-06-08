# MuJoCo / mjx output convention — derivation, findings, and scope

Reference for `mujoco_convention.py` (the validated transforms) and for the GRiD
binding's planned `output_convention="mujoco"` flag. All claims here are validated
in `RBDReference/tests/test_mujoco_convention.py` (FD self-consistency, no MuJoCo
needed) and, where noted, directly against a real MuJoCo install (`mj_fullM`,
`mj_inverse`, `mj_forward`) on a minimal matched model — to **machine precision**.

GRiD is natively **pinocchio**-convention. MuJoCo differs for the free-floating
base in exactly two ways, and the second one is subtler than it first looks.

## The two raw differences

1. **Quaternion order.** MuJoCo `qpos` quaternion is **wxyz** (scalar first);
   pin/GRiD are **xyzw**. A pure relabel of the free-flyer quat on `q` (and on the
   `integrate`/`qnext` output). Never touches velocities, forces, or any gradient.
2. **Free-joint velocity frame.** MuJoCo `qvel` is `[v_lin GLOBAL ; omega LOCAL]`;
   pin/GRiD spatial twist is `[v_lin LOCAL ; omega LOCAL]`. One root-block basis
   change `G(q) = blockdiag(R, I_3)` on the leading 6 tangent DOF (`R` = base
   orientation). `G` is orthogonal, so `G^{-1}=G^T`, `G^{-T}=G`. Concretely `G` is
   the identity except its top-left 3×3 (base-linear) block, which is `R`.

## Value transforms (all confirmed vs real MuJoCo to ~1e-15)

| quantity | pin → mjx |
|---|---|
| `q` / `qnext` | reorder free-flyer quat xyzw → wxyz |
| `qd` (velocity) | `v_mjx = G v_pin` (rotate base-linear by `R`) |
| `u`/`tau` (force, covector) | `tau_mjx = G^{-T} tau_pin = G tau_pin` |
| `M` | `G^{-T} M G^{-1} = G M G^T` |
| `Minv` | `G Minv G^T` |
| **`qdd` (acceleration)** | **`a_mjx = G a_pin + Ġ v_pin`** — NOT a plain rotation (see below) |

### ⚠ The acceleration is not a rotation (key finding)

The mjx base-linear velocity is the **global position rate** `ṗ = R v_local`.
Differentiating, `a_mjx_lin = R(a_pin_lin + ω × v_local)`. The extra `ω × v` term
(`Ġ v_pin`, with `Ġ = blockdiag(R[ω]_×, 0)`) is what makes the inverse-dynamics
**Coriolis** term frame-dependent. The original design doc treated acceleration
like velocity (a pure `G` rotation) and so was silently wrong for any
velocity-dependent term: gravity and the `M·a` inertial term matched MuJoCo, but
the Coriolis term was off by O(1). Both the accel **input** conversion
(mjx→pin, before calling GRiD) and the accel **output** conversion (fd `qdd`,
pin→mjx) carry this term. With it, `mj_inverse`/`mj_forward` match to ~1e-14.

Fixed base ⇒ `G=I`, `Ġ=0` ⇒ every transform is the identity (the flag is a
provable no-op; the regression guard).

## First-order gradients — MuJoCo-native parity (validated, FD ~1e-7)

MuJoCo's `d/dqpos` derivatives **hold `qvel` and `qacc`/applied-force fixed in the
MJX frame**. Because those mjx-frame inputs are global-linear, perturbing the base
*orientation* rotates the **pin-frame** inputs GRiD actually consumed
(`v_pin = G^{-1} v_mjx` etc. track `R`). Confirmed empirically: holding qvel-mjx vs
body-velocity fixed changes `dtau/dq` only on the 3 base-rotation columns, by O(1).

So the mjx gradient is **not** the doc's simple `T(q)·grad_pin`. It is assembled
modularly (see `id_gradient_pin_to_mjx` / `fd_gradient_pin_to_mjx`):

```
dy_mjx/dξ = PrefDeriv(y_pin) + Tout · ( dy_pin/dq · Jq + dy_pin/dqd · Jv + dy_pin/d{a|u} · Ja )
```

* `Tout` — output value-map (`G` for tau; the accel-map for fd `qdd`).
* `PrefDeriv` — derivative of the output map wrt the base rotation (the "frame term").
* `Jq = G^{-1}` — the input-tangent column reframe (mjx base-linear tangent is
  global, pin's is local).
* `Jv`, `Ja` — Jacobians of the (validated) mjx→pin **input** conversions wrt the
  perturbed quantity; nonzero only on base-linear/rotation columns. These are the
  Coriolis (`dy/dqd`) and inertial/force (`M` or `Minv`) couplings, **plus** the
  `ω×v` acceleration-coupling cross terms.

Localization holds: only base-linear and base-rotation columns differ from a plain
transport; all internal-joint columns are pure transport. **Extra quantities the
boundary needs:** the matching first-order `d/dqd` gradient (already returned with
`d/dq`), the mass matrix `M` / its inverse `Minv`, and the pin-frame inputs. No
codegen/kernel change — pure binding-side.

## Second order (idsva_so) — DONE + validated

`idsva_so` returns `(d²τ/dq², d²τ/dqd², d²τ/dqd∂q, dM/dq)`. All four are transformed
to the mjx frame by `second_order_id_pin_to_mjx`, validated to FD precision vs FD of
the first-order mjx gradients and, on a matched model, vs **FD-of-FD of `mj_inverse`**
(~8e-6 relative, the double-FD floor).

**Key simplification — no retract-curvature term.** I initially expected the genuine
second derivatives to need a connection/curvature correction (MuJoCo's global-linear
retract vs pin's SE(3) retract differ at second order). They do **not**, because the
mjx SO tensor is the **single** derivative of the *already-MuJoCo-correct* analytic
first-order gradient. Differentiating an exact first-order quantity once more needs
only the **first-order** sensitivities of that gradient's inputs — the second-order
retract behaviour never enters (we differentiate a function whose value already
matches MuJoCo, we do not compose two retract steps).

Implementation: `d²τ/dq²`, the cross tensor, and `d²τ/dqd²` are obtained by
**complex-step differentiating** `id_gradient_pin_to_mjx` along the mjx perturbation,
feeding it the pin SO tensors contracted with the input-conversion Jacobians + the
frame derivative. Complex-step is exact and sidesteps hand-expanding the messy
acceleration-coupling derivative. `dM/dq` (a first-order quantity) uses its clean
closed form. For jax/torch the same result comes for free by autodiffing the value
transform twice.

**`fdsva_so`** (forward-dynamics SO) follows the identical pattern — complex-step
`fd_gradient_pin_to_mjx` — and is the one remaining mechanical extension (it also
needs `Minv` and the `dM/dq`/SO plumbing that `RBDReference.fdsva_so` already composes).

## Performance cost & the "native-frame" question

The transform is a **boundary post/pre-process**, not a kernel change:

* Values: a few 3×3 mat-vecs (`tau`, `qdd`) or two `nv×nv` mat-mats with a matrix
  that is identity except a 3×3 block (`M`, `Minv`). O(nv²) at most, vs the kernel's
  O(nv²–nv³) dynamics — **sub-1% overhead**, and **exactly zero** on fixed base
  (the `G=I` branch returns the inputs untouched, byte-identical).
* First-order gradients: the modular assembly is a handful of `nv×nv` mat-mats on
  the host. Still small next to the GPU gradient kernel, but larger than the value
  path; for tight loops it can be fused.
* The couplings touch only the 6 base-tangent rows/cols, so a tuned implementation
  is essentially O(nv) added work for the corrections + the transport mat-mats.

**Do we ever need a native-mjx-frame kernel?** Not for correctness, and almost
certainly not for performance: the boundary transform is asymptotically dominated by
the dynamics kernel for every robot of interest, and it is a perfect no-op for
fixed-base (the common case). A native-frame reimplementation would duplicate every
algorithm for a sub-1% value-path saving and a modest gradient saving, against the
maintenance cost of a second convention threaded through the codegen — not worth it.
The one scenario that could change the calculus is an all-floating, gradient-heavy
workload where the host-side gradient assembly becomes measurable; even then,
fusing the assembly into the existing post-process (or a small dedicated kernel for
just the 6-DOF base corner) is far cheaper than a frame-native rewrite. Recommend:
ship the boundary transform; measure the gradient-path overhead on a real
floating + gradient workload before considering anything kernel-side.

## Binding implementation plan (`output_convention` flag)

Pure binding-side, default `"pinocchio"` (existing behaviour byte-identical):

1. `_handle.py`: add `output_convention: "pinocchio"|"mujoco"`. For `"mujoco"` +
   floating base, mirror the `mujoco_convention.py` helpers (the binding cannot
   import RBDReference): reorder the free-flyer quat on `q`-in / `qnext`-out, apply
   `G^{-1}` to the `qd` input and the `ω×v` accel transform to any `qdd` input,
   and the `M`/`Minv`/`tau`/`qdd` output transforms. `R` comes from the `q` input.
   Fixed base ⇒ the `G=I` path (no-op, assert byte-identical).
2. jax / torch wrappers: same transform, kept inside the traced/autograd path so
   it stays differentiable; keep numpy/jax/torch identical.
3. Gradients (`dtau`, `dqdd`): apply the modular first-order transform; the wrapper
   must also obtain the matching VALUE output and `M`/`Minv` for the couplings.
4. SO: guard `output_convention="mujoco"` on the SO entry points until the closed
   form lands (raise/clear note), per the scope above.

Validation before release: build a floating robot's kernels and diff the binding's
mujoco outputs against `mujoco_convention.py` on identical states, plus the
fixed-base byte-identical no-op check.
```
