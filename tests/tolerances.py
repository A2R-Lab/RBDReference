from dataclasses import dataclass


@dataclass(frozen=True)
class Tolerance:
    rtol: float
    atol: float
    note: str = ""


DEFAULT_TOLERANCE = Tolerance(
    rtol=1e-7,
    atol=1e-9,
    note="Default tolerance for CPU-side reference comparisons against Pinocchio.",
)


ALGORITHM_TOLERANCES = {
    "rnea": DEFAULT_TOLERANCE,
    "minv": DEFAULT_TOLERANCE,
    "aba": DEFAULT_TOLERANCE,
    "pose_gradient": Tolerance(
        rtol=1e-6,
        atol=1e-8,
        note="End-effector pose gradients are compared against Pinocchio using a mix of analytic and finite-difference paths, so they use a slightly wider absolute tolerance than the primary dynamics checks.",
    ),
    "pose_hessian": Tolerance(
        rtol=1e-5,
        atol=5e-4,
        note="End-effector pose Hessians compare the analytic GRiD path against a finite-difference Pinocchio reference, so they use a wider tolerance than the first-order dynamics checks.",
    ),
    "idsva_so_body_frame": Tolerance(
        rtol=1e-4,
        atol=1e-5,
        note="Second-order inverse-dynamics tensors are validated against finite differences of already-verified first-order quantities, so they use a wider tolerance than the primary first-order dynamics checks.",
    ),
    "energy": Tolerance(
        rtol=1e-6,
        atol=1e-7,
        note="Energy / generalized-gravity / Coriolis reference oracles compose the project RNEA / CRBA and are compared against Pinocchio's C++ implementations; cross-library float64 round-off in the velocity-product terms is slightly larger than the primary RNEA bucket, so the absolute floor is one decade wider.",
    ),
    "centroidal": Tolerance(
        rtol=1e-6,
        atol=1e-7,
        note="CoM / CMM / centroidal-momentum reference oracles compose the project kinematics + spatial inertias and are compared against Pinocchio's ccrba / centerOfMass; the mass-weighted world-frame accumulation carries cross-library round-off a decade above the primary RNEA bucket.",
    ),
    "centroidal_grad": Tolerance(
        rtol=1e-3,
        atol=1e-4,
        note="The C2 centroidal dynamics derivatives (dh_dq / dhdot_dq / dhdot_dv) are central finite differences of the exact value layer against Pinocchio's analytic getCentroidalDynamicsDerivatives. dh_dq is a single FD (~1e-8); dhdot_dq / dhdot_dv compose a second nested FD for the Adot-qd bias, so they carry ~1e-4 absolute FD truncation/round-off, a few decades above the exact value-layer bucket. dhdot_da == A is exact and is checked under the tight 'centroidal' bucket.",
    ),
    "regressor": Tolerance(
        rtol=1e-6,
        atol=1e-7,
        note="The joint-torque regressor reproduces Pinocchio's computeJointTorqueRegressor after the per-link basis permutation; residuals are float64 round-off scaled by the (large) inertia-parameter magnitudes.",
    ),
    "second_order_fdsva": Tolerance(
        rtol=1e-4,
        atol=1e-5,
        note="Second-order forward-dynamics tensors are validated against finite differences of already-verified first-order quantities, so they use a wider tolerance than the primary first-order dynamics checks.",
    ),
    "f_ext_grad": Tolerance(
        rtol=1e-6,
        atol=1e-7,
        note="First-order f_ext gradients (-J^T, M^-1 J^T) are exact against pinocchio's RNEA-with-unit-fext response, so they use the primary first-order bucket.",
    ),
    "f_ext_grad_so": Tolerance(
        rtol=1e-4,
        atol=1e-5,
        note="The mixed second-order f_ext gradient -dJ^T/dq is validated against finite differences of the exact first-order -J^T, so it uses a wider tolerance like the other FD-of-first-order second-order checks.",
    ),
}

ROBOT_ALGORITHM_TOLERANCES = {
    ("iiwa14", "pose_hessian"): Tolerance(
        rtol=1e-5,
        atol=3e-2,
        note="Floating-base iiwa14 pose Hessians now use the analytic free-flyer path and agree with the Pinocchio reference finite-difference check to within a few hundredths on the root-root block.",
    ),
    ("g1", "rnea"): Tolerance(
        rtol=1e-6,
        atol=3e-6,
        note="G1 fixed-base and floating-base dynamics, ABA, and derivative comparisons show stable agreement against Pinocchio at the low-micro scale, with floating gradients needing a slightly wider absolute tolerance.",
    ),
    ("g1", "minv"): Tolerance(
        rtol=1e-6,
        atol=1e-7,
        note="G1 inverse-mass and CRBA comparisons need a slightly wider absolute tolerance than the smaller smoke robots.",
    ),
    ("g1", "aba"): Tolerance(
        rtol=1e-6,
        atol=2e-5,
        note="G1 ABA is a round-trip check (tau=rnea(qdd) then aba(tau)). The GRiD reference inverts its own RNEA to ~1e-12; the residual is entirely Pinocchio-side cross-library round-off in the velocity-product terms, which scales with |qd|^2 and reaches ~1.7e-5 on the high-velocity samples (qd up to 10) where the qdd magnitude is too small for the comparator's scale-floor to cover. A structural error would be O(|qdd|), orders of magnitude larger.",
    ),
    ("g1", "fd"): Tolerance(
        rtol=1e-6,
        atol=5e-1,
        note="G1 floating-base FD-parameter-gradient (dqdd/dpi = -Minv . Y(q,qd,qdd_actual)) float32-CUDA equivalence. qdd_actual = Minv.(u-c) and |Minv| reaches ~3.2e3 at the zero/static sample, so the float32 round-off in that inner mass-matrix product is ~eps_f32*|Minv|*|c| ~ 0.1; that perturbed qdd then flows through Y(qdd) and the final -Minv.Y contraction, leaving an ABSOLUTE residual ~0.2 at the static sample (where the float64 reference cancels to ~0 and the output-scale headroom is therefore blind to it). This is a float32 conditioning floor identical on the in-smem PERF path and the g1 spilled-s_Y path (the high-energy samples agree to ~1e-6 RELATIVE, confirming the spilled placement is numerically identical). A genuine structural error would be O(|dqdd/dpi|) ~ 3e5, six orders of magnitude larger. Same cond(M) rationale as the h1_2-minv bucket; the atol floor is only consulted via the comparator's atol+5e-3*scale formula, so the high-magnitude entries stay governed by the relative headroom.",
    ),
    ("baxter", "aba"): Tolerance(
        rtol=1e-7,
        atol=5e-9,
        note="Baxter fixed-base ABA agrees with Pinocchio to within a few nanounits; this narrowly scoped absolute tolerance avoids failing on near-zero residuals.",
    ),
    ("h1_2", "aba"): Tolerance(
        rtol=1e-3,
        atol=1e-2,
        note="h1_2 has 12 mimic joints whose reduced mass matrix is genuinely near-singular (min singular value ~4.3e-6 fixed-base, ~1.4e-5 floating-base; condition number ~7e5 / ~5e6). Project-side and Pinocchio-side reduced-model ABA agree algebraically but accumulate cond(M) * float64-epsilon noise (~1e-3 on the high-velocity samples, ~5e-5 on the zero-state sample). A structural error would be O(|qdd|) ~ 1e1, orders of magnitude larger.",
    ),
    ("h1_2", "rnea"): Tolerance(
        rtol=1e-3,
        atol=1e-2,
        note="h1_2's reduced-model `forward_dynamics` (used in the equivalence tests under the `rnea` tolerance bucket) inherits the same cond(M)~7e5 noise floor as ABA: project-side and Pinocchio-side both compute qdd = Minv * (tau - bias) but their float64 Minv differs at the 1e-9 level which the conditioning amplifies to ~1e-3. The check still catches algorithmic divergence (which would be O(|qdd|) ~ 1e1).",
    ),
    ("h1_2", "minv"): Tolerance(
        rtol=1e-6,
        atol=1e-2,
        note="h1_2's reduced mass matrix has cond(M) ~7e5 (fixed) / ~5e6 (floating) due to the genuinely small minimum singular value (~4e-6 / ~1.4e-5). Inverting that matrix amplifies the ~1e-9 cross-library round-off in CRBA to ~1e-2 on the largest Minv entries (scale ~6e4). Relative error stays at ~1e-7 (machine epsilon * cond), which is the floor; a structural bug would scale with the matrix magnitude.",
    ),
    # ----- gen3 continuous-joint cross-library round-off bucket -----
    # gen3 has 4 continuous joints. Pinocchio encodes each as an RUBZ SO(2)
    # (cos/sin, NQ=2) slot while GRiD stores a raw scalar angle (NQ=1). The
    # dynamics/kinematics OUTPUTS are numerically identical (they depend on the
    # angle only through cos/sin); the residual is pure cross-library float64
    # round-off that SCALES WITH SAMPLE ENERGY (~1e-8 at the zero state, up to
    # ~2.4e-4 relative-to-array-scale on the highest-velocity/acceleration
    # samples) and is amplified through M^-1 in the ABA / forward-dynamics /
    # integrator paths. A STRUCTURAL error (a wrong continuous-joint model)
    # would be O(scale), orders of magnitude larger, so these overrides do not
    # mask real bugs. Measured worst-case relative residuals (fixed / floating):
    #   rnea ~1.6e-5 / ~2.7e-5,  minv ~9e-7,  aba ~1.4e-4 / ~3e-6,
    #   fwd-dyn-grad ~2.4e-4 / ~6e-6 (rnea bucket), pose_grad ~1.6e-5,
    #   integrator v ~1.4e-4 (rnea bucket), f_ext dtau ~1.1e-5,
    #   second-order FDSVA ~2.4e-4.
    ("gen3", "rnea"): Tolerance(
        rtol=5e-4,
        atol=5e-5,
        note="gen3 continuous-joint RUBZ cross-library round-off; this bucket also fronts the kinematics-pose, forward-dynamics-gradient and integrator equivalence checks (worst ~2.4e-4 relative on the high-energy samples). The integrator q-residual is compared against EXACT ZEROS (scale=0, so only atol applies) and carries continuous-joint tangent-space round-off up to ~2e-5 on the semi_implicit_euler step (q += dt*(v + dt*qdd) folds in the velocity update's round-off), so the absolute floor is 5e-5 (a structural integration error would be O(dt*|qdd|), orders of magnitude larger). See the gen3 round-off block comment.",
    ),
    ("gen3", "minv"): Tolerance(
        rtol=1e-5,
        atol=1e-7,
        note="gen3 continuous-joint round-off reaches ~9e-7 relative on CRBA / Minv; one decade wider than the default 1e-7. See the gen3 round-off block comment.",
    ),
    ("gen3", "aba"): Tolerance(
        rtol=1e-3,
        atol=4e-1,
        note="gen3 ABA is a ROUND-TRIP check (tau=rnea(qdd) then aba(tau)). Both libraries' ABA are exact: GRiD's aba(GRiD-rnea(qdd)) recovers qdd to ~4e-13 (fixed) / ~1e-11 (floating), and pin's aba(pin-rnea(qdd)) to ~5e-13. The residual is ENTIRELY the cross-library RNEA round-off in tau -- gen3's 4 continuous joints make GRiD-tau and pin-tau differ by ~3e-3 (RUBZ cos/sin vs raw-angle) -- amplified through M^-1. The FIXED-base mass matrix is well conditioned so the amplified residual stays ~1.6e-2; the synthetic FLOATING-base config (a fixed-base arm mounted on a free-flyer) has cond(M) ~6e4 (min singular value ~1.8e-4), which amplifies the ~3e-3 tau noise to ~0.33 absolute at the structurally-near-zero qdd entries. The wide absolute floor covers this cond-amplified cross-library noise (same treatment as h1_2's near-singular reduced-model aba/rnea buckets); a genuine algorithmic error would be O(|qdd|) on the well-conditioned directions and is still caught by the 1e-3 relative tolerance. See the gen3 round-off block comment.",
    ),
    ("gen3", "pose_gradient"): Tolerance(
        rtol=1e-4,
        atol=1e-6,
        note="gen3 end-effector pose gradients inherit the continuous-joint round-off (~1.6e-5 relative) on top of the analytic/finite-difference mix the default pose_gradient bucket already allows. See the gen3 round-off block comment.",
    ),
    ("gen3", "second_order_fdsva"): Tolerance(
        rtol=1e-3,
        atol=1e-4,
        note="gen3 second-order FDSVA tensors compound the continuous-joint round-off (~2.4e-4 relative) with the FD-of-first-order step error, so the gen3 floors are one decade wider. See the gen3 round-off block comment.",
    ),
    ("fetch", "rnea"): Tolerance(
        rtol=1e-6,
        atol=1e-8,
        note="Fetch floating-base inverse dynamics reaches single-digit nanounit residuals on near-zero entries; this narrow override avoids spurious failures without loosening the suite globally.",
    ),
    ("fetch", "aba"): Tolerance(
        rtol=1e-7,
        atol=1e-8,
        note="Fetch floating-base ABA reaches single-digit nanounit residuals on near-zero entries; this narrow override avoids spurious failures without loosening the suite globally.",
    ),
    ("rizon4", "rnea"): Tolerance(
        rtol=1e-7,
        atol=2e-9,
        note="Rizon4 fixed-base pose and dynamics checks stay at nanounit residual scale; this narrow absolute tolerance covers tiny frame-placement differences.",
    ),
    ("gen3", "energy"): Tolerance(
        rtol=1e-4,
        atol=1e-3,
        note="Gen3's nonlinear-effects / gravity composition diverges from Pinocchio at ~1.7e-5 relative on the high-energy samples (cross-library float64 round-off in the velocity-product terms, scaling with |qd|^2); the small zero-state gravity scale (~1.5e-3) also leaves a ~1.3e-8 residual above the default 1e-8 floor. A structural error would be O(|tau|), orders of magnitude larger.",
    ),
    ("gen3", "centroidal"): Tolerance(
        rtol=1e-4,
        atol=1e-4,
        note="Gen3 centroidal quantities inherit the same ~1.7e-5 relative cross-library round-off as its dynamics (the project world-frame accumulation vs Pinocchio's ccrba); a structural error would be O(scale), orders of magnitude larger.",
    ),
    ("gen3", "regressor"): Tolerance(
        rtol=1e-4,
        atol=1e-2,
        note="Gen3's joint-torque regressor matches Pinocchio at ~1.1e-5 relative; the absolute residual reaches ~4e-3 because the regressor entries carry the (large) inertia x acceleration magnitudes, so the absolute floor is set to the matching scale. A structural basis/permutation error would be O(scale).",
    ),
    ("gen3", "f_ext_grad"): Tolerance(
        rtol=1e-4,
        atol=1e-5,
        note="Gen3 has continuous joints, which Pinocchio encodes as RUBZ (cos/sin) 2-D q-slots. The fixed-base -J^T unit-fext response leaks ~4e-6 cross-library round-off at the structurally-zero entries (project yields exact +/-0, pin's expanded model carries the round-off); the floating-base dtau component reaches ~1.1e-5 relative. The wider relative floor covers both; a structural error would be O(1). See the gen3 round-off block comment.",
    ),
    ("gen3", "f_ext_grad_so"): Tolerance(
        rtol=1e-4,
        atol=1e-4,
        note="Gen3's -dJ^T/dq (FD-of-exact-first-order) inherits both the FD step error and the continuous-joint cross-library round-off, so it uses a wider absolute floor than the default FD-of-first-order bucket.",
    ),
    ("g1", "energy"): Tolerance(
        rtol=1e-5,
        atol=1e-3,
        note="G1's mass matrix has entries up to ~1e4; the kinetic-energy quadratic form and the Coriolis/gravity terms inherit that scale, so cross-library round-off reaches ~1e-4 absolute. Relative residual stays at ~1e-7.",
    ),
    ("g1", "centroidal"): Tolerance(
        rtol=1e-5,
        atol=1e-4,
        note="G1's world-frame centroidal accumulation over a 35-DoF humanoid reaches ~1e-7..1e-8 residuals scaled by the large inertia magnitudes; a slightly wider absolute floor covers the high-energy samples.",
    ),
    ("g1", "regressor"): Tolerance(
        rtol=1e-5,
        atol=1e-4,
        note="G1's regressor columns carry the link inertia magnitudes (up to ~1e2..1e3) times the velocity-product terms, so cross-library round-off reaches ~1e-4 absolute on the high-velocity samples. Relative residual stays at ~1e-7.",
    ),
    ("h1_2", "energy"): Tolerance(
        rtol=1e-3,
        atol=1e-2,
        note="h1_2 (51 DoF, 12 mimic joints, near-singular reduced mass matrix) inherits the same cond(M)~7e5 noise floor as its ABA/RNEA buckets for the kinetic-energy / Coriolis composition.",
    ),
    ("h1_2", "centroidal"): Tolerance(
        rtol=1e-4,
        atol=1e-3,
        note="h1_2's mimic-folded centroidal map carries the reduced-model round-off (folded mimic columns scaled by the URDF multiplier) at the same scale as its CRBA bucket. The C2 centroidal-derivative blocks (dh_dq / dhdot_dq / dhdot_dv) are central finite differences of the exact value layer, adding ~1e-6 FD noise on top.",
    ),
    ("fr3", "centroidal"): Tolerance(
        rtol=1e-3,
        atol=1e-4,
        note="fr3 (mimic joints) C2 centroidal-derivative blocks (dh_dq / dhdot_dq / dhdot_dv) are central finite differences of the exact value layer; the FD carries ~1e-6 absolute noise that, on the near-zero derivative entries (mimic-folded columns), exceeds the primary 1e-6/1e-7 bucket's relative floor. The value layer (com/Jcom/A/h) still matches to the primary bucket; this override only covers the FD-sourced derivative blocks.",
    ),
    ("h1_2", "regressor"): Tolerance(
        rtol=1e-4,
        atol=1e-2,
        note="h1_2's regressor rows fold mimic v-slots into their target with the URDF multiplier; the reduced-row regressor inherits the cond(M)~7e5 amplified round-off of its dynamics buckets.",
    ),
    ("g1", "second_order_fdsva"): Tolerance(
        rtol=1e-4,
        atol=1e-2,
        note="G1's mass matrix has entries up to ~1e4, which amplifies the ~1e-7 idsva_so residual to ~1e-3 when composing fdsva via Minv multiplication. Relative norm stays at ~1e-7 (8 significant digits).",
    ),
    ("h1_2", "second_order_fdsva"): Tolerance(
        rtol=1e-4,
        atol=1e-1,
        note="H1-2 (51 DoF, 12 mimic joints, mass matrix entries up to ~1e5) has the same Minv-amplification issue as G1, exacerbated by higher dimensionality. The reduced CRBA from the project analytic path vs pinocchio's C++ CRBA diverge at ~1e-7 relative (float64 round-off through 51x51 inversion); composing fdsva via Minv@..@Minv compounds this to ~1e-4 relative max in the tensor entries we care about. Relative residual norms stay at ~1e-7 (7-8 significant digits).",
    ),
}

def get_tolerance(algorithm: str, robot_id: str | None = None) -> Tolerance:
    if robot_id is not None:
        override = ROBOT_ALGORITHM_TOLERANCES.get((robot_id, algorithm))
        if override is not None:
            return override
    return ALGORITHM_TOLERANCES.get(algorithm, DEFAULT_TOLERANCE)
