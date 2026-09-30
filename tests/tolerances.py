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
    "inverse_dynamics": DEFAULT_TOLERANCE,
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
    "dccrba": Tolerance(
        rtol=1e-6,
        atol=1e-7,
        note="The ANALYTIC dCCRBA (Adot = dAg/dt and the dA_dq tensor, _dccrba_analytic: one world-frame sweep + spatial cross-product / inertia-derivative operators) matches pinocchio's exact pin.dccrba / computeCentroidalDynamicsDerivatives to ~1e-12 (relative ~1e-9 on the big humanoids where the world-frame inertia accumulation carries float64 round-off scaled by the large inertia magnitudes). The 4th-order-FD cross-check (dccrba_fd / cmm_time_variation_fd of the exact value layer) adds ~1e-10..1e-8 absolute FD truncation. This bucket is intentionally a few decades tighter than the nested-FD 'centroidal_grad' bucket — the analytic path has no nested FD. assert_close floors atol at rtol*max|expected|, so the per-element check scales with each robot's magnitude (no per-robot override needed for g1/h1_2).",
    ),
    "inverse_dynamics_regressor": Tolerance(
        rtol=1e-6,
        atol=1e-7,
        note="The joint-torque regressor reproduces Pinocchio's computeJointTorqueRegressor after the per-link basis permutation; residuals are float64 round-off scaled by the (large) inertia-parameter magnitudes.",
    ),
    "forward_dynamics_parameter_gradient": Tolerance(
        rtol=1e-6,
        atol=1e-7,
        note="The forward-dynamics inertial-parameter gradient dqdd/dpi = -Minv . Y(q,qd,qdd_actual) composes the already-verified Minv and joint-torque regressor against an independent Pinocchio oracle built from pin.computeMinverse/crba and pin.computeJointTorqueRegressor at qdd_actual = aba(q,qd,u). Float64 round-off scaled by the (large) inertia-parameter magnitudes, one decade above the primary first-order bucket on the conditioned robots.",
    ),
    "second_order_fdsva": Tolerance(
        rtol=1e-4,
        atol=1e-5,
        note="Second-order forward-dynamics tensors are validated against finite differences of already-verified first-order quantities, so they use a wider tolerance than the primary first-order dynamics checks.",
    ),
    "f_ext_gradient": Tolerance(
        rtol=1e-6,
        atol=1e-7,
        note="First-order f_ext gradients (-J^T, M^-1 J^T) are exact against pinocchio's RNEA-with-unit-fext response, so they use the primary first-order bucket.",
    ),
    "f_ext_gradient_so": Tolerance(
        rtol=1e-4,
        atol=1e-5,
        note="The mixed second-order f_ext gradient -dJ^T/dq is validated against finite differences of the exact first-order -J^T, so it uses a wider tolerance like the other FD-of-first-order second-order checks.",
    ),
}

ROBOT_ALGORITHM_TOLERANCES = {
    ("baxter", "aba"): Tolerance(
        rtol=1e-7,
        atol=5e-9,
        note="Baxter fixed-base ABA agrees with Pinocchio to within a few nanounits; this narrowly scoped absolute tolerance avoids failing on near-zero residuals.",
    ),
    ("h1_2", "inverse_dynamics"): Tolerance(
        rtol=3e-4,
        atol=3e-6,
        note="Re-measured 2026-09-30 (GRiD docs/agent_debugging_guide.md 7.z34): after the floating-mimic fixes RBDReference matches Pinocchio on h1_2 to ~1e-15 for aba/minv/crba and every other bucket passes the default, so the old cond(M)~5e6 overrides were deleted. This bucket keeps slack only for the fp64 RK4 integrator cells (h1_2-floating, dt=0.1, energetic samples): four forward-dynamics evaluations at extreme stage states differ from Pinocchio's by ~8e-5 of the default allowance x 1e3 (worst 846x the 1e-7/1e-9 default); ~3x headroom.",
    ),
    # ----- rizon4 healed-asset aba round-trip bucket (2026-09-17) -----
    ("rizon4", "aba"): Tolerance(
        rtol=1e-6,
        atol=5e-6,
        note="rizon4's ABA round-trip (tau=inverse_dynamics(qdd) then aba(tau)) first EXECUTED after the 2026-09-15 <inertial> heal (the flexiv values are ROUNDED: diagonal inertias 0.001-0.03). MEASURED 2026-09-17: worst cross-library residual 5.9e-7 abs (fixed) / 1.1e-6 (floating), landing on structurally-near-zero qdd entries (whole-array scale ~2e-8 on the low-energy sample) — cross-library RNEA tau round-off amplified through M^-1 (min singular value ~1e-3, cond ~5e3 fixed / 3e4 floating). Same treatment as the gen3 aba bucket: atol 5e-6 covers the measured noise with ~5x margin while a structural error would be O(|qdd|) ~ 10 and is still caught by the 1e-6 relative tolerance on well-conditioned directions.",
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
    #   inverse_dynamics ~1.6e-5 / ~2.7e-5,  minv ~9e-7,  aba ~1.4e-4 / ~3e-6,
    #   fwd-dyn-grad ~2.4e-4 / ~6e-6 (inverse_dynamics bucket), pose_grad ~1.6e-5,
    #   integrator v ~1.4e-4 (inverse_dynamics bucket), f_ext dtau ~1.1e-5,
    #   second-order FDSVA ~2.4e-4.
    ("gen3", "inverse_dynamics"): Tolerance(
        rtol=5e-4,
        atol=5e-5,
        note="gen3 continuous-joint RUBZ cross-library round-off; this bucket also fronts the kinematics-pose, forward-dynamics-gradient and integrator equivalence checks (worst ~2.4e-4 relative on the high-energy samples). The integrator q-residual is compared against EXACT ZEROS (scale=0, so only atol applies) and carries continuous-joint tangent-space round-off up to ~2e-5 on the semi_implicit_euler step (q += dt*(v + dt*qdd) folds in the velocity update's round-off), so the absolute floor is 5e-5 (a structural integration error would be O(dt*|qdd|), orders of magnitude larger). See the gen3 round-off block comment.",
    ),
    ("gen3", "inverse_dynamics_gradient"): Tolerance(
        rtol=5e-4,
        atol=5e-5,
        note="Same continuous-joint RUBZ round-off bucket as gen3 inverse_dynamics: the regressor-gradient pi-identity check (dY/dx . pi vs pin's analytic dtau/dx) measures ~1.6e-4 abs (rel-to-scale ~2e-6) — triangulated 2026-09-17 as the PRE-EXISTING project-vs-pin dtau/dq residual (the project's own analytic id_du differs from pin by the identical amount; the project's tau/id_du/Y/dY are mutually consistent to <5e-10). See the gen3 round-off block comment.",
    ),
    ("gen3", "minv"): Tolerance(
        rtol=1e-5,
        atol=1e-7,
        note="gen3 continuous-joint round-off reaches ~9e-7 relative on CRBA / Minv; one decade wider than the default 1e-7. See the gen3 round-off block comment.",
    ),
    ("gen3", "aba"): Tolerance(
        rtol=1e-3,
        atol=4e-1,
        note="gen3 ABA is a ROUND-TRIP check (tau=inverse_dynamics(qdd) then aba(tau)). Both libraries' ABA are exact: GRiD's aba(GRiD-inverse_dynamics(qdd)) recovers qdd to ~4e-13 (fixed) / ~1e-11 (floating), and pin's aba(pin-inverse_dynamics(qdd)) to ~5e-13. The residual is ENTIRELY the cross-library RNEA round-off in tau -- gen3's 4 continuous joints make GRiD-tau and pin-tau differ by ~3e-3 (RUBZ cos/sin vs raw-angle) -- amplified through M^-1. The FIXED-base mass matrix is well conditioned so the amplified residual stays ~1.6e-2; the synthetic FLOATING-base config (a fixed-base arm mounted on a free-flyer) has cond(M) ~6e4 (min singular value ~1.8e-4), which amplifies the ~3e-3 tau noise to ~0.33 absolute at the structurally-near-zero qdd entries. The wide absolute floor covers this cond-amplified cross-library noise; a genuine algorithmic error would be O(|qdd|) on the well-conditioned directions and is still caught by the 1e-3 relative tolerance. See the gen3 round-off block comment.",
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
    ("gen3", "inverse_dynamics_regressor"): Tolerance(
        rtol=1e-4,
        atol=1e-2,
        note="Gen3's joint-torque regressor matches Pinocchio at ~1.1e-5 relative; the absolute residual reaches ~4e-3 because the regressor entries carry the (large) inertia x acceleration magnitudes, so the absolute floor is set to the matching scale. A structural basis/permutation error would be O(scale).",
    ),
    ("gen3", "f_ext_gradient"): Tolerance(
        rtol=1e-4,
        atol=1e-5,
        note="Gen3 has continuous joints, which Pinocchio encodes as RUBZ (cos/sin) 2-D q-slots. The fixed-base -J^T unit-fext response leaks ~4e-6 cross-library round-off at the structurally-zero entries (project yields exact +/-0, pin's expanded model carries the round-off); the floating-base dtau component reaches ~1.1e-5 relative. The wider relative floor covers both; a structural error would be O(1). See the gen3 round-off block comment.",
    ),
    ("gen3", "forward_dynamics_parameter_gradient"): Tolerance(
        rtol=1e-3,
        atol=1e-1,
        note="gen3's 4 continuous joints (RUBZ cos/sin vs raw-angle) leave cross-library round-off in qdd_actual=aba(q,qd,u) and in Minv, amplified through the -Minv.Y compose (~2.4e-4 relative on high-energy samples, larger absolute on the floating cond~6e4 configuration). A structural continuous-joint error would be O(scale). See the gen3 round-off block comment.",
    ),
    ("h1_2", "second_order_fdsva"): Tolerance(
        rtol=1e-4,
        atol=3e-3,
        note="Re-measured 2026-09-30 (GRiD docs/agent_debugging_guide.md 7.z34): after the floating-mimic fixes RBDReference matches Pinocchio on h1_2 to ~1e-15 for aba/minv/crba and every other bucket passes the default, so the old cond(M)~5e6 overrides were deleted. fdsva_so composition: one near-zero entry of 91,125 (h1_2-floating) differs by 1.0e-3 absolute, above the rtol*scale floor (9.7e-4 at scale 9.7); atol 3e-3 gives ~3x headroom. The previous atol 0.1 is gone.",
    ),
}

def get_tolerance(algorithm: str, robot_id: str | None = None) -> Tolerance:
    if robot_id is not None:
        override = ROBOT_ALGORITHM_TOLERANCES.get((robot_id, algorithm))
        if override is not None:
            return override
    return ALGORITHM_TOLERANCES.get(algorithm, DEFAULT_TOLERANCE)
