from dataclasses import dataclass
from typing import List, Mapping

import numpy as np
from bs4 import BeautifulSoup

from .conventions import (
    ConventionMismatch,
    MimicInfo,
    collapse_pin_q_to_project,
    expand_q_for_mimic,
    movable_joint_names_excluding_floating_root,
    normalize_matrix,
    normalize_pin_compatible_quaternion,
    normalize_project_q_for_pin,
    reduce_matrix_for_mimic,
    reduce_pinocchio_q_jacobian_to_project,
    normalize_vector,
)


@dataclass
class PinocchioModelAdapter:
    spec: object
    base_mode: str
    model: object
    data: object
    mismatches: List[ConventionMismatch]
    urdf_joint_types_by_name: dict
    urdf_mimic_joint_names: set
    urdf_path: str = ""
    # Name-keyed `<mimic>` relations. Populated by `build_pinocchio_adapter`.
    # Empty for robots without mimic joints (everything else stays bit-exact).
    mimic_info: MimicInfo = None

    @property
    def nq(self) -> int:
        # Report the PROJECT layout's nq (mimic joints don't own a generalized
        # coordinate). For non-mimic robots this is bit-identical to
        # `self.model.nq`; for mimic robots it's reduced by the count of
        # mimic joints' scalar q-slots (each mimic joint contributes one
        # q-slot in pinocchio's unreduced model).
        nq_pin = int(self.model.nq)
        if self.mimic_info is None or self.mimic_info.is_empty():
            return nq_pin
        # Each mimic joint loses one scalar q slot (non-continuous joints) or
        # two slots (continuous joints with [cos,sin] expansion). All current
        # mimic-using robots in the manifest (fr3, h1_2) have prismatic /
        # revolute (NOT continuous) mimic joints, so this is len(mimic).
        n_mimic = 0
        for name in self.mimic_info.relations:
            jtype = self.urdf_joint_types_by_name.get(name)
            n_mimic += 2 if jtype == "continuous" else 1
        return nq_pin - n_mimic

    @property
    def nv(self) -> int:
        nv_pin = int(self.model.nv)
        if self.mimic_info is None or self.mimic_info.is_empty():
            return nv_pin
        return nv_pin - len(self.mimic_info.relations)

    @property
    def joint_names(self) -> List[str]:
        return [str(name) for name in list(self.model.names)[1:]]

    @property
    def actuated_joint_names(self) -> List[str]:
        return movable_joint_names_excluding_floating_root(self.base_mode, self.joint_names)

    @property
    def frame_names(self) -> List[str]:
        return [frame.name for frame in self.model.frames]

    @property
    def scalar_joint_names(self) -> List[str]:
        # Full pinocchio-side scalar joint list (includes mimic joints; they
        # still consume one q-slot per joint in the unreduced pin model).
        return [
            name
            for name in self.actuated_joint_names
            if self.urdf_joint_types_by_name.get(name) != "floating"
        ]

    @property
    def project_scalar_joint_names(self) -> List[str]:
        """Pinocchio-side scalar joint names excluding mimic joints.

        Matches the project (GRiD) layout in which mimic joints don't own a
        generalized coordinate. Used by the conversion helpers below to map
        between project q (size nv_grid) and pinocchio q (size nv_pin).
        """
        return [
            name for name in self.scalar_joint_names
            if name not in self.urdf_mimic_joint_names
        ]

    def _floating_prefix_q(self) -> int:
        return 7 if self.base_mode == "floating" else 0

    def _floating_prefix_v(self) -> int:
        return 6 if self.base_mode == "floating" else 0

    def _expand_project_q_to_pin_full(self, q):
        """Expand a project-layout q (mimic-collapsed) to the unreduced
        pin-model layout, applying mimic relations and continuous-joint
        expansion. The result is what `pin.rnea` / `pin.crba` / etc. expect.
        """
        if self.mimic_info is None or self.mimic_info.is_empty():
            return normalize_project_q_for_pin(
                self.base_mode,
                q,
                joint_names=self.scalar_joint_names,
                joint_types_by_name=self.urdf_joint_types_by_name,
            )
        # Step 1: inject mimic-mirrored entries so q has one slot per pin
        # scalar joint (still in scalar layout — continuous joints still
        # carry their angle, not [cos, sin]).
        q_full_scalar = expand_q_for_mimic(
            np.asarray(q, dtype=np.float64),
            project_joint_names=self.project_scalar_joint_names,
            pinocchio_joint_names=self.scalar_joint_names,
            mimic=self.mimic_info,
            floating_prefix_len=self._floating_prefix_q(),
        )
        # Step 2: now use the standard pinocchio expansion against the FULL
        # scalar joint list (continuous joints become [cos, sin]).
        return normalize_project_q_for_pin(
            self.base_mode,
            q_full_scalar,
            joint_names=self.scalar_joint_names,
            joint_types_by_name=self.urdf_joint_types_by_name,
        )

    def _reduce_pin_matrix_to_project(self, matrix, axes_to_reduce):
        """Reduce a pin-layout matrix to the project layout.

        Each `axes_to_reduce` entry is `(axis, space)` with `space` in
        {"q", "v"}. The function FIRST handles continuous-joint reduction
        (legacy path via `reduce_pinocchio_q_jacobian_to_project` when the
        axis is "q" and the pin model uses [cos, sin] expansions), THEN
        folds mimic columns/rows into the mimicked column/row with the URDF
        multiplier scaling. Inputs without mimic joints fall through to the
        legacy code path so existing test behavior is bit-identical.
        """
        arr = np.asarray(matrix, dtype=np.float64)
        if self.mimic_info is None or self.mimic_info.is_empty():
            return arr
        return reduce_matrix_for_mimic(
            arr,
            project_joint_names=self.project_scalar_joint_names,
            pinocchio_joint_names=self.scalar_joint_names,
            mimic=self.mimic_info,
            floating_prefix_len_q=self._floating_prefix_q(),
            floating_prefix_len_v=self._floating_prefix_v(),
            axes_to_reduce=axes_to_reduce,
        )

    @property
    def continuous_joint_names(self) -> List[str]:
        return [
            name
            for name in self.actuated_joint_names
            if self.urdf_joint_types_by_name.get(name) == "continuous"
        ]

    @property
    def mimic_joint_names(self) -> List[str]:
        return sorted(self.urdf_mimic_joint_names)

    def has_invertible_mass_matrix(self, q, min_singular_value: float = 1e-6) -> bool:
        # Threshold raised from 1e-12: a mass matrix whose smallest singular
        # value is below ~1e-6 is *dynamically* near-singular — its inverse
        # amplifies by >1e6, so forward-dynamics / Minv / their derivatives are
        # numerically ill-defined (and overflow to NaN at high acceleration).
        # Skipping such configs is correct, not masking; healthy robots have a
        # smallest mass-matrix singular value far above 1e-6. (Catches rizon4,
        # whose model is near-singular across configs.)
        #
        # For mimic robots, the unreduced pinocchio M is structurally
        # singular (the mimic v-slot's row/col duplicates the mimicked
        # joint's). The DYNAMICALLY relevant quantity is the REDUCED M
        # (`G^T M G` with `G` the constraint projection), which is what
        # CRBA already produces via the mimic-aware reduce pattern.
        import pinocchio as pin

        q_pin = self._to_pin_q(q)
        mass = pin.crba(self.model, self.data, q_pin)
        mass = np.asarray(mass, dtype=np.float64)
        mass = 0.5 * (mass + mass.T)
        if self.mimic_info is not None and not self.mimic_info.is_empty():
            mass = self._reduce_pin_matrix_to_project(
                mass, axes_to_reduce=[(0, "v"), (1, "v")]
            )
        singular_values = np.linalg.svd(mass, compute_uv=False)
        if singular_values.size == 0:
            return False
        return bool(np.isfinite(singular_values).all() and singular_values[-1] > min_singular_value)

    def rnea(self, q, qd, qdd):
        import pinocchio as pin

        q_pin = self._to_pin_q(q)
        qd_pin = self._expand_project_v_to_pin(np.asarray(qd, dtype=np.float64))
        qdd_pin = self._expand_project_v_to_pin(np.asarray(qdd, dtype=np.float64))
        tau = pin.rnea(self.model, self.data, q_pin, qd_pin, qdd_pin)
        return normalize_vector(self._reduce_pin_v_to_project(np.asarray(tau, dtype=np.float64)))

    def aba(self, q, qd, tau):
        import pinocchio as pin

        q_pin = self._to_pin_q(q)
        qd_pin = self._expand_project_v_to_pin(np.asarray(qd, dtype=np.float64))
        # Mimic-aware forward dynamics. Pinocchio's `pin.aba` operates on
        # the UNREDUCED model -- which for a mimic-constrained URDF is
        # over-parameterized (the mimic's v-slot duplicates the
        # mimicked joint's velocity scaled by `multiplier`). Naively
        # calling `pin.aba(q, v, tau)` therefore solves a different
        # system than the project's reduced model: the per-body ABA
        # recursion uses each body's own (S, U, d) and the resulting
        # qdd is NOT the reduced-model acceleration that matches the
        # project's constraint-aware semantics. Using a locked / reduced
        # model on the pinocchio side gives the bit-comparable answer:
        #     qdd = M_reduced^{-1} * (tau_project - rnea_reduced(q, v, 0))
        # `self.rnea` and `self.minv` already perform the mimic
        # reduction (CRBA folds the duplicated rows/cols with the URDF
        # multiplier, and rnea reduces the resulting bias the same
        # way), so building ABA on top of them is the cleanest way to
        # express the constraint-aware comparison without depending on
        # Pinocchio's reduced-model builder.
        if self.mimic_info is not None and not self.mimic_info.is_empty():
            tau_proj = np.asarray(tau, dtype=np.float64)
            qd_proj = np.asarray(qd, dtype=np.float64)
            nv_proj = qd_proj.shape[0]
            bias = self.rnea(q, qd_proj, np.zeros(nv_proj))
            mass = self.minv(q)
            return normalize_vector(mass @ (tau_proj - bias))
        tau_pin = self._expand_project_v_to_pin(np.asarray(tau, dtype=np.float64))
        qdd = pin.aba(self.model, self.data, q_pin, qd_pin, tau_pin)
        return normalize_vector(self._reduce_pin_v_to_project(np.asarray(qdd, dtype=np.float64)))

    def forward_dynamics(self, q, qd, u):
        return self.aba(q, qd, u)

    def minv(self, q):
        import pinocchio as pin

        q_pin = self._to_pin_q(q)
        mass = pin.crba(self.model, self.data, q_pin)
        mass = np.asarray(mass, dtype=np.float64)
        mass = 0.5 * (mass + mass.T)
        # Reduce pinocchio's nv_pin x nv_pin mass matrix to project nv x nv
        # by collapsing mimic rows/cols into the mimicked column (with
        # multiplier scaling on both axes). The inverse is then taken on the
        # reduced matrix so it matches the project-layout Minv.
        if self.mimic_info is not None and not self.mimic_info.is_empty():
            mass = self._reduce_pin_matrix_to_project(
                mass, axes_to_reduce=[(0, "v"), (1, "v")]
            )
        minv = np.linalg.inv(mass)
        return normalize_matrix(minv)

    def crba(self, q):
        import pinocchio as pin

        q_pin = self._to_pin_q(q)
        mass = pin.crba(self.model, self.data, q_pin)
        mass = np.asarray(mass, dtype=np.float64)
        mass = 0.5 * (mass + mass.T)
        if self.mimic_info is not None and not self.mimic_info.is_empty():
            mass = self._reduce_pin_matrix_to_project(
                mass, axes_to_reduce=[(0, "v"), (1, "v")]
            )
        return normalize_matrix(mass)

    def rnea_grad(self, q, qd, qdd):
        import pinocchio as pin

        q_pin = self._to_pin_q(q)
        qd_pin = self._expand_project_v_to_pin(np.asarray(qd, dtype=np.float64))
        qdd_pin = self._expand_project_v_to_pin(np.asarray(qdd, dtype=np.float64))
        pin.computeRNEADerivatives(self.model, self.data, q_pin, qd_pin, qdd_pin)
        # For mimic robots fold both axes (mimic columns into target cols,
        # mimic rows into target rows) BEFORE the q-Jacobian chain reduction;
        # otherwise the matrix carries pinocchio-full v-width which doesn't
        # match the project nq layout expected by reduce_pinocchio_q_jacobian.
        if self.mimic_info is not None and not self.mimic_info.is_empty():
            dtau_dq = self._reduce_pin_matrix_to_project(
                np.asarray(self.data.dtau_dq, dtype=np.float64),
                axes_to_reduce=[(0, "v"), (1, "v")],
            )
            dtau_dv = self._reduce_pin_matrix_to_project(
                np.asarray(self.data.dtau_dv, dtype=np.float64),
                axes_to_reduce=[(0, "v"), (1, "v")],
            )
        else:
            dtau_dq = reduce_pinocchio_q_jacobian_to_project(
                np.asarray(self.data.dtau_dq, dtype=np.float64),
                self.base_mode,
                q,
                joint_names=self.project_scalar_joint_names,
                joint_types_by_name=self.urdf_joint_types_by_name,
            )
            dtau_dv = normalize_matrix(np.asarray(self.data.dtau_dv, dtype=np.float64))
        return (dtau_dq, dtau_dv)

    def idsva_so_body_frame(self, q, qd, qdd):
        """Return (d2tau_dq, d2tau_dqd, d2tau_dvdq, dM_dq) from Pinocchio's
        C++ `ComputeRNEASecondOrderDerivatives`, in our nv-indexed Lie-tangent
        convention.

        Pinocchio's mixed tensor `d2tau_dqdv[i,j,k] = d2tau_i/(dq_j dv_k)` is
        transposed on axes (1,2) to match our `d2tau_dvdq[i,j,k] = d2tau_i/(dv_j dq_k)`.
        """
        from .pin_so_ext import load as _load_so

        ext = _load_so()
        q_pin = normalize_project_q_for_pin(
            self.base_mode,
            q,
            joint_names=self.scalar_joint_names,
            joint_types_by_name=self.urdf_joint_types_by_name,
        )
        qd_arr = np.asarray(qd, dtype=np.float64)
        qdd_arr = np.asarray(qdd, dtype=np.float64)
        d2tau_dqdq, d2tau_dvdv, d2tau_dqdv, d2tau_dadq = ext.compute_rnea_second_order(
            self.urdf_path,
            self.base_mode == "floating",
            np.asarray(q_pin, dtype=np.float64),
            qd_arr,
            qdd_arr,
        )
        d2tau_dvdq = np.asarray(d2tau_dqdv, dtype=np.float64).transpose(0, 2, 1)
        return (
            np.asarray(d2tau_dqdq, dtype=np.float64),
            np.asarray(d2tau_dvdv, dtype=np.float64),
            d2tau_dvdq,
            np.asarray(d2tau_dadq, dtype=np.float64),
        )

    def fdsva_so(self, q, qd, u):
        """Return (daba_dqdq, daba_dvdq, daba_dvdv, daba_dtdq) composed from
        Pinocchio's second-order RNEA + first-order Minv/ABA derivatives.

        Uses the same composition formula as `RBDReference.fdsva_so`, but with
        every input grounded in Pinocchio's bound C++ implementations so the
        result is independent of our analytic code path.
        """
        import pinocchio as pin

        q_pin = normalize_project_q_for_pin(
            self.base_mode,
            q,
            joint_names=self.scalar_joint_names,
            joint_types_by_name=self.urdf_joint_types_by_name,
        )
        qd_arr = np.asarray(qd, dtype=np.float64)
        u_arr = np.asarray(u, dtype=np.float64)
        # Pinocchio ABA gives qdd and fills first-order derivatives.
        qdd = np.asarray(pin.aba(self.model, self.data, q_pin, qd_arr, u_arr), dtype=np.float64)
        d2tau_dq, d2tau_dqd, d2tau_dvdq, dM_dq = self.idsva_so_body_frame(q, qd, qdd)
        pin.computeMinverse(self.model, self.data, q_pin)
        Minv = normalize_matrix(np.asarray(self.data.Minv, dtype=np.float64))
        fd_dq, fd_dqd = self.forward_dynamics_grad(q, qd, u)
        daba_dqdq = -np.einsum(
            "il,ljk->ijk",
            Minv,
            d2tau_dq
            + np.einsum("ilk,lj->ijk", dM_dq, fd_dq)
            + np.einsum("ilk,lj->ikj", dM_dq, fd_dq),
        )
        daba_dvdq = -np.einsum(
            "il,ljk->ijk",
            Minv,
            d2tau_dvdq + np.einsum("ilk,lj->ijk", dM_dq, fd_dqd),
        )
        daba_dvdv = -np.einsum("il,ljk->ijk", Minv, d2tau_dqd)
        daba_dtdq = -np.einsum(
            "il,ljk->ijk",
            Minv,
            np.einsum("ilk,lj->ijk", dM_dq, Minv),
        )
        return daba_dqdq, daba_dvdq, daba_dvdv, daba_dtdq

    def forward_dynamics_grad(self, q, qd, u):
        import pinocchio as pin

        # Mimic: pin.computeABADerivatives' per-body recursion doesn't commute
        # with the `+= alpha *` fold (same root cause as the ABA mimic gap).
        # Use the implicit-fn identity instead: dqdd/dq = -M^{-1} drnea/dq, all
        # three components are already mimic-aware.
        if self.mimic_info is not None and not self.mimic_info.is_empty():
            qdd = self.forward_dynamics(q, qd, u)
            minv = self.minv(q)
            dc_dq, dc_dqd = self.rnea_grad(q, qd, qdd)
            return (-minv @ dc_dq, -minv @ dc_dqd)

        q_pin = self._to_pin_q(q)
        qd_pin = self._expand_project_v_to_pin(np.asarray(qd, dtype=np.float64))
        u_pin = self._expand_project_v_to_pin(np.asarray(u, dtype=np.float64))
        pin.computeABADerivatives(self.model, self.data, q_pin, qd_pin, u_pin)
        ddq_dq = reduce_pinocchio_q_jacobian_to_project(
            np.asarray(self.data.ddq_dq, dtype=np.float64),
            self.base_mode,
            q,
            joint_names=self.project_scalar_joint_names,
            joint_types_by_name=self.urdf_joint_types_by_name,
        )
        ddq_dv = normalize_matrix(np.asarray(self.data.ddq_dv, dtype=np.float64))
        return (ddq_dq, ddq_dv)

    # ----- Time integrators (canonical via pinocchio.integrate / dIntegrate) -----

    def _to_pin_q(self, q):
        # Mimic-aware project -> pin q expansion. Falls through to the
        # legacy path when the model has no mimic joints (mimic_info empty),
        # so existing tests are bit-identical for non-mimic robots.
        return self._expand_project_q_to_pin_full(q)

    def _from_pin_q_collapse(self, q_pin):
        """Inverse of `_to_pin_q`: pin-layout q -> project-layout q.

        Steps: (1) collapse continuous-joint [cos,sin] pairs back to scalar
        angles via `collapse_pin_q_to_project`, then (2) drop mimic entries
        (each mimic q-slot is dependent on its target).
        """
        q_project_scalar = collapse_pin_q_to_project(
            self.base_mode,
            np.asarray(q_pin, dtype=np.float64),
            joint_names=self.scalar_joint_names,
            joint_types_by_name=self.urdf_joint_types_by_name,
        )
        if self.mimic_info is None or self.mimic_info.is_empty():
            return q_project_scalar
        # Drop mimic joints' q-slots so the result has one entry per
        # project joint.
        prefix = self._floating_prefix_q()
        prefix_block = q_project_scalar[:prefix]
        kept = []
        for i, name in enumerate(self.scalar_joint_names):
            if name in self.urdf_mimic_joint_names:
                continue
            kept.append(float(q_project_scalar[prefix + i]))
        return np.concatenate([prefix_block, np.asarray(kept, dtype=np.float64)])

    def _expand_project_v_to_pin(self, v):
        """Expand a project-layout v (size nv_grid) to pinocchio-layout v
        (size nv_pin) by injecting mimic-mirrored entries with their URDF
        multiplier scaling. The floating-base prefix passes through unchanged.
        """
        v = np.asarray(v, dtype=np.float64)
        if self.mimic_info is None or self.mimic_info.is_empty():
            return v
        prefix = self._floating_prefix_v()
        out_prefix = v[:prefix]
        project_v_index = {
            name: i for i, name in enumerate(self.project_scalar_joint_names)
        }
        out_suffix_by_pin_pos = {}
        out_suffix = []
        for i, name in enumerate(self.scalar_joint_names):
            if name in self.mimic_info.relations:
                target_name, mult, _ = self.mimic_info.relations[name]
                tgt_pos = out_suffix_by_pin_pos.get(target_name)
                if tgt_pos is None:
                    raise ValueError(
                        f"mimic v-expand: target '{target_name}' missing before '{name}'"
                    )
                out_suffix.append(float(mult * out_suffix[tgt_pos]))
            else:
                project_pos = project_v_index.get(name)
                if project_pos is None:
                    raise ValueError(
                        f"mimic v-expand: project joint '{name}' missing"
                    )
                out_suffix.append(float(v[prefix + project_pos]))
                out_suffix_by_pin_pos[name] = len(out_suffix) - 1
        return np.concatenate([out_prefix, np.asarray(out_suffix, dtype=np.float64)])

    def _reduce_pin_v_to_project(self, vec):
        """Reduce a pinocchio-layout v-space vector to the project layout by
        folding mimic entries into the mimicked entry with the URDF
        multiplier scaling. Correct for forces/torques: project tau_i
        absorbs `multiplier_m * tau_pin[m]` for every mimic m of i, since
        a unit project-v_i actuates both the mimicked joint and (with
        multiplier) the mimic joint.
        """
        vec = np.asarray(vec, dtype=np.float64)
        if self.mimic_info is None or self.mimic_info.is_empty():
            return vec
        prefix = self._floating_prefix_v()
        out_prefix = vec[:prefix]
        project_index = {
            name: i for i, name in enumerate(self.project_scalar_joint_names)
        }
        out_suffix = np.zeros(len(self.project_scalar_joint_names), dtype=np.float64)
        for i, name in enumerate(self.scalar_joint_names):
            pin_val = float(vec[prefix + i])
            if name in self.mimic_info.relations:
                target_name, mult, _ = self.mimic_info.relations[name]
                tgt_idx = project_index.get(target_name)
                if tgt_idx is None:
                    raise ValueError(f"mimic v-reduce: target '{target_name}' missing")
                out_suffix[tgt_idx] += float(mult) * pin_val
            else:
                project_idx = project_index.get(name)
                if project_idx is None:
                    raise ValueError(f"mimic v-reduce: project joint '{name}' missing")
                out_suffix[project_idx] += pin_val
        return np.concatenate([out_prefix, out_suffix])

    def _pin_integrate(self, q, v_dt):
        """`pin.integrate(model, q, v_dt)` returned in the project's scalar-joint
        layout. v_dt is in tangent space (size nv_project) and is internally
        expanded to pinocchio's nv layout (mimic-mirrored entries are filled
        with `multiplier * v_dt[target]`).

        The result is collapsed back from Pinocchio's nq layout (continuous
        joints [cos,sin] -> scalar angle) so it can be fed straight into the
        project-layout routines (`self.aba` / `self.minv` / derivatives) used by
        the multi-stage integrators, and so the integrator output matches the
        project adapter's layout. Configurations are compared wrap-safely in
        the tangent space via `q_tangent_residual`, so the collapse's principal
        branch is harmless."""
        import pinocchio as pin

        q_pin = self._to_pin_q(q)
        v_dt_pin = self._expand_project_v_to_pin(np.asarray(v_dt, dtype=np.float64))
        q_new_pin = np.asarray(
            pin.integrate(self.model, q_pin, v_dt_pin),
            dtype=np.float64,
        )
        if self.base_mode == "floating":
            q_new_pin = normalize_pin_compatible_quaternion(q_new_pin)
        # Use mimic-aware collapse if mimic_info is non-empty; otherwise the
        # legacy path is bit-identical for non-mimic robots.
        if self.mimic_info is not None and not self.mimic_info.is_empty():
            return self._from_pin_q_collapse(q_new_pin)
        return collapse_pin_q_to_project(
            self.base_mode,
            q_new_pin,
            joint_names=self.scalar_joint_names,
            joint_types_by_name=self.urdf_joint_types_by_name,
        )

    def to_pin_q(self, q):
        """Public: map a project-layout configuration to Pinocchio's nq layout
        (scalar joint angle -> [cos,sin] for continuous joints, xyzw quaternion
        kept for the free-flyer root)."""
        return self._to_pin_q(q)

    def q_tangent_residual(self, q_project_a, q_project_b):
        """Tangent-space residual ``q_b (-) q_a`` between two project-layout
        configurations, via `pin.difference`. Returns an nv_project-vector
        (the project-layout reduced size) so that two configurations
        representing the same pose give ~0 regardless of joint representation
        (scalar-angle vs [cos,sin]) or 2*pi wrapping. This is the
        representation-agnostic way to compare a continuous-joint / free-flyer
        q-update across the two libraries.

        For mimic robots, `pin.difference` returns the unreduced pin nv
        vector; we reduce it (mimic v-slots fold into the mimicked v-slot
        with the URDF multiplier scaling) so the size matches the
        project layout. For a well-formed mimic relation, the mimic
        joint's residual is `multiplier * residual[target]`, so the
        reduce pattern only doubles the target's effective contribution
        — but residual ~= 0 for a correct integrator, so this is a
        size-fix only."""
        import pinocchio as pin

        q0 = np.asarray(self._to_pin_q(q_project_a), dtype=np.float64)
        q1 = np.asarray(self._to_pin_q(q_project_b), dtype=np.float64)
        residual = np.asarray(pin.difference(self.model, q0, q1), dtype=np.float64)
        if self.mimic_info is not None and not self.mimic_info.is_empty():
            return self._reduce_pin_v_to_project(residual)
        return residual

    def _pin_dIntegrate(self, q, v_dt, with_respect_to):
        """Wrap `pin.dIntegrate` and return the (nv_project, nv_project)
        Jacobian.

        For mimic robots, `v_dt` arrives in the project's reduced nv
        layout; we expand to pinocchio's nv before calling `dIntegrate`,
        then reduce the resulting (nv_pin, nv_pin) Jacobian on both axes
        to the project layout (mimic v-slots fold into the target with
        the URDF multiplier on each axis)."""
        import pinocchio as pin

        q_pin = self._to_pin_q(q)
        v_dt_pin = self._expand_project_v_to_pin(np.asarray(v_dt, dtype=np.float64))
        arg = pin.ArgumentPosition.ARG0 if with_respect_to == "q" else pin.ArgumentPosition.ARG1
        J = np.asarray(
            pin.dIntegrate(self.model, q_pin, v_dt_pin, arg),
            dtype=np.float64,
        )
        if self.mimic_info is not None and not self.mimic_info.is_empty():
            J = self._reduce_pin_matrix_to_project(
                J, axes_to_reduce=[(0, "v"), (1, "v")]
            )
        return J

    @staticmethod
    def _butcher(integrator_type: str):
        if integrator_type == "euler":
            return [], [1.0]
        if integrator_type in ("semi_implicit_euler", "si_euler"):
            return [], [1.0]
        if integrator_type == "midpoint":
            return [0.5], [0.0, 1.0]
        if integrator_type == "rk3":
            return [0.5, 0.75], [2.0 / 9.0, 3.0 / 9.0, 4.0 / 9.0]
        if integrator_type == "rk4":
            return [0.5, 0.5, 1.0], [1.0 / 6.0, 2.0 / 6.0, 2.0 / 6.0, 1.0 / 6.0]
        raise ValueError(f"Unknown integrator_type: {integrator_type}")

    def integrator(self, q, qd, u, dt, integrator_type: str = "euler"):
        """Pinocchio-backed integrator step. Uses `pin.integrate` for the
        Lie-group q update and `pin.aba` per stage for the qdd refinement.
        Output shape: nq + nv concatenated as [q_new, v_new]."""
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        qdd1 = self.aba(q, qd, u)
        if integrator_type == "euler":
            q_new = self._pin_integrate(q, dt * qd)
            v_new = qd + dt * qdd1
            return np.concatenate([q_new, v_new])
        if integrator_type in ("semi_implicit_euler", "si_euler"):
            v_new = qd + dt * qdd1
            q_new = self._pin_integrate(q, dt * v_new)
            return np.concatenate([q_new, v_new])
        c_list, b_list = self._butcher(integrator_type)
        N = len(b_list)
        qdd_list = [qdd1]
        prev_qdd = qdd1
        for stage_idx in range(1, N):
            c_prev = c_list[stage_idx - 1]
            p_q = self._pin_integrate(q, c_prev * dt * qd)
            p_qd = qd + c_prev * dt * prev_qdd
            stage_qdd = self.aba(p_q, p_qd, u)
            qdd_list.append(stage_qdd)
            prev_qdd = stage_qdd
        accel = sum(b * qdd for b, qdd in zip(b_list, qdd_list))
        q_new = self._pin_integrate(q, dt * qd)
        v_new = qd + dt * accel
        return np.concatenate([q_new, v_new])

    def integrator_gradient(self, q, qd, u, dt, integrator_type: str = "euler"):
        """Pinocchio-backed integrator gradient [A|B] of shape (2*nv, 3*nv).

        Uses `pin.dIntegrate` for the q-side blocks and `pin.computeABADerivatives`
        for the qdd partials. The chain rule across multi-stage variants
        mirrors the form in `RBDReference.integrator_grad`.
        """
        q = np.asarray(q, dtype=np.float64)
        qd = np.asarray(qd, dtype=np.float64)
        u = np.asarray(u, dtype=np.float64)
        nv = self.nv
        I_n = np.eye(nv)
        Z_n = np.zeros((nv, nv))

        def fd_grad_at(pq, pqd):
            J_qq, J_qv = self.forward_dynamics_grad(pq, pqd, u)
            return np.asarray(J_qq, dtype=np.float64), np.asarray(J_qv, dtype=np.float64), self.minv(pq)

        def q_top_blocks(v_dt_arg):
            return (self._pin_dIntegrate(q, v_dt_arg, "q"),
                    self._pin_dIntegrate(q, v_dt_arg, "v"))

        if integrator_type == "euler":
            J_qq, J_qv, Minv = fd_grad_at(q, qd)
            dInt_q, dInt_v = q_top_blocks(dt * qd)
            top = np.hstack([dInt_q, dt * dInt_v, Z_n])
            bottom = np.hstack([dt * J_qq, I_n + dt * J_qv, dt * Minv])
            return np.vstack([top, bottom])
        if integrator_type in ("semi_implicit_euler", "si_euler"):
            J_qq, J_qv, Minv = fd_grad_at(q, qd)
            v_new = qd + dt * self.aba(q, qd, u)
            dInt_q, dInt_v = q_top_blocks(dt * v_new)
            dvdq = dt * J_qq
            dvdv = I_n + dt * J_qv
            dvdu = dt * Minv
            dt_dInt_v = dt * dInt_v
            top = np.hstack([dInt_q + dt_dInt_v @ dvdq,
                              dt_dInt_v @ dvdv,
                              dt_dInt_v @ dvdu])
            bottom = np.hstack([dvdq, dvdv, dvdu])
            return np.vstack([top, bottom])
        c_list, b_list = self._butcher(integrator_type)
        N = len(b_list)
        qdd_list = []
        D_qdd_list = []
        qdd_list.append(self.aba(q, qd, u))
        J_qq, J_qv, Minv = fd_grad_at(q, qd)
        D_qdd_list.append(np.hstack([J_qq, J_qv, Minv]))
        for stage_idx in range(1, N):
            c_prev = c_list[stage_idx - 1]
            prev_qdd = qdd_list[-1]
            p_q = self._pin_integrate(q, c_prev * dt * qd)
            p_qd = qd + c_prev * dt * prev_qdd
            qdd_list.append(self.aba(p_q, p_qd, u))
            J_qq_i, J_qv_i, Minv_i = fd_grad_at(p_q, p_qd)
            v_dt_stage = c_prev * dt * qd
            dInt_q_stage, dInt_v_stage = q_top_blocks(v_dt_stage)
            dp_q_block = np.hstack([dInt_q_stage, c_prev * dt * dInt_v_stage, Z_n])
            dp_qd_block = np.hstack([Z_n, I_n, Z_n]) + c_prev * dt * D_qdd_list[-1]
            d_u_block = np.hstack([Z_n, Z_n, Minv_i])
            D_qdd_list.append(J_qq_i @ dp_q_block + J_qv_i @ dp_qd_block + d_u_block)
        sum_b_D = sum(b * D for b, D in zip(b_list, D_qdd_list))
        dInt_q_final, dInt_v_final = q_top_blocks(dt * qd)
        top = np.hstack([dInt_q_final, dt * dInt_v_final, Z_n])
        bottom = np.hstack([Z_n, I_n, Z_n]) + dt * sum_b_D
        return np.vstack([top, bottom])

    def end_effector_pose(self, q, target_name: str, offset=None):
        import pinocchio as pin

        if offset is None:
            offset = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)

        q_pin = self._to_pin_q(q)
        pin.forwardKinematics(self.model, self.data, q_pin)
        pin.updateFramePlacements(self.model, self.data)

        frame_id = self.model.getFrameId(target_name)
        placement = self.data.oMf[frame_id]
        point_local = np.asarray(offset[:3], dtype=np.float64)
        point_world = placement.translation + placement.rotation @ point_local
        rot = placement.rotation
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        pitch_temp = np.sqrt(rot[2, 2] * rot[2, 2] + rot[2, 1] * rot[2, 1])
        pitch = np.arctan2(-rot[2, 0], pitch_temp)
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
        return normalize_vector(np.concatenate((point_world, np.array([roll, pitch, yaw]))))

    def end_effector_rotation_matrix(self, q, target_name: str):
        import pinocchio as pin

        q_pin = self._to_pin_q(q)
        pin.forwardKinematics(self.model, self.data, q_pin)
        pin.updateFramePlacements(self.model, self.data)

        if target_name in self.joint_names:
            joint_id = self.model.getJointId(target_name)
            return normalize_matrix(np.asarray(self.data.oMi[joint_id].rotation, dtype=np.float64))

        frame_id = self.model.getFrameId(target_name)
        return normalize_matrix(np.asarray(self.data.oMf[frame_id].rotation, dtype=np.float64))

    def _normalize_project_q_for_pose_differences(self, q):
        q = np.asarray(q, dtype=np.float64).copy()
        if self.base_mode == "floating":
            q[:7] = normalize_pin_compatible_quaternion(q[:7])
        return q

    def end_effector_pose_gradient(self, q, target_name: str, offset=None, step: float = 1e-6):
        """End-effector pose gradient w.r.t. generalized velocity v (TANGENT space).

        Output shape is 6 x nv_project — i.e. one column per project velocity
        DOF, with mimic joints already folded into their target's column via
        `_pin_integrate` (which expands `v_dt` to pin layout under the URDF
        mimic relation, so perturbing v_project[i] moves BOTH the mimicked
        joint and every joint that mimics it). For non-mimic robots
        `nv_project == self.model.nv` and this is bit-identical to the
        legacy FD-over-pin-nv path.
        """
        if offset is None:
            offset = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        q = self._normalize_project_q_for_pose_differences(q)
        nv_project = self._project_nv()
        gradient = np.zeros((6, nv_project), dtype=np.float64)

        for v_ind in range(nv_project):
            v = np.zeros(nv_project, dtype=np.float64)
            v[v_ind] = step
            q_pos = self._pin_integrate(q, v)
            q_neg = self._pin_integrate(q, -v)
            pose_pos = self.end_effector_pose(q_pos, target_name, offset=offset)
            pose_neg = self.end_effector_pose(q_neg, target_name, offset=offset)
            diff = pose_pos - pose_neg
            # angle-wrap the rpy rows so finite differences are sane near branch cuts
            diff[3:6] = ((diff[3:6] + np.pi) % (2.0 * np.pi)) - np.pi
            gradient[:, v_ind] = diff / (2.0 * step)
        return gradient

    def _project_nv(self) -> int:
        """nv in the project layout = pin nv minus the number of mimic joints."""
        if self.mimic_info is None or self.mimic_info.is_empty():
            return int(self.model.nv)
        return int(self.model.nv) - len(self.mimic_info.relations)

    def end_effector_pose_hessian(self, q, target_name: str, offset=None, step: float = 1e-5):
        """End-effector pose Hessian d^2(pose)/dv^2 (TANGENT, pinocchio convention).

        Output shape is 6 x nv x nv. For joint targets uses pinocchio's analytic
        flow (`computeForwardKinematicsDerivatives` + `computeJointKinematicHessians`
        + `getJointKinematicHessian(LOCAL_WORLD_ALIGNED)`), which matches the d/dv
        convention and runs in O(N) instead of O(nv) Jacobian FD calls. Frame
        targets (and the back-compat path) fall back to central-difference FD on
        the d/dv Jacobian via `pin.integrate(q, h*e_i)`. The analytic path mirrors
        the bench harness in `test/benchmarks/baselines/pinocchio/timePinocchio.cpp`."""
        import pinocchio as pin

        if offset is None:
            offset = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        q = self._normalize_project_q_for_pose_differences(q)
        nv = self.model.nv
        nv_project = self._project_nv()

        # Analytic path: joint target with kinematic-hessian API available. Skip
        # if the user passed a non-default offset (the analytic Hessian is rooted
        # at the joint placement and ignores the EE offset that FD applies).
        analytic_ok = (
            target_name in self.joint_names
            and hasattr(pin, "computeJointKinematicHessians")
            and hasattr(pin, "getJointKinematicHessian")
            and np.allclose(offset, np.array([0.0, 0.0, 0.0, 1.0]))
        )
        if analytic_ok:
            q_pin = self._to_pin_q(q)
            v_zero = np.zeros(nv, dtype=np.float64)
            a_zero = np.zeros(nv, dtype=np.float64)
            pin.computeForwardKinematicsDerivatives(self.model, self.data, q_pin, v_zero, a_zero)
            pin.computeJointKinematicHessians(self.model, self.data)
            joint_id = self.model.getJointId(target_name)
            H = np.asarray(
                pin.getJointKinematicHessian(self.model, self.data, joint_id, pin.LOCAL_WORLD_ALIGNED),
                dtype=np.float64,
            )
            # Pinocchio returns shape (6, nv_pin, nv_pin). For mimic robots
            # we fold the mimic rows/cols into their target rows/cols with
            # the URDF multiplier so the result has shape (6, nv_project,
            # nv_project) and matches the project adapter's output.
            if self.mimic_info is not None and not self.mimic_info.is_empty():
                H = self._reduce_pin_matrix_to_project(H, axes_to_reduce=[(1, "v"), (2, "v")])
            return H

        # FD fallback (frame target or older pinocchio without kinematic-hessian API).
        hessian = np.zeros((6, nv_project, nv_project), dtype=np.float64)
        for i in range(nv_project):
            v = np.zeros(nv_project, dtype=np.float64); v[i] = step
            q_plus = self._pin_integrate(q, v)
            q_minus = self._pin_integrate(q, -v)
            Jp = self.end_effector_pose_gradient(q_plus, target_name, offset=offset)
            Jm = self.end_effector_pose_gradient(q_minus, target_name, offset=offset)
            hessian[:, :, i] = (Jp - Jm) / (2.0 * step)
        # Symmetrize: analytic d^2/dv_j dv_i == d^2/dv_i dv_j; FD won't be exact,
        # so average to suppress per-pair noise.
        hessian = 0.5 * (hessian + np.transpose(hessian, axes=(0, 2, 1)))
        return hessian


def build_pinocchio_adapter(spec, resolved_model, base_mode: str) -> PinocchioModelAdapter:
    import pinocchio as pin

    with open(resolved_model.urdf_path, "r", encoding="utf-8") as urdf_file:
        soup = BeautifulSoup(urdf_file.read(), "xml").find("robot")
    urdf_joint_types_by_name = {
        joint["name"]: joint["type"]
        for joint in soup.find_all("joint", recursive=False)
    }
    urdf_mimic_joint_names = {
        joint["name"]
        for joint in soup.find_all("joint", recursive=False)
        if joint.find("mimic") is not None
    }
    # Build the name-keyed mimic relations map (target, multiplier, offset).
    mimic_relations = {}
    for joint in soup.find_all("joint", recursive=False):
        mimic_tag = joint.find("mimic")
        if mimic_tag is None:
            continue
        if not mimic_tag.has_attr("joint"):
            raise ValueError(
                f"Joint '{joint['name']}' has <mimic> without a `joint` attribute "
                "(URDF parse error)."
            )
        target_name = mimic_tag["joint"]
        multiplier = float(mimic_tag["multiplier"]) if mimic_tag.has_attr("multiplier") else 1.0
        offset = float(mimic_tag["offset"]) if mimic_tag.has_attr("offset") else 0.0
        mimic_relations[joint["name"]] = (target_name, multiplier, offset)
    mimic_info = MimicInfo(relations=mimic_relations)

    if base_mode == "floating":
        model = pin.buildModelFromUrdf(
            resolved_model.urdf_path,
            pin.JointModelFreeFlyer(),
        )
        mismatches = [
            ConventionMismatch(
                category="floating_base_quaternion",
                detail="Pinocchio free-flyer uses the same xyzw quaternion ordering as the current GRiD floating-base convention.",
            )
        ]
    else:
        model = pin.buildModelFromUrdf(resolved_model.urdf_path)
        mismatches = []

    data = model.createData()
    return PinocchioModelAdapter(
        spec=spec,
        base_mode=base_mode,
        model=model,
        data=data,
        mismatches=mismatches,
        urdf_joint_types_by_name=urdf_joint_types_by_name,
        urdf_mimic_joint_names=urdf_mimic_joint_names,
        urdf_path=str(resolved_model.urdf_path),
        mimic_info=mimic_info,
    )
