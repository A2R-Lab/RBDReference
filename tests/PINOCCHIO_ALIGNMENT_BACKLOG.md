# Pinocchio Alignment Backlog

This backlog records small, safe, additive improvements that would make
`URDFParser` and `RBDReference` easier to validate against Pinocchio and easier to
use from a Pinocchio-oriented workflow.

## Parsing And Testability Improvements For URDFParser

- Add an additive strict parse API or a `raise_on_error` option so parse failures
  can surface structured exceptions instead of returning `None`.
- Return parse diagnostics that include the URDF path, root link, and failing stage
  rather than only printing to stdout.
- Expose parse-order metadata directly:
  resolved root link
  joint traversal policy
  original URDF joint ordering
  final internal joint ordering
- Keep Pinocchio-compatible traversal available as an explicit parser policy so
  ordering changes remain intentional and testable.
- Add stable helpers for:
  joint names
  movable joint names
  joint limits
  fixed-joint retention/merge results
- Make parser stdout reporting optional so tests and downstream tools can opt into
  quieter machine-readable behavior.

Observed smoke-corpus pain points in this checkout:

- `iiwa14` fixed-base strict parse hit a SymPy parse failure on an empty expression.
- `go2` fixed-base strict parse hit an index error during GRiD-side parse flow.
- `g1` fixed-base strict parse hit a missing joint-origin assumption (`NoneType`
  subscript) in `parse_joints`.

## User-Facing Convention Improvements For URDFParser

- Add additive Pinocchio-friendly metadata helpers such as `nq`, `nv`, and a clear
  base-joint description without removing the current API.
- Make floating-base state conventions explicit in the Python interface and docs,
  including quaternion ordering.
- Clarify how fixed joints are merged and how that affects the model structure that
  downstream algorithms see.
- Expose joint and parent/child indexing in a way that is easier to compare with
  Pinocchio's model semantics.

## User-Facing Convention Improvements For RBDReference

- Add Pinocchio-shaped convenience wrappers for the primary outputs of:
  inverse dynamics
  mass matrix workflows
  gravity or bias terms when semantics are clarified
- Standardize public returns on `numpy.ndarray` instead of mixing arrays, matrices,
  and bundled tuples where a narrow primary result would be clearer.
- Make floating-base input expectations explicit and queryable, especially for
  quaternion order and free-flyer state layout.
- Mark methods that are fixed-base only or floating-base incomplete in the public
  Python docs.
- Prefer additive wrappers and metadata accessors over breaking changes so existing
  GRiD users are not disrupted.

## D.1 Alignment Audit (T1, 2026-05-30)

Concrete gaps found by auditing `RBDReference` / `URDFParser` against Pinocchio
semantics. Each verified against the code in this checkout. Fix-ownership noted.

### Bugs to fix (T4 owns the fix)

- **`apply_external_forces` is buggy** (`RBDReference.py:1665-1691`):
  1. `Xa` is only assigned inside the `parent_id == -1` branch. For the first
     non-root body whose parent is the root chain, the `else` branch references
     `Xa` from a prior loop iteration (uninitialized / stale on the first such
     body), so the accumulated transform is wrong.
  2. Line 1688 calls `get_Xmat_Func_by_id(curr_id)(curr_id)` — it passes the
     **joint id** `curr_id` where a **q value** is expected (the `parent_id==-1`
     branch correctly passes `_q = q[inds_q]`). The transform is evaluated at the
     joint index, not the joint angle.
  Net effect: per-link external-force distribution is incorrect for any non-root
  body. Record only; **T4 owns the fix** (with a regression test vs Pinocchio
  `aba` with `fext`).

- **`f_ext` not threaded through the mimic FD fast path**
  (`RBDReference.py:2139-2156`, NOTE relocated to `docs/open-tasks/notes.md`):
  the mimic-aware `aba` computes `qdd = Minv @ (tau - rnea(q, qd, 0))`; neither
  the bias nor the solve applies `f_ext`. **T4 owns the fix** (consolidate
  external-force handling so the mimic path supports it).

### Intentional divergences (known-correct — record, do not "fix")

- **`minv` for mimic robots returns dense `inv(crba(q))`**
  (`RBDReference.py:2031+`, caveat at `:1907` relocated to notes). The ABA-style
  `minv` backward recursion is per-body `(U, d)` and not reduced-model aware, so
  for mimic models `minv` falls back to the dense inverse of the (mimic-aware)
  CRBA mass matrix. This matches Pinocchio's reduced-model `M^{-1}`. Known-correct.

### Reference-backend oracle note

- **No direct Pinocchio second-order oracle.** The reference backend's SO
  equivalence (`idsva_so` / `fdsva_so`) is validated against the `pin_so_ext`
  pybind11 C++ extension under `RBDReference/equivalents/pin_so_ext/`, not a
  Pinocchio Python SO API. This is by design (Pinocchio's Python surface does not
  expose the SO tensors directly); recorded so future audits don't mistake the
  ext binding for a gap.

### Already aligned (DONE)

- **Floating-base quaternion + spatial-velocity ordering is aligned.**
  `Joint.set_type` (`URDFParser/Joint.py:209-258`) uses an **xyzw** quaternion
  (`q1..q4` → `quat_to_rot_sp`) and a floating-base `S` whose user-facing velocity
  order is Pinocchio's `[vx, vy, vz, wx, wy, wz]` (mapped to GRiD's internal
  `[w; v]` spatial layout). No action needed — recorded as DONE.
