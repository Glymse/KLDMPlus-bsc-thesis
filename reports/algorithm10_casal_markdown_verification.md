# Algorithm 10 CASAL Markdown Verification

Source spec: `/Users/glymov/Downloads/casal_chart_kldm_pipeline(1).md`

This note records what is now implemented in:

- `/Users/glymov/DTU/6 Semester/Bachelor/Github/Main/kldm/src/kldmPlus/algorithm10_casal_chart.py`
- `/Users/glymov/DTU/6 Semester/Bachelor/Github/Main/kldm/src/kldmPlus/fixed_template_ssvd_project.py`

## 1. Paper map

### 1.1 CASAL
- Implemented as explicit split state with ambient KLDM state, constrained chart state, and dual variables.
- Ambient/coupling update is now written directly in Algorithm 10 instead of delegating to the old `SplitLangevin` helper.

### 1.2 DiffCSP++ / Wyckoff chart parameterization
- Implemented through explicit Wyckoff-template chart states from `initialize_constrained_template_states(...)`.
- Continuous chart variables are `free_vars` and lattice free variables in the DiffCSP++ `k` basis.

### 1.3 KLDM
- Implemented ambient state uses `f`, `v`, and `l`.
- Physical coupling is applied in `(f, k)` space, not raw KLDM lattice-feature space.

### 1.4 DAPS++
- Not used inside Algorithm 10. The sampler is now a chart-constrained KLDM path, not a blended DAPS-style refinement backend.

### 1.5 Score-MCMC correction
- Not implemented, matching the markdown guidance to defer this until projection is stable.

## 2. Correct mathematical target
- Implemented split target with ambient KLDM state and constrained chart output `z_T`.
- Final returned sample is the constrained chart state, not the ambient state.

## 3. Faithful CASAL-Chart-KLDM state
- Implemented in `_CasalGraphState`:
  - constrained `z_pos`, `z_l`, `z_k`, `z_h`
  - dual `mu_f`, `mu_k`
  - current projection diagnostics
- Ambient state remains in the KLDM sampling state prepared by `_prepare_csp_sampling(...)`.

## 4. Projection operator `P_C`
- Implemented over:
  - discrete template `W`
  - continuous chart variables `theta`
  - global origin shift `tau`
  - species-preserving assignment `pi`
- Implemented as:
  - template enumeration with `initialize_constrained_template_states(...)`
  - exhaustive origin-shift search
  - Hungarian assignment inside the fixed-template residual
  - hard validation after materialization

## 4.1 Why `tau` is mandatory
- Implemented.
- Default config now uses axis-grid origin shifts from the markdown via:
  - `origin_shift_mode: axis_grid`
  - `origin_shift_values: [0.0, 0.125, 0.875, 0.25, 0.75, 0.5]`

## 5. Fixed-template projection with SVD / SSVD
- Implemented in `fixed_template_ssvd_project.py`.
- Includes:
  - explicit residual construction
  - Jacobian via `torch.autograd.functional.jacobian`
  - damped SVD solve
  - rank truncation
  - delta norm clipping
  - line search
  - torus wrapping after updates

## 5.1 SSVD projection pseudocode
- Implemented directly in `fixed_template_ssvd_project(...)`.

## 5.2 What SVD can and cannot fix
- Implemented by embedding SSVD inside outer template and origin-shift search.
- Continuous SSVD does not decide the template alone; the outer loop does.

## 6. Discrete template moves with MH
- Implemented via exhaustive search over the enumerated template pool, not by an MH kernel.
- This satisfies discrete support over the evaluated template set, but it is deterministic rather than stochastic chart-space MH.

## 7. Faithful CASAL-Chart-KLDM update

### 7.1 Physical coupling residual
- Implemented in `(f, k)` space with wrapped torus residuals for fractional coordinates.

### 7.2 Time-dependent CASAL update
- Implemented as a direct Algorithm 10 coupling step in `(f, k)` space.
- The KLDM prior step and CASAL coupling step are still sequenced inside the same sampling loop rather than fused into one derived closed-form KLDM+CASAL integrator.

### 7.3 Operator splitting version
- Implemented and made explicit.
- The sampler logs `mode=faithful_chart_projection_operator_split`.

## 8. Review of old uploaded implementation
- Addressed:
  - old zero-gradient `SplitLangevin` path removed
  - old `(f, l)` coupling removed
  - old selector-backed projection replaced
  - final sample no longer uses “best projection seen”; it returns the actual final constrained state
  - output velocity is now explicitly zeroed

## 9. Faithful implementation roadmap

### Stage A: Projection correctness
- Implemented.

### Stage B: Algorithm 9 baseline
- External baseline remains separate; not part of Algorithm 10.

### Stage C: CASAL-lite
- Surpassed by the current rewrite in the sense that the split update is now explicit and uses `(f, k)` residuals.

### Stage D: Faithful time-dependent CASAL-KLDM
- Partially implemented.
- The coupling and dual updates are now explicit, but the KLDM reverse integrator is still used in operator-split form.

### Stage E: MH over templates
- Not implemented as MH; replaced with exhaustive discrete search over evaluated templates.

### Stage F: Score-MCMC correction
- Not implemented.

## 10. Faithful Codex pseudocode
- Mapped to the new top-level `sample_kldm_casal_chart(...)`.
- The projection helper now follows the “enumerate templates -> enumerate tau -> SSVD project -> validate -> choose best” structure.

## 11. Diagnostics
- Implemented logs now include:
  - step
  - `rho`
  - `tau`
  - template labels
  - origin shift
  - projection energy
  - coordinate loss
  - lattice-k loss
  - physical lattice loss
  - steric loss
  - minimum pair distance
  - detected SG
  - requested SG match
  - residual norms in coordinate and `k` space
  - dual norms in coordinate and `k` space
  - SSVD rank
  - SSVD condition number
  - SSVD delta norm

## 12. Things to look for in experiments
- Not code, but the needed diagnostics are now present for the checks described in the markdown.

## 13. Criteria for “faithful 1:1 CASAL-Chart-KLDM”
- Implemented:
  - split variables
  - physical coupling in `(f, k)`
  - torus residuals
  - projection over `(W, theta, tau, pi)`
  - SSVD fixed-chart solver
  - origin-gauge handling
  - final validation
  - detailed diagnostics
- Partially implemented:
  - time-dependent KLDM + CASAL update is still operator-split around the KLDM reverse step
- Implemented via exhaustive search instead of MH:
  - discrete template support

## 14. Current implementation classification
- Updated classification:
  - no longer the earlier selector-backed “CASAL-lite chart projection”
  - currently best described as:
    - `faithful chart projection + explicit (f,k) CASAL operator split`

## 14.1 CASAL paper check
- Checked against `cascal.pdf`.
- Algorithm 1 uses the split variables `x`, `z`, and `mu`, returns `z_T`, and updates:
  - `x_{t+1} = x_t - tau grad f(x_t) - tau rho (x_t - z_t + mu_t) + sqrt(2 tau) w_t`
  - `z_{t+1} = P_C(z_t - tau rho (z_t - x_{t+1} - mu_t))`
  - `mu_{t+1} = mu_t + (tau / rho) (x_{t+1} - z_{t+1})`
- The previous code used an unscaled raw dual update. It has been corrected so `casal_mu_eta` is a multiplier on the paper-scaled `(tau / rho)` update.
- This does not make the KLDM update fully fused; the implementation remains operator-split around KLDM's existing reverse step.

## 15. Recommended next code tasks
- Remaining high-value tasks if we want to close the last theoretical gap:
  1. replace operator-split KLDM/coupling with a single derived KLDM+CASAL ambient update
  2. optionally add stochastic MH chart moves on top of the exhaustive projection path
  3. add an explicit ground-truth projection notebook/test harness

## 16. Final summary
- The rewrite now follows the markdown’s intended chart machinery:
  - ambient KLDM state
  - physical `(f, k)` coupling
  - explicit SSVD chart projection
  - mandatory origin shifts
  - hard materialized validation
- The main remaining theoretical difference is the operator-split ambient KLDM update versus a fully fused time-dependent KLDM+CASAL integrator.
