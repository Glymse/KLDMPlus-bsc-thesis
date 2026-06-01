# Algorithm 19 Notebook Pipeline

This file summarizes the full debug pipeline used in
[notebooks/test_algorithm19_kldm_ppr_diffcsppp.ipynb](/Users/glymov/DTU/6%20Semester/Bachelor/Github/Main/kldm/notebooks/test_algorithm19_kldm_ppr_diffcsppp.ipynb).

## Goal

Algorithm 19, `KLDM-PPR-diffcsppp`, keeps the KLDMPLUS data/model pipeline but
uses a DiffCSP++-style symmetry backend as the source of truth for Wyckoff
constraints.

## Pipeline

1. Load KLDMPLUS data/model state with `SamplingCompareRunner`.
2. Select small debug graphs from the MP-20 compare batch.
3. Build the ground-truth structure with the existing KLDMPLUS sample helpers.
4. Build an oracle DiffCSP++ payload from that structure:
   - `spacegroup`
   - `ops`
   - `ops_inv`
   - `anchor_index`
   - `wyckoff_letters`
   - `atom_types`
   - `anchor_frac_coords`
   - `expanded_frac_coords`
5. Verify the symmetry backend before any PPR step:
   - torus utility checks
   - payload shape checks
   - expand identity
   - random anchor projection identity
   - GT residual with the correct payload
   - wrong-payload separation
   - finite gradient and boundary stress tests
6. Build a KLDM noisy state from ground truth with the native TDM forward kernel:
   - `model.tdm.sample_noisy_state(...)`
   - fixed lattice inside PPR
   - graph-wise mean-free velocity
7. Reconstruct clean fractional coordinates with the Algorithm 19 denoiser:
   - `D_f(F_t, V_t, L_t, A, t)`
   - default branch is the `minus` denoiser from the tutorial
8. Run one DiffCSP++ operator-aware PPR project step:
   - optimize `xi_r`, `xi_v`
   - evaluate the Wyckoff constraint on `D_f(...)`
   - keep the trust penalty in noise coordinates
9. Run the repeated PPR kernel:
   - project
   - form `F0*`
   - renoise through the native KLDM/TDM forward kernel
10. Evaluate endpoints with the same facitKLDM path:
   - `sample_evaluation.evaluate_csp_reconstruction(...)`
   - matcher-based `match` and `RMSE` via `StructureMatcher`
   - fractional `frac_RMSE` from the same evaluation stack

## Corrected Theory Notes

The current notebook now follows the corrected Algorithm 19 theory more
closely:

1. PPR is treated as a **coordinate/velocity corrector at fixed lattice**.
   Inside PPR we optimize only `(F_t, V_t)` and keep `L_t` frozen.
2. The DiffCSP++ payload is treated as the semantic source of truth:
   - `spacegroup`
   - `wyckoff_letters`
   - `anchor_index`
   - `wyckoff_ops`
   - `anchor_free_coordinate_masks`
3. The torch Wyckoff operator projection is **local and reference-based**:
   - expand a reference anchor chart
   - compute wrapped residuals in full-coordinate space
   - pull them back with `ops_inv`
   - average per orbit
   - update only free coordinates
4. Soft PPR is marked faithful only when the soft anchor is itself near-feasible:
   - `c_anchor_soft`
   - `soft_anchor_feasible`
   - `ppr_faithfulness`
5. The notebook blocks scientific interpretation when the GT structure is not
   already near-zero under `c_w_ops(...)` in the model frame.
6. The payload can carry explicit frame metadata in `debug_info`:
   - `model_to_payload_linear`
   - `model_to_payload_tau`
   - `model_to_payload_order`
   - `payload_to_model_linear`
   - `payload_to_model_tau`
   - `payload_to_model_order`

   This lets the DiffCSP++ operators stay in their standardized order while the
   loss is evaluated on KLDM model-frame coordinates.

## Notebook Sections

The notebook now has four practical sections:

1. **Phase 1: backend/operator checks**
   - payload shape and DOF masks
   - payload-frame identity
   - local free-anchor projection identity
   - model-frame GT identity precondition
   - SG mismatch hard-failure test
   - finite-difference projection check
   - perturbation stability

2. **Phase 1b: denoiser checks**
   - denoiser finiteness and GT proximity
   - oracle algebra
   - KLDM noise-chart identity
   - explicit `coordinate_score_mode`
   - GT-lattice isolation note

3. **Phase 2: local PPR controls**
   - `algo19_baseline_M0`
   - `algo19_renoise_only`
   - `algo19_project_step`
   - `algo19_kernel_step`
   - `algo19_wrong_template_kernel`
   - model-gradient routing check
   - project-only sweep over projection budgets

4. **Scientific summary**
   PPR is only promising when:
   - the frame precondition passes,
   - the soft anchor is feasible,
   - correct PPR beats renoise-only,
   - correct PPR beats the wrong-template control,
   - velocity means stay small,
   - lattice remains unchanged inside PPR.

## Debug split in the notebook

The notebook separates the work into two tracks:

1. **Backend/operator math**
   Verifies that the DiffCSP++ payload and operator projection behave correctly
   on their own.

2. **KLDM coupling**
   Tests whether the KLDM noisy-state optimization through `D_f` actually
   reduces the DiffCSP++ Wyckoff residual and improves endpoint metrics.

## Current interpretation rule

If backend/operator tests fail, the symmetry layer is wrong.

If backend/operator tests pass but KLDM coupling fails, the issue is in:
- frame/coupling between KLDM coordinates and the payload frame,
- denoiser algebra,
- or the PPR optimization itself.
