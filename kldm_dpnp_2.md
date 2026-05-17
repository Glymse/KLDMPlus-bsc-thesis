# Faithful DPnP-KLDM v2

This note records the current intent and implementation shape for
`sampling_algorithm: 8` in
[src/kldmPlus/dpnpsvd.py](/Users/glymov/DTU/6%20Semester/Bachelor/Github/Main/kldm/src/kldmPlus/dpnpsvd.py),
paired with the active config
[configs/kldm_plus/mp_20/mp20_sampling_dpnp.yaml](/Users/glymov/DTU/6%20Semester/Bachelor/Github/Main/kldm/configs/kldm_plus/mp_20/mp20_sampling_dpnp.yaml).

The important split is:

- PCS lives on the Wyckoff-chart union and enforces requested-SG structure.
- DDS is an ambient KLDM reverse chain initialized from the PCS materialization.
- In faithful mode, extra post-hoc refinement and anchor pair-distance
  heuristics are disabled.
- Because the learned \(P(a, W \mid G)\) MLP is not trained yet, the active
  config currently uses a temporary oracle-surrogate only for the discrete
  template-prior/proposal path.

This is a change from the earlier chart-local repair interpretation.

## Purpose of this note

This document is meant to be easy to fact-check against the code. It therefore
does two things:

- states the intended mathematics of the DPnP-KLDM pipeline,
- states the concrete approximations and exact config values used by the
  current implementation.

When the implementation differs from an idealized derivation, that difference
is called out explicitly.

## Faithful mode versus temporary surrogate prior

There are now two ideas that need to stay separate:

- the DPnP kernel itself,
- the source of the discrete template prior over \(W\).

In faithful mode, the sampler kernel is forced to stay close to the DPnP
factorization:

- PCS only uses the chart residual, steric penalty, volume penalty, and local
  Jacobian metric term,
- DDS only uses the KLDM denoising posterior step at the same \(\eta_k\),
- the extra pair-distance anchor heuristic is disabled,
- the extra final fixed-template cleanup pass is disabled,
- oracle orbit reranking/filtering are disabled.

The active config still uses a temporary oracle-surrogate for template prior
selection because the learned \(P(a,W \mid G)\) MLP is not trained yet. This
surrogate only affects the discrete template proposal/initialization path. It
does not alter the PCS energy or the DDS kernel.

## Active config values

The current `DPnPSVD` sampling config is:

- `sampling_algorithm: 8`
- `n_steps: 1000`
- `t_start: 1.0`
- `t_final: 1.0e-3`

Outer DPnP schedule:

- `outer_steps: 4`
- `eta_start: 0.06`
- `eta_end: 0.012`

So the actual outer-loop noise levels are:

\[
\eta_1 = 0.06,\qquad
\eta_2 = 0.044,\qquad
\eta_3 = 0.028,\qquad
\eta_4 = 0.012.
\]

PCS parameters:

- `pcs_mh_steps: 300`
- `final_pcs_mh_steps: 500`
- `faithful_dpnp: true`
- `final_fixed_template_refine: false`
- `svd_step_size: 2.0e-5`
- `svd_damping: 2.0e-1`
- `coord_weight: 3.0`
- `lattice_weight: 10.0`
- `residual_volume_weight: 1.0`
- `steric_weight: 40`
- `pair_distance_weight: 0.0`
- `volume_weight: 2.0`
- `theta_proposal_free_std: 0.03`
- `theta_proposal_lattice_std: 0.03`
- `template_move_probability: 0.1`
- `template_proposal_temperature: 0.75`
- `template_proposal_epsilon: 0.05`
- `template_init_restarts: 12`
- `template_prior_weight: 1.0`
- `template_prior_mode: oracle_surrogate`
- `oracle_template_prior_success_prob: 0.95`
- `steric_softplus_tau: 0.1`

Template/oracle settings:

- `oracle_template_orbit_rerank: false`
- `oracle_template_orbit_filter: false`

Ambient DDS parameters:

- `ambient_dds_steps: 150`
- `ambient_dds_t_final: 1.0e-3`
- `ambient_dds_velocity_steps: 50`
- `ambient_dds_velocity_step_size: 5.0e-5`

Fixed-template debug diagnostics:

- `debug_fixed_template_multistart_restarts: 0`
- `debug_fixed_template_multistart_steps: 0`
- `debug_fixed_template_multistart_eta: 0.0`

Code defaults that remain active unless overridden:

- `min_distance = 1.0`
- `outer_hard_reject_distance_ratio = 1.0`

So the outer PCS hard-reject floor is effectively

\[
1.0 \times 1.0 = 1.0.
\]

## Target posterior

For composition \(a\) and requested space group \(G\), the desired constrained
posterior is

\[
\pi_G(x_0 \mid a)
\propto
p_{\mathrm{KLDM}}(x_0 \mid a)\,
\mathbf{1}\{\operatorname{SG}(x_0)=G\}\,
\mathbf{1}\{d_{\min}(x_0)\ge d_0\}\,
\exp\!\left(-E_{\mathrm{phys}}(x_0)\right).
\]

Define the constraint factor

\[
\psi_G(x)
=
\mathbf{1}\{\operatorname{SG}(x)=G\}\,
\mathbf{1}\{d_{\min}(x)\ge d_0\}\,
\exp\!\left(-E_{\mathrm{phys}}(x)\right).
\]

Then

\[
\pi_G(x_0 \mid a)
\propto
p_{\mathrm{KLDM}}(x_0\mid a)\,\psi_G(x_0).
\]

DPnP splits this into two kernels:

- PCS: enforce \(\psi_G\),
- DDS: inject \(p_{\mathrm{KLDM}}(x\mid a)\).

## State representation

We write a clean crystal state as

\[
x=(f,l),
\]

where:

- \(f \in \mathbb{T}^{3N}\) are fractional coordinates,
- \(l\) is the lattice representation used by KLDM.

In the current config:

- the dataset says `lattice_representation: kldm`,
- the model says `lattice_parameterization: eps`,
- the model says `lattice_diffusion_type: VP`.

So the current lattice branch is interpreted as an ordinary VP diffusion branch
with \(\epsilon\)-prediction.

## Initialization

The sampler starts by drawing an unconstrained KLDM sample:

\[
x^{\mathrm{prior}} = (f^{\mathrm{prior}}, l^{\mathrm{prior}})
\sim p_{\mathrm{KLDM}}(x\mid a).
\]

In the code this comes from the ambient KLDM sampling path before any PCS step.

This initial sample is not yet forced to satisfy the requested space group.

## PCS state space: Wyckoff charts

PCS does not sample directly in unconstrained ambient coordinates. Instead it
uses a union of Wyckoff charts:

\[
\Omega_G
=
\bigcup_W \{W\}\times \Theta_W,
\]

where:

- \(W\) is a discrete Wyckoff template,
- \(\theta=(z,u)\) are the chart coordinates,
- \(z\) are free Wyckoff coordinates,
- \(u\) are free lattice variables.

The materialization map is

\[
x=\Phi_{G,W}(\theta).
\]

More explicitly:

\[
\Phi_{G,W}(\theta)
=
\bigl(f_{G,W}(\theta), l_{G,W}(\theta)\bigr).
\]

So PCS evolves a chart state \((W,\theta)\), then materializes it to a crystal.

## PCS target density

At outer step \(k\), PCS samples from an energy-based density

\[
\pi_{\mathrm{PCS},k}(W,\theta\mid x_k)
\propto
\exp\!\left(-E_{\mathrm{PCS},k}(W,\theta;x_k)\right).
\]

The intended form is

\[
E_{\mathrm{PCS},k}(W,\theta;x_k)
=
\frac{1}{2\eta_k^2}\rho_W(\theta;x_k)^2
+ \lambda_s E_{\mathrm{steric}}(\Phi(\theta))
+ \lambda_v E_{\mathrm{vol}}(\Phi(\theta))
- \log J_W(\theta).
\]

That is also the right mental model for the implementation, but the exact
residual used in code is slightly more specific.

## Actual PCS residual used by the implementation

The implementation constructs a residual vector in `_proposal_view(...)`.

It uses:

1. wrapped coordinate residual,
2. lattice free-variable residual,
3. log-volume residual.

### Coordinate residual

Let \(f(\theta)\) be the materialized fractional coordinates and let
\(f_{\mathrm{anchor}}\) be the current ambient anchor coordinates. Then

\[
r_f(\theta)
=
\operatorname{wrap}\!\left(f(\theta)-f_{\mathrm{anchor}}\right).
\]

This term is weighted by

\[
w_f = \texttt{coord\_weight} = 3.0.
\]

### Lattice residual

The implementation does not compare full ambient lattice features directly at
this point. It compares the free lattice chart variables \(u\) to the anchor
free variables \(u_{\mathrm{anchor}}\):

\[
r_u(\theta)=u-u_{\mathrm{anchor}}.
\]

This term is weighted by

\[
w_u = \texttt{lattice\_weight} = 10.0.
\]

### Volume residual

The code also adds a residual on the log volume ratio:

\[
r_V(\theta)
=
\log \frac{V(\theta)}{V_{\mathrm{anchor}}}.
\]

This is weighted by

\[
w_V = \texttt{residual\_volume\_weight} = 1.0.
\]

### Residual vector

So the actual residual vector is effectively

\[
\rho(\theta)
=
\begin{bmatrix}
\sqrt{3.0}\,r_f(\theta) \\
\sqrt{10.0}\,r_u(\theta) \\
\sqrt{1.0}\,r_V(\theta)
\end{bmatrix},
\]

and the proximal energy is

\[
E_{\mathrm{prox}}(\theta)
=
\frac{\|\rho(\theta)\|^2}{2\eta_k^2}.
\]

This is important for fact-checking: the lattice residual is not written
directly in ambient \(l\)-space here. It is written in chart free-variable
space plus a separate log-volume term.

## Steric and volume terms in PCS

The remaining PCS energy terms are:

\[
\lambda_s E_{\mathrm{steric}}(\theta)
+ \lambda_v E_{\mathrm{vol}}(\theta).
\]

The active weights are:

\[
\lambda_s = 40,\qquad \lambda_v = 2.0.
\]

### Steric loss

The steric loss is a smooth close-contact penalty based on pair distances,
using:

- `min_distance = 1.0`
- `steric_softplus_tau = 0.1`

So close contacts below 1.0 are penalized softly by the steric term, and some
states may also be hard-rejected depending on the phase of PCS.

### Volume loss

The volume term is a separate penalty on bad cell-volume behavior. In code this
is built through the volume-ratio loss path rather than being folded directly
into the proximal residual.

## Jacobian term and SVD metric

PCS includes a chart volume correction term

\[
-\log J_W(\theta).
\]

The implementation builds a local metric from the residual Jacobian

\[
J_\rho(\theta)=\frac{\partial \rho(\theta)}{\partial \theta},
\]

and computes an SVD

\[
J_\rho = U \Sigma V^\top.
\]

With damping

\[
\lambda = \texttt{svd\_damping} = 0.2,
\]

the code uses the preconditioner

\[
P(\theta)
=
\left(J_\rho(\theta)^\top J_\rho(\theta)+\lambda I\right)^{-1}
=
V(\Sigma^2+\lambda I)^{-1}V^\top.
\]

This same SVD structure enters both:

- the proposal geometry,
- the Jacobian-style correction in the target.

## PCS continuous proposal kernel

At fixed template \(W\), PCS uses a preconditioned Langevin/MALA-style update
for \(\theta\).

Let the local proposal step size be

\[
h = \texttt{svd\_step\_size} = 2\times 10^{-5}.
\]

Then the forward proposal mean is approximately

\[
m(\theta)
=
\theta
- \frac{h}{2} P(\theta)\,\nabla E_{\mathrm{proposal}}(\theta),
\]

and the actual proposal is

\[
\theta'
=
m(\theta)+\sqrt{h}\,P(\theta)^{1/2}\xi,
\qquad
\xi\sim\mathcal N(0,I).
\]

The move is then accepted with a Metropolis-Hastings correction:

\[
\alpha(\theta,\theta')
=
\min\left(
1,
\exp\!\left[-E(\theta')+E(\theta)\right]
\frac{q(\theta\mid \theta')}{q(\theta'\mid \theta)}
\right).
\]

So even though the proposal is local and geometry-aware, the target is still
corrected by MH acceptance.

## PCS template moves

PCS can also propose a discrete template move \(W \to W'\).

This happens with probability

\[
p_{\mathrm{move}}
=
\texttt{template\_move\_probability}
=
0.1.
\]

So:

- with probability \(0.9\), do a continuous \(\theta\)-proposal in the current
  chart,
- with probability \(0.1\), propose a different template.

Template proposals are built from a softened categorical distribution with:

- `template_proposal_temperature = 0.75`
- `template_proposal_epsilon = 0.05`

In addition to the ranker score, the implementation can add a template prior
bonus derived from dataset template counts:

\[
\ell_{\mathrm{proposal}}(W)
=
\ell_{\mathrm{ranker}}(W)
+
\lambda_W \log\!\left(1 + \mathrm{count}(W\mid G,a)\right),
\]

with

\[
\lambda_W = \texttt{template\_prior\_weight} = 1.0.
\]

Conceptually:

\[
q(W')
=
(1-\epsilon)\,
\operatorname{softmax}\!\left(\frac{\ell(W')}{\tau}\right)
+ \epsilon \frac{1}{|T|},
\]

with

\[
\tau = 0.75,\qquad \epsilon = 0.05.
\]

In faithful mode, both of the following stay off:

- `oracle_template_orbit_rerank: false`
- `oracle_template_orbit_filter: false`

template moves are no longer allowed to depend directly on target-orbit oracle
information inside the PCS kernel.

When `template_prior_mode: oracle_surrogate` is active, oracle information is
used only as a temporary stand-in for the missing learned discrete prior over
templates. The active config approximates a "correct-template prior works
95% of the time" assumption by strongly biasing the discrete template proposal
toward orbit-matching templates with probability `0.95`, while leaving the PCS
and DDS kernels unchanged.

## PCS hard rejection rules

A materialized PCS proposal is rejected if it violates the requested manifold
or basic physical sanity.

The implementation rejects states when:

- the requested SG check fails,
- the lattice or coordinates are non-finite,
- pair distances are too small.

There are two relevant thresholds:

1. soft steric penalty threshold:
\[
d_0 = \texttt{min\_distance} = 1.0,
\]
2. outer-loop hard reject threshold:
\[
d_{\mathrm{hard,outer}}
=
\texttt{outer\_hard\_reject\_distance\_ratio}
\times
\texttt{min\_distance}
=
1.0.
\]

In earlier experimental configs, PCS also included a weak pair-distance
consistency term against the current ambient anchor. In the active faithful
config this term is explicitly disabled:

- `pair_distance_weight: 0.0`

So the active theorem-facing PCS energy does not include this heuristic
local-environment stabilizer.

So outer PCS can hard-reject if

\[
d_{\min} < 1.0.
\]

Final PCS uses the stricter final validity requirement against
\(\texttt{min\_distance}=1.0\).

## DDS is now ambient KLDM

After PCS returns a materialized state

\[
x_h = x_{k+\frac12} = (f_h, l_h),
\]

the DDS step no longer performs a chart-local repair update.

Instead it does:

1. rebuild a KLDM-compatible graph batch from \(x_h\),
2. map the shared DPnP noise level \(\eta_k\) to a VP time \(t_k\),
3. construct an ambient noisy anchor,
4. sample an initial velocity,
5. run the ordinary KLDM reverse chain from \(t_k\) down to
   `ambient_dds_t_final`.

This is implemented in `_dds_kldm_ambient_kernel(...)`.

## Matching the DPnP noise level to VP time

For the VP lattice branch, the forward kernel is

\[
l_t = \alpha(t) l_0 + \sigma(t)\epsilon_l,
\qquad
\epsilon_l \sim \mathcal N(0,I).
\]

Equivalently,

\[
\frac{l_t}{\alpha(t)}
=
l_0
+
\frac{\sigma(t)}{\alpha(t)}\epsilon_l.
\]

DPnP interprets the proximal noise level \(\eta_k\) through the relation

\[
\eta_k \approx \frac{\sigma(t_k)}{\alpha(t_k)}.
\]

For a KLDM-native DDS, time matching should respect both:

- the wrapped coordinate/TDM noise scale,
- the VP lattice width.

The current implementation therefore uses `_map_eta_to_kldm_time(...)` rather
than a lattice-only VP mapping. It chooses \(t_k\) by minimizing a combined
error over:

- the TDM wrapped coordinate noise scale,
- the lattice width \(\sigma(t)/\alpha(t)\).

So the same \(\eta_k\) used by PCS is reused to choose the DDS KLDM time.

## Lattice anchor used by DDS

Once \(t_k\) has been chosen, the code builds the lattice anchor as

\[
l_t^h = \alpha(t_k)\, l_h.
\]

This is the ordinary VP anchor.

The implementation explicitly avoids the MatterGen-style mean-shift anchor

\[
\alpha(t)l_h + (1-\alpha(t))\mu_N.
\]

So the lattice branch now follows the ordinary VP interpretation rather than a
MatterGen mean-shifted prior.

## Coordinate and velocity anchor

KLDM does not reverse-sample coordinates from \(f_t\) alone. Its coordinate
state includes velocity:

\[
z_t = (f_t, v_t, l_t).
\]

After PCS, the code sets:

\[
f_t^h = f_h,
\qquad
l_t^h = \alpha(t_k)\, l_h.
\]

It then needs an initial velocity

\[
v_t^h.
\]

## Conditional velocity initialization

The implementation approximates

\[
v_t^h \sim p_t(v \mid f_t=f_h, l_t=l_t^h, a)
\]

with a short score-based sampler in `_sample_conditional_velocity(...)`.

It uses:

- `ambient_dds_velocity_steps = 6`
- `ambient_dds_velocity_step_size = 5.0e-5`

The update is approximately

\[
v^{m+1}
=
v^m
+
\delta_v\, s_v(t_k, f_h, v^m, l_t^h, a)
+
\sqrt{2\delta_v}\,\xi^m,
\qquad
\xi^m \sim \mathcal N(0,I),
\]

with

\[
\delta_v = 5\times 10^{-5}.
\]

After each step, the code re-centers velocity graphwise so the per-graph mean
velocity is zero.

This is an approximation to the desired conditional velocity distribution, not
an exact direct draw.

## Fresh KLDM batch rebuild

One of the main implementation changes is that DDS now rebuilds a fresh
single-graph KLDM batch from the PCS materialization instead of trying to reuse
the old batch blindly.

This is done in `_build_chart_compatible_batch(...)`.

The rebuilt batch uses:

- `pos = f_h`
- `l = l_h`
- `atomic_numbers = a_h`
- `num_atoms = N_h`
- the same `FullyConnectedGraph` transform used by the CSP preprocessing path

In this repository, the standard CSP/KLDM preprocessing also uses fully
connected directed edges, so this keeps DDS aligned with the graph
construction used by the normal KLDM sampler rather than introducing a
separate manual edge builder.

So if PCS expanded or rearranged the representation, DDS still gets a batch
that is shape-compatible with the materialized structure.

This replaces the old pattern where a chart/model shape mismatch could lead to
an effectively zero-prior DDS step.

## Ambient DDS kernel used by algorithm 8

Operationally, the ambient DDS kernel is:

1. rebuild a KLDM batch from the PCS materialization,
2. compute \(t_k = \tau(\eta_k)\),
3. set
   \[
   f_t = f_h,\qquad l_t = \alpha(t_k) l_h,
   \]
4. sample
   \[
   v_t \approx p_t(v\mid f_t,l_t,a),
   \]
5. run KLDM reverse from \(t_k\) down to `ambient_dds_t_final = 10^{-3}`.

So conceptually:

\[
(f_{k+1}, l_{k+1})
=
\operatorname{KLDMReverse}
\left(
f_t^h,\,
v_t^h,\,
l_t^h,\,
t_k \to 10^{-3}
\right).
\]

The reverse chain length is

\[
\texttt{ambient\_dds\_steps} = 150.
\]

DDS is allowed to leave the requested space-group manifold. The next PCS step
is what pulls the chain back toward the requested constrained support.

## Lattice reverse score interpretation

Because the lattice branch is configured with:

- `lattice_diffusion_type: VP`
- `lattice_parameterization: eps`

the lattice score is interpreted as

\[
s_l(t,l_t)
\approx
-\frac{\widehat{\epsilon}_l}{\sigma(t)}.
\]

So the reverse VP drift has the standard form

\[
l_{t-\Delta t}
=
l_t
-
\left(
-\frac12 \beta(t) l_t
- \beta(t) s_l(t,l_t)
\right)\Delta t
+
\sqrt{\beta(t)\Delta t}\,\xi.
\]

Substituting the \(\epsilon\)-parameterized score gives

\[
l_{t-\Delta t}
=
l_t
-
\left(
-\frac12 \beta(t) l_t
+
\beta(t)\frac{\widehat{\epsilon}_l}{\sigma(t)}
\right)\Delta t
+
\sqrt{\beta(t)\Delta t}\,\xi.
\]

In practice this reverse integration is delegated to the KLDM reverse-sampling
implementation already present in the model code.

## Complete outer DPnP loop used now

Let \(x_k\) be the ambient state at outer step \(k\). The current algorithm is:

1. start from an unconstrained KLDM prior sample,
2. fit/initialize a chart state for the requested SG,
3. for \(k=1,2,3,4\):
   1. run PCS at \(\eta_k\),
   2. materialize \(x_h = x_{k+\frac12}\),
   3. run ambient KLDM DDS from the PCS materialization,
   4. set the DDS output to be the next ambient state \(x_{k+1}\),
4. optionally run a final fixed-template PCS refinement,
5. return the final PCS state if valid, else report failure.

Writing that more compactly:

\[
x_k
\xrightarrow{\mathrm{PCS}_{\eta_k}}
x_{k+\frac12}
\xrightarrow{\mathrm{DDS}_{\eta_k}}
x_{k+1}.
\]

The actual schedule is:

\[
\eta_1 = 0.06,\qquad
\eta_2 = 0.044,\qquad
\eta_3 = 0.028,\qquad
\eta_4 = 0.012.
\]

The final PCS pass, when enabled, uses:

\[
\texttt{final\_pcs\_mh\_steps} = 500.
\]

However, in the current evaluation config we explicitly set

- `final_fixed_template_refine: false`

because graph 2 often looked better before the last deterministic cleanup.

If the refinement is turned on, template moves are explicitly disabled:

\[
\texttt{template\_move\_probability} = 0
\]

for the refinement kernel only. So that stage becomes a fixed-template
continuous refinement rather than another chart-switching stage.

## Final return semantics

The sampler no longer silently returns the historical best-valid state from an
earlier iteration.

The current return rule is:

- if the final PCS state is valid, return it,
- if the final PCS state is invalid, report failure and return an invalid
  sample.

This is intentionally more honest than a hidden "best valid so far" fallback.

## Oracle diagnostics and logging

The implementation keeps oracle diagnostics available for debugging even when
oracle template reranking/filtering are off.

That means:

- real template selection no longer cheats with oracle structure information,
- but we can still evaluate step-by-step closeness to ground truth during
  debugging.

The current config is intentionally more conservative for PCS than the earlier
aggressive debug setting:

- lower `template_move_probability` to reduce template churn once a plausible
  chart has been found,
- higher `steric_weight` to push back earlier against close-contact collapse,
- a slightly softer schedule than the earlier `0.08 -> 0.01` run,
- stronger fixed-template search tooling with `final_pcs_mh_steps = 500`,
- final fixed-template refinement disabled in evaluation mode so it does not
  silently pull a good step-4 sample away from its best matcher basin,
- a separate debug-only fixed-template multi-start probe to test whether the
  current chart can reach a better continuous motif without DDS.

The current debug logs include:

- `kldm_dpnpsvd_oracle_step`
- `kldm_dpnpsvd_oracle_matcher`
- `kldm_dpnpsvd_oracle_delta`
- `kldm_dpnpsvd_oracle_effect`
- `kldm_dpnpsvd_pcs_alert`
- `kldm_dpnpsvd_ambient_dds_time`
- `kldm_dpnpsvd_best_oracle`
- `kldm_dpnpsvd_fixed_template_multistart`
- `kldm_dpnpsvd_fixed_template_multistart_restart`
- `kldm_dpnpsvd_fixed_template_multistart_best`

The delta log reports:

- `oracle_delta_frac_rmse`
- `oracle_delta_std_frac_rmse`
- `oracle_delta_min_pair`
- `oracle_delta_lengths_mae`
- `oracle_delta_angles_mae`

Each delta is reported as

\[
\text{after} - \text{before},
\]

where "before" is the previous oracle checkpoint for the same graph.

The `kldm_dpnpsvd_oracle_effect` line is a compact qualitative summary of
whether the current PCS or DDS phase improved or worsened:

- `frac_rmse`
- standardized matcher RMSE
- minimum pair distance
- lattice length error
- lattice angle error

based on the sign of the step-to-step delta.

The `kldm_dpnpsvd_pcs_alert` line is emitted when PCS accumulates close-contact
rejections. It is meant as a quick warning that the chain may be trapped at the
steric boundary, which was the main failure mode for graph 2 during debugging.

The `kldm_dpnpsvd_ambient_dds_time` line reports the time selected for ambient
DDS from the current \(\eta_k\), so it is easier to verify whether DDS is
using the intended KLDM-native time mapping.

The `kldm_dpnpsvd_best_oracle` line is a pure debugging summary. It reports:

- the best phase by standardized matcher RMSE,
- the best phase among checkpoints that actually matched.

This is meant to answer questions like "did step 4 PCS look best before final
cleanup?" without changing the returned sample.

The fixed-template multi-start debug block is also diagnostic-only. It keeps
the current template fixed, turns DDS off, samples multiple independent
continuous initializations, and runs PCS-only refinement under:

- fixed template,
- hard \(d_{\min}\ge 1.0\),
- low \(\eta\),
- no chart switching.

It does not change the returned structure in evaluation mode. It only tells us
whether a better same-template minimum seems reachable inside the current chart.

To keep repeated evaluation runs lighter, the current debug setting uses only:

- `4` fixed-template multi-start restarts,
- `300` PCS steps per restart,

rather than the earlier heavier `8 x 500` diagnostic pass.

The compare runner now also mirrors the complete stdout/stderr stream into a
single overwritten repo-root log file:

- `dpnp-log`

So every run leaves behind one fresh complete trace without accumulating old
logs across runs.

## Prior participation logging

The sampler now also makes KLDM-prior participation explicit.

Per graph, final logs include:

- `prior_available`
- `prior_used_steps`
- `compat_batch_steps`
- `prior_last_reason`

And globally:

- `prior_available_graphs`
- `prior_used_steps`
- `compat_batch_steps`

So a run can no longer appear to be "DPnP-KLDM" while the KLDM prior was
quietly skipped.

## What changed relative to the earlier chart-DDS version

The main changes are:

- `_chart_dds_step(...)` is no longer the main DDS path inside the outer loop,
- `_dds_kldm_ambient_kernel(...)` now performs the main DDS update,
- `_build_chart_compatible_batch(...)` rebuilds a KLDM batch from PCS
  materialization using the same `FullyConnectedGraph` transform used by CSP
  preprocessing,
- shape mismatch now triggers batch rebuild instead of an effective zero-prior
  DDS path,
- the final sampler output is the final PCS state or an explicit failure,
  rather than a silent best-valid fallback.

## What is still approximate

The pipeline is now much closer to the intended DPnP-KLDM story, but a few
pieces are still implementation-level approximations rather than exact
posterior sampling primitives.

### 1. PCS residual is implementation-specific

The code uses a residual composed of:

- wrapped coordinate residual,
- lattice free-variable residual,
- log-volume residual,

rather than a single abstract residual written directly in ambient
\((f,l)\)-space.

### 2. Conditional velocity sampling is approximate

The initial velocity \(v_t^h\) is not drawn from a closed-form conditional. It
is approximated by a short score-based sampler with 6 tiny updates.

### 3. DDS start state is built by state override

The code prepares a KLDM reverse state, then overwrites it with the PCS-derived
anchor \((f_t, v_t, l_t)\). This is operationally clear and intentional, but it
is still an implementation construction rather than a separately formalized
exact kernel object.

### 4. Legacy chart-DDS helper code still exists

Some chart-DDS support code still exists in
[src/kldmPlus/dpnpsvd.py](/Users/glymov/DTU/6%20Semester/Bachelor/Github/Main/kldm/src/kldmPlus/dpnpsvd.py),
but it is no longer the main outer-loop DDS path when `ambient_dds_steps > 0`.

## Compact summary

With the current config, algorithm 8 is best described as:

1. sample an unconstrained KLDM prior crystal,
2. move to a requested-SG Wyckoff chart representation,
3. alternate for 3 outer steps:
   - PCS on the chart union with \(\eta \in \{0.04, 0.025, 0.01\}\),
   - ambient KLDM DDS initialized from the PCS materialization,
4. run final PCS with 500 MH steps,
5. return final PCS if valid, otherwise fail explicitly.

That is the current implementation-level meaning of "DPnP-KLDM v2" in this
repository.
