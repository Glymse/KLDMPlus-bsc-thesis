# KLDM DPnP

This file is the short authoritative description of what we mean by a
"faithful DPnP" implementation in this repo.

The detailed implementation note lives in
[kldm_dpnp_2.md](/Users/glymov/DTU/6%20Semester/Bachelor/Github/Main/kldm/kldm_dpnp_2.md).

## Faithful DPnP in this project

For `sampling_algorithm: 8`, the theorem-facing sampler should follow the
standard DPnP split:

\[
x_k
\xrightarrow{\mathrm{PCS}_{\eta_k}}
x_{k+\frac12}
\xrightarrow{\mathrm{DDS}_{\eta_k}}
x_{k+1}.
\]

with the same outer noise level \(\eta_k\) used by both kernels.

In our project:

- PCS operates on the union of Wyckoff charts for the requested space group,
- DDS operates in ambient KLDM space using the KLDM reverse process,
- the requested space group enters through the PCS chart/manifold and validity
  checks,
- the KLDM prior enters only through the DDS step,
- extra post-hoc cleanup stages are not part of faithful evaluation.

## What is excluded from faithful mode

The following are useful diagnostics or experiments, but are not part of the
faithful DPnP kernel:

- oracle orbit reranking/filtering inside PCS,
- anchor pair-distance heuristic terms inside PCS,
- final fixed-template cleanup passes after the outer DPnP chain,
- debug multistart probes affecting the returned sample.

## Temporary surrogate prior

The learned discrete prior \(P(a, W \mid G)\) is not trained yet.

So, for now, we allow a temporary oracle-surrogate only for the discrete
template-prior/proposal path. This is meant to stand in for a future learned
prior and should be described honestly as a surrogate, not as faithful
evaluation.

That surrogate is acceptable as long as:

- it affects only template proposal / initialization,
- it does not change the PCS energy,
- it does not change the DDS kernel,
- it is clearly labeled in logs and notes.
