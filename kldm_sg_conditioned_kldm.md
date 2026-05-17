# SG-Conditioned KLDM

This note records the repo-local implementation of a MatterGen-style
space-group-conditioned KLDM prior on top of the vanilla KLDM checkpoint at
`epoch_8900.pt`.

The goal is to learn

\[
p_{\theta,G}(x \mid a, G)
\]

from the existing unconditional KLDM prior

\[
p_{\theta,0}(x \mid a),
\]

without retraining the whole network from scratch.

## What is implemented

The conditioning path is implemented in
[src/kldmPlus/scoreNetwork/scoreNetwork.py](/Users/glymov/DTU/6%20Semester/Bachelor/Github/Main/kldm/src/kldmPlus/scoreNetwork/scoreNetwork.py)
and threaded through KLDM in
[src/kldmPlus/kldm.py](/Users/glymov/DTU/6%20Semester/Bachelor/Github/Main/kldm/src/kldmPlus/kldm.py).

The training config is
[configs/kldm_plus/mp_20/mp20_sg_adapter_finetune.yaml](/Users/glymov/DTU/6%20Semester/Bachelor/Github/Main/kldm/configs/kldm_plus/mp_20/mp20_sg_adapter_finetune.yaml).

The implementation follows the MatterGen adapter idea:

- add a property embedding for `space_group`
- add a small adapter MLP before each message-passing layer
- add a zero-initialized mix-in linear layer
- freeze the base model and train only the SG adapters first
- use classifier-free dropout by replacing the SG label with a null token

## Network change

`CSPVNet` now accepts:

- `sg_conditional: bool`
- `sg_emb_dim: int`
- `num_space_groups: int`

and `forward(...)` now optionally accepts:

- `space_group: Tensor | None`

The null token is:

- `0 = unconditional`
- `1..230 = actual space groups`

Per message-passing layer, the conditional update is:

\[
H^{(\ell)} \leftarrow H^{(\ell)} + f_{\text{mixin}}^{(\ell)}\!\left(
f_{\text{adapter}}^{(\ell)}(e_G)
\right)\mathbf{1}[G \neq 0].
\]

All mix-in layers are zero-initialized, so loading `epoch_8900.pt` with
`strict=False` starts from the unconditional model exactly.

## Training path

`ModelKLDM` now supports:

- `model.sg_condition_dropout`
- optional `space_group` conditioning inside training batches

During training:

- the KLDM loss stays unchanged
- `batch.space_group` is read from the dataset
- with probability `sg_condition_dropout`, the label is replaced by `0`

So the model learns both:

\[
p_{\theta}(x \mid a, G)
\quad\text{and}\quad
p_{\theta}(x \mid a, \varnothing).
\]

This is what makes classifier-free guidance possible at sampling time.

## Fine-tuning from epoch 8900

The repo config for this is:

- [configs/kldm_plus/mp_20/mp20_sg_adapter_finetune.yaml](/Users/glymov/DTU/6%20Semester/Bachelor/Github/Main/kldm/configs/kldm_plus/mp_20/mp20_sg_adapter_finetune.yaml)

Important settings:

- `checkpoint.resume_from: ../../../artifacts/HPC/checkpoints/epoch_8900.pt`
- `checkpoint.load_optimizer_state: false`
- `checkpoint.load_ema_state: false`
- `checkpoint.load_time_sampler_state: false`
- `finetune.freeze_base: true`
- `finetune.train_sg_adapters_only: true`
- `model.sg_condition_dropout: 0.2`
- `model.score_network.sg_conditional: true`

The optimizer only sees:

- `score_network.sg_embedding.*`
- `score_network.sg_adapters.*`
- `score_network.sg_mixins.*`

This filtering is implemented in
[src/kldmPlus/utils/model_loader.py](/Users/glymov/DTU/6%20Semester/Bachelor/Github/Main/kldm/src/kldmPlus/utils/model_loader.py).

## Sampling path

KLDM sampling now supports optional conditioning through:

- `space_group`
- `sg_guidance_scale`

in `sample_CSP_algorithm3(...)` and the internal reverse-chain helpers.

If `sg_guidance_scale = 1.0`, the sampler uses the conditional model directly.

If `sg_guidance_scale > 1.0`, it uses a classifier-free guidance combination:

\[
\hat{s}_\gamma = \hat{s}_{\varnothing} + \gamma(\hat{s}_{G} - \hat{s}_{\varnothing}).
\]

In repo terms, this is applied directly to:

- `pred_v`
- `pred_l`

before the usual KLDM reverse updates.

## How this plugs into DPnP

DPnPSVD now has two new knobs in
[src/kldmPlus/dpnpsvd.py](/Users/glymov/DTU/6%20Semester/Bachelor/Github/Main/kldm/src/kldmPlus/dpnpsvd.py):

- `sg_conditioned_dds: bool`
- `sg_guidance_scale: float`

When `sg_conditioned_dds: true`:

- the initial KLDM prior sample is drawn with requested `batch.space_group`
- the ambient DDS repair step also runs with requested `batch.space_group`
- the PCS kernel stays unchanged

So the new outer loop is:

\[
x_k \xrightarrow{\mathrm{PCS}_G} x_{k+1/2}
\xrightarrow{\mathrm{DDS}_{\theta,G}} x_{k+1}.
\]

This is the intended split:

- PCS enforces exact requested-SG support on the Wyckoff charts
- DDS uses a learned prior already biased toward the requested SG

## What to run

First train the adapter:

```bash
python3.11 src/kldmPlus/run_experiment.py \
  --config configs/kldm_plus/mp_20/mp20_sg_adapter_finetune.yaml
```

Then evaluate three modes:

1. SG-KLDM only
2. DPnP + SG-conditioned DDS, no pair-distance
3. DPnP + SG-conditioned DDS + pair-distance

The two existing faithful DPnP configs now expose the DDS knobs:

- [configs/kldm_plus/mp_20/mp20_sampling_dpnp_debug_oracle.yaml](/Users/glymov/DTU/6%20Semester/Bachelor/Github/Main/kldm/configs/kldm_plus/mp_20/mp20_sampling_dpnp_debug_oracle.yaml)
- [configs/kldm_plus/mp_20/mp20_sampling_dpnp_real.yaml](/Users/glymov/DTU/6%20Semester/Bachelor/Github/Main/kldm/configs/kldm_plus/mp_20/mp20_sampling_dpnp_real.yaml)

Set:

- `sg_conditioned_dds: true`
- `sg_guidance_scale: 1.0` first

and point `sampling_compare.checkpoint_path` at the SG-conditioned fine-tuned
checkpoint.

## Why this does not replace DPnP

Normal SG-conditioned reverse sampling targets

\[
p_{\theta,G}(x \mid a, G),
\]

which is still a soft conditional model.

DPnP targets the constrained posterior

\[
\pi_G(x \mid a)
\propto
p_{\theta,G}(x \mid a, G)\,
\mathbf{1}\{\mathrm{SG}(x)=G\}\,
\mathbf{1}\{d_{\min}(x)\ge d_0\}\,
\exp(-E_{\mathrm{phys}}(x)).
\]

So SG-conditioned KLDM should make DDS less destructive, but PCS is still the
piece that enforces exact requested-SG support and hard close-contact control.
