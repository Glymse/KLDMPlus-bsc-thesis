# Algorithm 20 Constrained Oracle Prior Notes

This note records the current interpretation of the constrained-init
Algorithm 20 experiments in:

- [test_algorithm20_constrained_init_oracle_prior.ipynb](/Users/glymov/DTU/6%20Semester/Bachelor/Github/Main/kldm/notebooks/test_algorithm20_constrained_init_oracle_prior.ipynb)

## Current Mode

For now we run **Mode A: frozen-lattice coordinate PPR**.

This is a debugging ablation. The lattice is treated as fixed conditioning, not
as a projection variable.

Concretely, for the reverse chain and every PPR call we keep:

```text
l_t = l_seed
```

and we do **not** run lattice reverse updates during this experiment.

This means the method currently tests:

```text
p(f_t, v_t | l_seed, a, T)
```

rather than a fully coupled coordinate+lattice constrained sampler.

## Best Non-Lattice PPR Formulation

The current method should be described as:

```text
coordinate-only KLDM-PPR conditioned on a fixed or native lattice path
```

with objective:

```text
min_{xi_r, xi_v, q}
rho_W(
  B[
    D_f(
      f_t(xi_r, xi_v),
      v_t(xi_v),
      stopgrad(l_t),
      a,
      t
    )
  ],
  Phi_T(q)
)
+ lambda(t) (||xi_r||^2 + ||xi_v||^2)
```

The important modeling decision is:

```text
stopgrad(l_t)
```

The lattice is only conditioning here. It is not optimized or projected inside
Algorithm 20.

## What This Ablation Answers

This mode isolates the cleanest coordinate question:

```text
Can coordinate PPR work if the lattice conditioning is stable and correct?
```

For oracle testing, the fixed lattice can be:

- the GT lattice
- the CrystalFormer oracle seed lattice
- a PyXtal seed lattice

For real testing later, the fixed lattice should be the proposed prior lattice.

## Cheap Lattice-Path Ablation

To check whether the observed witness improvements depend strongly on the
lattice path, the notebook now includes a cheap three-mode comparison:

### Mode A: frozen lattice

```text
l_t = l_seed
```

for all reverse and PPR steps, with no lattice reverse update.

### Mode B: native lattice

Start from `l_seed`, then let KLDM update `l_t` normally during reverse.
Algorithm 20 still treats `l_t` as:

```text
stopgrad(l_t)
```

conditioning only.

### Mode C: consistent noised lattice

Start from `l_seed`, forward-noise the seed lattice to `t_start`, and then
reverse `l_t` normally. Algorithm 20 still does not project lattice.

This is intentionally run cheaply:

- PPR-only
- small projection budget
- same graph/init modes as the main constrained oracle experiment

The point is not to produce a final sampler result, but to answer:

```text
how much of the witness behavior is coming from the coordinate correction
versus the lattice path?
```

## What The Current Results Say

The constrained-init facitEM experiments now suggest:

1. With an oracle/template-compatible prior, late soft Algorithm 20 can
   repeatedly reduce the witness loss through the reverse chain.
2. The improvement usually survives renoising at lower times, even if the
   anchor is not strictly feasible.
3. Random template priors also improve, but they plateau much higher than the
   oracle prior, which shows the quality of the initial basin matters.
4. The witness metric is improving more reliably than endpoint RMSE, so for now
   the witness residual should be treated as the primary debugging signal.

In short:

```text
the coordinate-only PPR mechanism appears locally useful when the prior is in
the right symmetry basin
```

but we are not yet at the stage where strict feasibility or final structural
quality is consistently solved.
