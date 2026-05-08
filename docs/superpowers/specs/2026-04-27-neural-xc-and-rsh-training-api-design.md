# Neural XC and RSH Training API Design

## Goal

Define a concise public API for functional optimization while keeping RSH parameter optimization and neural XC training as two separate training routes.

The public naming should use `neural_xc`, not `dm21_like`, because the current functional family is broader than a DM21-like implementation.

## Public Naming

Use `td_graddft.neural_xc` for neural XC construction:

```python
from td_graddft import neural_xc

functional = neural_xc.Functional(
    hidden_dims=(64, 64),
    input_feature_mode="dm21_original",
    architecture="residual",
)
```

Also expose a factory for users who prefer function-style construction:

```python
functional = neural_xc.make_functional(
    hidden_dims=(64, 64),
    input_feature_mode="dm21_original",
    architecture="residual",
)
```

`dm21_original` remains valid as a feature mode name. It describes a feature set, not the public functional type.

## RSH Parameter Optimization

RSH optimization is a small-parameter optimization route over physical parameters such as `omega`, `alpha`, and `beta`.

```python
from td_graddft import gto, nn_rsh, training

mol = gto.M(..., basis="6-31g*")

functional = nn_rsh.RSH("lc-wpbe").trainable(
    params=("omega", "alpha", "beta"),
)

result = training.RSHOptimizer(
    functional=functional,
    molecules=[mol],
).kernel(
    steps=500,
    learning_rate=1e-3,
    loss="koopmans_ip_ea",
)
```

Required history fields:

- `loss`
- `omega`
- `alpha`
- `beta`
- `ip_error`
- `ea_error`

## Neural XC Training

Neural XC training is a network-parameter training route over `neural_xc.Functional` parameters.

```python
from td_graddft import gto, neural_xc, training

mol = gto.M(..., basis="6-31g*")

functional = neural_xc.Functional(
    hidden_dims=(64, 64),
    input_feature_mode="dm21_original",
    architecture="residual",
)

result = training.NeuralXCTrainer(
    functional=functional,
    molecules=[mol],
).kernel(
    steps=1000,
    learning_rate=1e-4,
    loss="ground_state",
    scf_gradient_mode="unrolled",
)
```

Required history fields:

- `loss`
- `energy_mae`
- `density_mse`
- `orbital_energy_mae`
- `scf_cycles`
- `scf_converged`

## Shared Result Shape

Both routes may return a shared result container:

```python
result.functional
result.params
result.history
result.final_metrics
```

The optimizer classes remain separate:

- `training.RSHOptimizer`
- `training.NeuralXCTrainer`

## Compatibility

Keep existing implementation-level names during migration:

- `DM21LikeFunctional` remains as a compatibility alias.
- `BoundDM21LikeFunctional` remains as a compatibility alias.
- `make_dm21_like_functional(...)` remains as a deprecated wrapper around `make_neural_xc_functional(...)`.

New docs and examples should use only:

- `neural_xc.Functional`
- `neural_xc.make_functional`
- `training.NeuralXCTrainer`
- `training.RSHOptimizer`

## File Structure

Do not create a new `td_graddft/xc/` package because `src/td_graddft/xc.py` already exists.

Add thin public API files:

```text
src/td_graddft/
  neural_xc/
    __init__.py
    api.py

  nn_rsh/
    api.py

  training/
    results.py
    neural_xc_trainer.py
    rsh_optimizer.py
```

Reuse existing implementation layers:

```text
neural_xc.api
  -> neural_xc.dm21.functional
  -> neural_xc.base.functional

nn_rsh.api
  -> nn_rsh.functional
  -> nn_rsh.losses

training.NeuralXCTrainer
  -> training.losses
  -> training.trainer
  -> workflows.core training helpers where useful

training.RSHOptimizer
  -> nn_rsh.losses
  -> nn_rsh.functional
```

## Scope

This design only standardizes the public API and naming. It does not change the numerical functional form, loss definitions, SCF implementation, or TDDFT implementation.

