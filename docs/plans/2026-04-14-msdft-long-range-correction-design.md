# MSDFT-Style Long-Range Correction for TD-GradDFT

**Date:** 2026-04-14
**Author:** Brainstorming session with STOKES-DOT
**Status:** Design approved, ready for implementation

---

## Overview

This document describes the design for integrating MSDFT-style long-range correction into TD-GradDFT. The goal is to improve excited-state predictions (excitation energies and oscillator strengths) through a two-stage training approach:

1. **Stage 1:** Ground-state NeuralXC training (pure ground-state supervision)
2. **Stage 2:** Excited-state fine-tuning of long-range correction network

The key innovation is using a **real-space long-range correction** that avoids the frequency-dependence loop problem in TDDFT while fixing the adiabatic approximation's limitations.

---

## Problem Statement

### Current Limitations

TD-GradDFT has a differentiable architecture for backpropagating S1 excited-state loss to neural network XC parameters, but:

- Limited to **single excited state** (S1)
- Ground-state trained functionals may not generalize well to excited states
- Adiabatic approximation in XC kernel fails for:
  - Charge-transfer excitations
  - Rydberg states
  - Long-range electron-hole interactions

### Design Goals

1. Pure ground-state training for NeuralXC (Stage 1)
2. Excited-state fine-tuning for correction only (Stage 2)
3. Avoid frequency-dependence loop in TDDFT
4. Minimal changes to existing codebase

---

## Architecture

### Three-Layer Separation

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Ground-state NeuralXC (unchanged)                  │
│  Input: ρ(r), ∇ρ(r), ∇²ρ(r)                                  │
│  Output: θ* (trained parameters)                             │
│  Loss: |E_pred - E_ref|²                                     │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Long-Range Correction Module (NEW)                 │
│  Input: [ρ(r), ρ(r'), ∇ρ(r), ∇ρ(r'), |r-r'|]                  │
│  Output: δK_xc (real-space XC kernel correction)             │
│  Form: Analytic framework + Neural residual                  │
└─────────────────────────────────────────────────────────────┘
                           ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: TDDFT Response Solver (enhanced)                   │
│  Casida/TDA: A → A + δK_xc                                   │
│  Solve: [A + δK_xc]X = ωX                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Long-Range Correction Module

### Correction Formula

$$\delta f_{xc}(r, r') = -\alpha(r, r') \cdot \frac{\exp(-\gamma(r, r') \cdot |r - r'|)}{|r - r'|}$$

Where:
- $\alpha(r, r')$: correction strength (spatially dependent)
- $\gamma(r, r')$: screening parameter (spatially dependent)

Both predicted by a dual-head neural network.

### Network Architecture

```
Input: [ρ(r), ρ(r'), ∇ρ(r), ∇ρ(r'), |r-r'|]  (dim=5)
    ↓
Shared Hidden Layers: 64 → 64 → 32  (swish activation)
    ↓
    ├──→ Alpha Head (dense → softplus)  → α(r, r')
    └──→ Gamma Head (dense → softplus)  → γ(r, r')
```

**Key Design Decisions:**
- Shared trunk for efficiency
- Softplus (or project-specific constrained softmax) for non-negativity
- Exchange symmetry: $f(r, r') = f(r', r)$

---

## Two-Stage Training Workflow

### Stage 1: Ground-State NeuralXC Training

**Status:** Existing code, no changes needed

```python
from td_graddft.workflows import ExperimentPipeline, ExperimentConfig

config = ExperimentConfig(
    experiment_name="ground_state_training",
    systems=[...],
    training=NeuralXCTrainingConfig(steps=1200, learning_rate=0.005),
)

pipeline = ExperimentPipeline(config)
result = pipeline.run()
neural_xc_params = result.neural_xc_params
```

**Output:** $\theta^*$ (trained NeuralXC parameters)

### Stage 2: Excited-State Fine-Tuning

**Status:** New module to implement

```python
from td_graddft.training.excited_state_trainer import (
    ExcitedStateFineTuner,
    ExcitedStateFineTuneConfig,
)

# Configure fine-tuning
es_config = ExcitedStateFineTuneConfig(
    steps=500,
    learning_rate=0.001,
    excited_states=[1, 2, 3],     # Fine-tune on S1, S2, S3
    weight_energy=1.0,             # Excitation energy loss weight
    weight_osc=0.5,                # Oscillator strength loss weight
    freeze_neural_xc=True,         # Keep ground-state parameters fixed
)

# Load excited-state reference data
excited_data = load_excited_state_reference(...)

# Initialize fine-tuner with Stage 1 parameters
fine_tuner = ExcitedStateFineTuner(es_config, neural_xc_params)

# Execute fine-tuning
lr_params = fine_tuner.fine_tune(excited_data)
```

**Output:** $(\theta^*, \alpha^*, \gamma^*)$ - complete model

### Loss Function (Stage 2)

$$\mathcal{L}_{\text{total}} = w_E \sum_{i \in \text{states}} |\omega_i^{\text{pred}} - \omega_i^{\text{ref}}|^2 + w_f \sum_{i \in \text{states}} |f_i^{\text{pred}} - f_i^{\text{ref}}|^2$$

Where:
- $\omega_i$: excitation energy
- $f_i$: oscillator strength
- $w_E, w_f$: configurable weights (default: 1.0, 0.5)

---

## Implementation Plan

### Phase 1: Core Modules

**File:** `td_graddft/tddft/long_range_correction.py`

```python
from flax import linen as nn
import jax.numpy as jnp

class LongRangeXCNet(nn.Module):
    """Dual-head network for spatially-dependent alpha and gamma"""

    shared_hidden: tuple = (64, 64, 32)
    alpha_head_dim: int = 1
    gamma_head_dim: int = 1

    @nn.compact
    def __call__(self, features):
        # Shared trunk
        x = features
        for dim in self.shared_hidden:
            x = nn.Dense(dim)(x)
            x = nn.swish(x)

        # Dual heads (use project-specific constrained transformation)
        alpha = self._constrained_output(x, self.alpha_head_dim)
        gamma = self._constrained_output(x, self.gamma_head_dim)

        return alpha, gamma

    def _constrained_output(self, x, dim):
        # TODO: Use TD-GradDFT's parameter constraint mechanism
        return nn.Dense(dim)(x)


def compute_lr_correction(alpha, gamma, r12):
    """Compute long-range correction value"""
    return -alpha * jnp.exp(-gamma * r12) / r12


def build_lr_correction_matrix(density_grid, coords, params):
    """Build complete XC kernel correction matrix"""
    # TODO: Implement
    # 1. Extract (r, r') pair features
    # 2. Predict alpha(r,r'), gamma(r,r') via network
    # 3. Compute correction matrix
    return delta_K_xc
```

**Tasks:**
1. Implement `LongRangeXCNet` with Flax
2. Integrate project's parameter constraint mechanism
3. Implement feature extraction for (r, r') pairs
4. Implement matrix assembly

**Estimated effort:** 3-5 days

---

### Phase 2: TDDFT Integration

**File:** `td_graddft/tddft/response.py` (modify)

```python
def build_xc_kernel(mf, lr_correction_params=None):
    """
    Build XC kernel with optional long-range correction

    Args:
        mf: mean-field object
        lr_correction_params: long-range correction network parameters (optional)
    """
    # 1. Compute adiabatic XC kernel (existing logic)
    K_xc_adia = build_adiabatic_xc_kernel(mf)

    # 2. Add long-range correction if provided
    if lr_correction_params is not None:
        from td_graddft.tddft.long_range_correction import build_lr_correction_matrix
        K_xc_lr = build_lr_correction_matrix(
            mf.density_grid,
            mf.coords,
            lr_correction_params
        )
        return K_xc_adia + K_xc_lr
    else:
        return K_xc_adia
```

**Tasks:**
1. Modify `build_xc_kernel()` to accept correction parameters
2. Add conditional logic for correction application
3. Ensure compatibility with existing Casida/TDA solvers

**Estimated effort:** 1-2 days

---

### Phase 3: Excited-State Fine-Tuner

**File:** `td_graddft/training/excited_state_trainer.py` (new)

```python
from dataclasses import dataclass
from jax import grad, jit
import optax

@dataclass
class ExcitedStateFineTuneConfig:
    """Configuration for excited-state fine-tuning"""
    steps: int = 500
    learning_rate: float = 0.001
    excited_states: tuple = (1, 2, 3)
    weight_energy: float = 1.0
    weight_osc: float = 0.5
    freeze_neural_xc: bool = True


class ExcitedStateFineTuner:
    """Fine-tune long-range correction on excited states"""

    def __init__(self, config, neural_xc_params):
        self.config = config
        self.neural_xc_params = neural_xc_params  # Fixed
        self.lr_correction_params = self._init_lr_net()

    def _init_lr_net(self):
        """Initialize long-range correction dual-head network"""
        from td_graddft.tddft.long_range_correction import LongRangeXCNet

        net = LongRangeXCNet(
            shared_hidden=(64, 64, 32),
            alpha_head_dim=1,
            gamma_head_dim=1
        )

        key = jax.random.PRNGKey(0)
        dummy_input = jnp.zeros((5,))
        return net.init(key, dummy_input)

    def compute_excited_loss(self, lr_params, system_data):
        """Compute excited-state loss"""
        # 1. Run SCF with fixed NeuralXC
        # 2. Solve TDDFT with long-range correction
        # 3. Compute loss vs reference
        # TODO: Implement
        pass

    def fine_tune(self, system_data):
        """Execute fine-tuning"""
        optimizer = optax.adam(self.config.learning_rate)
        opt_state = optimizer.init(self.lr_correction_params)

        @jit
        def train_step(lr_params, opt_state, system_data):
            loss, grads = jax.value_and_grad(
                self.compute_excited_loss
            )(lr_params, system_data)

            updates, opt_state = optimizer.update(grads, opt_state)
            lr_params = optax.apply_updates(lr_params, updates)

            return lr_params, opt_state, loss

        for step in range(self.config.steps):
            self.lr_correction_params, opt_state, loss = train_step(
                self.lr_correction_params, opt_state, system_data
            )

            if step % 50 == 0:
                print(f"Step {step}, Loss: {loss:.4f}")

        return self.lr_correction_params
```

**Tasks:**
1. Implement `ExcitedStateFineTuner` class
2. Implement `compute_excited_loss()` method
3. Implement training loop with JAX JIT
4. Add logging and checkpointing

**Estimated effort:** 3-4 days

---

### Phase 4: Data Loading

**File:** `td_graddft/training/excited_state_data.py` (new)

```python
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

@dataclass
class ExcitedStateReference:
    """Container for excited-state reference data"""
    system_name: str
    ground_state_energy: float
    excited_energies: Dict[int, float]      # {state_index: energy}
    oscillator_strengths: Dict[int, float]   # {state_index: strength}
    geometry: np.ndarray                     # Atomic coordinates
    basis: str                               # Basis set name


def load_excited_state_reference(source):
    """
    Load excited-state reference data

    Sources:
    - High-level calculations (EOM-CCSD, CASPT2)
    - Experimental spectra
    - Pre-computed datasets
    """
    # TODO: Implement multiple source formats
    pass


def load_from_eomccsd(filename):
    """Load from EOM-CCSD calculation output"""
    pass


def load_from_experiment(filename):
    """Load from experimental spectrum"""
    pass
```

**Tasks:**
1. Define data structures for excited-state references
2. Implement loaders for common formats (EOM-CCSD, experimental)
3. Add validation and error handling

**Estimated effort:** 2-3 days

---

### Phase 5: Workflow Integration

**File:** `td_graddft/workflows/presets.py` (modify)

Add new preset function:

```python
def two_stage_training_experiment(
    system_name: str,
    basis: str = "6-31g",
    gs_steps: int = 1200,
    es_steps: int = 500,
    excited_states: tuple = (1, 2, 3),
):
    """
    Complete two-stage training: ground-state → excited-state fine-tune

    Returns:
        dict: {'neural_xc': θ*, 'lr_correction': (α*, γ*)}
    """
    # ===== Stage 1: Ground-state training =====
    print("=== Stage 1: Ground-state NeuralXC training ===")
    gs_config = ExperimentConfig(
        experiment_name=f"{system_name}_two_stage_gs",
        systems=[...],
        training=NeuralXCTrainingConfig(steps=gs_steps),
    )
    gs_pipeline = ExperimentPipeline(gs_config)
    gs_result = gs_pipeline.run()
    neural_xc_params = gs_result.neural_xc_params

    # ===== Stage 2: Excited-state fine-tuning =====
    print("=== Stage 2: Excited-state fine-tuning ===")
    es_config = ExcitedStateFineTuneConfig(
        steps=es_steps,
        learning_rate=0.001,
        excited_states=excited_states,
    )

    excited_data = load_excited_state_reference(
        system=system_name,
        basis=basis,
    )

    from td_graddft.training.excited_state_trainer import ExcitedStateFineTuner
    fine_tuner = ExcitedStateFineTuner(es_config, neural_xc_params)
    lr_params = fine_tuner.fine_tune(excited_data)

    # ===== Save complete model =====
    final_model = {
        'neural_xc': neural_xc_params,
        'lr_correction': lr_params,
    }

    return final_model
```

**Tasks:**
1. Implement two-stage training preset
2. Add model checkpointing
3. Add result reporting and visualization

**Estimated effort:** 1-2 days

---

## Testing Strategy

### Unit Tests

| Module | Tests |
|--------|-------|
| `long_range_correction.py` | Network forward pass; correction value computation; symmetry constraints |
| `response.py` | Kernel assembly with/without correction; backward compatibility |
| `excited_state_trainer.py` | Loss computation; gradient flow; parameter freezing |

### Integration Tests

1. **Smoke test:** Small molecule (H₂O) with minimal basis
2. **Regression test:** Compare excited-state energies before/after correction
3. **End-to-end test:** Full two-stage training workflow

### Validation Tests

1. Compare with standard TD-DFT (no correction)
2. Compare with high-level reference (EOM-CCSD) if available
3. Validate physical constraints (sum rules, oscillator strengths)

---

## Dependencies

**No new external dependencies required.**

Uses existing:
- JAX/Flax (neural networks)
- Optax (optimization)
- GradDFT (ground-state)
- jax_xc (XC functionals)

---

## Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Frequency-dependence loop | High | Avoided by using real-space correction |
| Training instability | Medium | Careful initialization; small learning rate |
| Increased compute cost | Low | Correction only applied in TDDFT stage |
| Backward compatibility | Low | Optional correction via parameter flag |

---

## Future Extensions

1. **Frequency-dependent correction** (self-consistent approach)
2. **Spin-flip TDDFT** support
3. **Non-adiabatic dynamics** applications
4. **Transfer learning** across molecules

---

## References

1. MSDFT Theory: WIREs Comput. Mol. Sci. 10.1002/wcms.70043
2. TD-GradDFT Architecture: `/docs/architecture.md`
3. Existing Training Pipeline: `/docs/plans/2026-03-25-architecture.md`

---

**Document Status:** Approved for implementation
**Next Steps:** Create detailed implementation plan with task breakdown
