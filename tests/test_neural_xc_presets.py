from td_graddft.jax_libxc import b3lyp_component_basis, b3lyp_component_coefficients
from td_graddft.neural_xc import (
    DEFAULT_NEURAL_XC_COEFFICIENT_PRIOR_MODE,
    DEFAULT_NEURAL_XC_COEFFICIENT_PRIOR_VALUES,
    DEFAULT_NEURAL_XC_DENSITY_SUPERVISION,
    DEFAULT_NEURAL_XC_SEMILOCAL_XC,
    resolve_coefficient_prior_values,
)


def test_default_neural_xc_basis_matches_component_basis() -> None:
    assert DEFAULT_NEURAL_XC_SEMILOCAL_XC == b3lyp_component_basis()
    assert DEFAULT_NEURAL_XC_COEFFICIENT_PRIOR_VALUES == b3lyp_component_coefficients()
    assert DEFAULT_NEURAL_XC_COEFFICIENT_PRIOR_MODE == "mean"
    assert DEFAULT_NEURAL_XC_DENSITY_SUPERVISION == "spin_resolved"


def test_resolve_coefficient_prior_values_uses_explicit_values() -> None:
    resolved = resolve_coefficient_prior_values(("lda_x",), (1.0, 2.0))

    assert resolved == (1.0, 2.0)


def test_resolve_coefficient_prior_values_uses_dm21_b3lyp_defaults() -> None:
    resolved = resolve_coefficient_prior_values(DEFAULT_NEURAL_XC_SEMILOCAL_XC)

    assert resolved == b3lyp_component_coefficients()


def test_resolve_coefficient_prior_values_returns_none_for_other_basis() -> None:
    resolved = resolve_coefficient_prior_values(("lda_x", "gga_x_pbe"))

    assert resolved is None


def test_resolve_coefficient_prior_values_returns_none_for_dldh_mode() -> None:
    resolved = resolve_coefficient_prior_values(
        DEFAULT_NEURAL_XC_SEMILOCAL_XC,
        energy_mode="dldh_two_lmf",
    )

    assert resolved is None


def test_resolve_coefficient_prior_values_returns_none_for_hybrid_head_mode() -> None:
    resolved = resolve_coefficient_prior_values(
        DEFAULT_NEURAL_XC_SEMILOCAL_XC,
        energy_mode="graddft_coeff_basis_hf_pt2_heads",
    )

    assert resolved is None
