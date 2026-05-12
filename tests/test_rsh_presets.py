import numpy as np
from td_graddft.nn_rsh import (
    get_rsh_functional_preset,
    list_rsh_functional_presets,
    make_rsh_template,
)


def test_lc_wpbe_preset_matches_literature_coefficients():
    preset = get_rsh_functional_preset("lc-wpbe")
    params = preset.default_params

    assert preset.canonical_xc_name == "LC_WPBE"
    assert preset.strict_jax_supported is True
    assert preset.jax_local_xc_spec == "lc_wpbe_local"
    assert np.allclose(
        tuple(float(x) for x in params.to_range_separated_coefficients()),
        (0.4, 1.0, -1.0),
    )
    assert np.allclose(
        tuple(float(x) for x in params.to_range_separated_hybrid_coefficients()),
        (0.4, 1.0, 0.0),
    )


def test_wb97xd_preset_matches_literature_coefficients():
    preset = get_rsh_functional_preset("wb97x-d")
    params = preset.default_params

    assert preset.canonical_xc_name == "WB97X_D"
    assert preset.has_dispersion is True
    assert np.allclose(
        tuple(float(x) for x in params.to_range_separated_coefficients()),
        (0.2, 1.0, -0.777964),
    )
    assert np.allclose(
        tuple(float(x) for x in params.to_range_separated_hybrid_coefficients()),
        (0.2, 1.0, 0.222036),
    )


def test_rsh_preset_aliases_and_templates_are_canonical():
    names = list_rsh_functional_presets()
    assert "lc-wpbe" in names
    assert "hse03" in names
    assert "hse06" in names
    assert "wb97x-d" in names

    lc = get_rsh_functional_preset("LC_WPBE")
    template = make_rsh_template("lc-wpbe")

    assert lc.name == "lc-wpbe"
    assert template.name == "lc-wpbe"
    assert template.local_backend == "libxc_range_separated"
    assert template.exchange_backend_id == "HYB_GGA_XC_LC_WPBE"
    assert template.default_sr_hf_fraction == 0.0
    assert template.default_lr_hf_fraction == 1.0
    assert template.default_omega == 0.4


def test_hse06_preset_maps_to_screened_rsh_template():
    preset = get_rsh_functional_preset("hyb_gga_xc_hse06")
    template = make_rsh_template("hse06")

    assert preset.name == "hse06"
    assert preset.default_sr_hf_fraction == 0.25
    assert preset.default_lr_hf_fraction == 0.0
    assert template.monotonic_lr_hf is False
    assert template.default_sr_hf_fraction == 0.25
    assert template.default_lr_hf_fraction == 0.0
    assert template.default_omega == 0.11
    assert len(preset.local_term_specs) == 3


def test_lc_wpbe_preset_exposes_optxc_tuning_omega_range():
    lc = get_rsh_functional_preset("LC_WPBE")
    template = make_rsh_template("lc-wpbe", omega_source="optxc")

    assert lc.optxc_default_omega == 0.205
    assert lc.optxc_omega_bounds == (0.13, 0.30)
    assert template.default_omega == 0.205
    assert template.omega_bounds == (0.13, 0.30)
