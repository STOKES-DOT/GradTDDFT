import numpy as np
import pytest

from td_graddft.nn_rsh import (
    get_rsh_functional_preset,
    list_rsh_functional_presets,
    make_rsh_template,
)


def test_lc_wpbe_preset_matches_pyscf_libxc_coefficients():
    pytest.importorskip("pyscf")
    from pyscf import dft, gto

    preset = get_rsh_functional_preset("lc-wpbe")
    params = preset.default_params

    assert preset.pyscf_xc_name == "LC_WPBE"
    assert preset.strict_jax_supported is True
    assert preset.jax_local_xc_spec == "lc_wpbe_local"
    assert np.allclose(
        tuple(float(x) for x in params.to_pyscf_rsh()),
        dft.libxc.rsh_coeff("LC_WPBE"),
    )
    assert np.allclose(
        (float(params.omega), float(params.lr_hf_fraction), float(params.sr_hf_fraction)),
        dft.RKS(gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0))._numint.rsh_and_hybrid_coeff(
            "LC_WPBE", spin=0
        ),
    )


def test_wb97xd_preset_matches_pyscf_libxc_coefficients():
    pytest.importorskip("pyscf")
    from pyscf import dft, gto

    preset = get_rsh_functional_preset("wb97x-d")
    params = preset.default_params
    mol = gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0)

    assert preset.pyscf_xc_name == "WB97X_D"
    assert preset.has_dispersion is True
    assert np.allclose(
        tuple(float(x) for x in params.to_pyscf_rsh()),
        dft.libxc.rsh_coeff("WB97X_D"),
    )
    assert np.allclose(
        (float(params.omega), float(params.lr_hf_fraction), float(params.sr_hf_fraction)),
        dft.RKS(mol)._numint.rsh_and_hybrid_coeff("WB97X_D", spin=0),
    )


def test_rsh_preset_aliases_and_templates_are_canonical():
    names = list_rsh_functional_presets()
    assert "lc-wpbe" in names
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


def test_lc_wpbe_preset_exposes_optxc_tuning_omega_range():
    lc = get_rsh_functional_preset("LC_WPBE")
    template = make_rsh_template("lc-wpbe", omega_source="optxc")

    assert lc.optxc_default_omega == 0.205
    assert lc.optxc_omega_bounds == (0.13, 0.30)
    assert template.default_omega == 0.205
    assert template.omega_bounds == (0.13, 0.30)
