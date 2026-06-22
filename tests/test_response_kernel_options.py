import pytest

from td_graddft.tddft.response_options import (
    ResponseKernelOptions,
    normalize_response_kernel_options,
)


def test_response_kernel_options_default_to_current_auto_backend():
    options = normalize_response_kernel_options(None)

    assert options == ResponseKernelOptions()
    assert options.two_electron_mode == "auto"
    assert options.ris_theta == pytest.approx(0.2)
    assert options.ris_j_fit == "sp"
    assert options.ris_k_fit == "s"
    assert options.ris_aux_chunk_size == 256


def test_response_kernel_options_accept_mapping_and_reject_unknown_modes():
    options = normalize_response_kernel_options(
        {
            "two_electron_mode": "ris",
            "ris_theta": 0.25,
            "ris_j_fit": "spd",
            "ris_k_fit": "sp",
            "ris_aux_chunk_size": 128,
        }
    )

    assert options == ResponseKernelOptions(
        two_electron_mode="ris",
        ris_theta=0.25,
        ris_j_fit="spd",
        ris_k_fit="sp",
        ris_aux_chunk_size=128,
    )
    with pytest.raises(ValueError, match="two_electron_mode"):
        normalize_response_kernel_options({"two_electron_mode": "ris_frozen_mo"})


def test_response_kernel_options_validate_ris_parameters():
    with pytest.raises(ValueError, match="ris_theta"):
        normalize_response_kernel_options({"ris_theta": 0.0})
    with pytest.raises(ValueError, match="ris_j_fit"):
        normalize_response_kernel_options({"ris_j_fit": "spdf"})
    with pytest.raises(ValueError, match="ris_aux_chunk_size"):
        normalize_response_kernel_options({"ris_aux_chunk_size": 0})
