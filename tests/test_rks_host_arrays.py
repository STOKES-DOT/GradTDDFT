from td_graddft.scf import rks


def test_rks_no_longer_keeps_cuda_host_device_helpers():
    assert not hasattr(rks, "_host_device_zeros")
    assert not hasattr(rks, "_host_device_scalar")
    assert not hasattr(rks, "_host_device_zeros_like")
