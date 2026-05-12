import jax
import jax.numpy as jnp
import numpy as np
import pytest

from td_graddft.data.integrals.libcint.autodiff import (
    bind_libcint_integral_constant,
    libcint_int1e_with_coords,
)


def test_bind_libcint_integral_constant_preserves_value():
    anchor = jnp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    value = jnp.asarray([[1.0, 0.2], [0.2, 1.0]])
    out = bind_libcint_integral_constant(
        value,
        geometry_anchor=anchor,
        integral_name="int1e_ovlp",
    )
    assert np.allclose(np.asarray(out), np.asarray(value), atol=0.0, rtol=0.0)


def test_bind_libcint_integral_constant_geometry_grad_raises():
    value = jnp.asarray([[1.0, 0.2], [0.2, 1.0]])

    def objective(anchor):
        out = bind_libcint_integral_constant(
            value,
            geometry_anchor=anchor,
            integral_name="int1e_ovlp",
        )
        return jnp.sum(out)

    with pytest.raises(
        NotImplementedError,
        match="Gradient through libcint integrals",
    ):
        _ = jax.grad(objective)(jnp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]))


def test_bind_libcint_integral_constant_geometry_grad_zero_policy():
    value = jnp.asarray([[1.0, 0.2], [0.2, 1.0]])

    def objective(anchor):
        out = bind_libcint_integral_constant(
            value,
            geometry_anchor=anchor,
            integral_name="int1e_ovlp",
            geometry_grad_policy="zero",
        )
        return jnp.sum(out)

    grad = jax.grad(objective)(jnp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]]))
    assert np.allclose(np.asarray(grad), 0.0, atol=0.0, rtol=0.0)


def test_bind_libcint_integral_constant_invalid_policy_raises():
    with pytest.raises(ValueError, match="Unsupported geometry_grad_policy"):
        _ = bind_libcint_integral_constant(
            jnp.asarray([[1.0]]),
            geometry_anchor=jnp.asarray([[0.0, 0.0, 0.0]]),
            integral_name="int1e_ovlp",
            geometry_grad_policy="invalid_policy",  # type: ignore[arg-type]
        )


def test_libcint_int1e_r_coordinate_gradient_matches_finite_difference():
    try:
        from pyscf import gto
    except ModuleNotFoundError:
        pytest.skip("PySCF is required for libcint coordinate-gradient tests.")

    symbols = ("O", "H", "H")
    basis = "6-31g*"
    coords0 = jnp.asarray(
        [
            [0.0, 0.0, -0.12],
            [0.0, 1.43, 1.02],
            [0.0, -1.37, 1.08],
        ],
        dtype=jnp.float64,
    )
    ref_mol = gto.M(
        atom=[
            (sym, tuple(float(x) for x in coords0[ia]))
            for ia, sym in enumerate(symbols)
        ],
        basis=basis,
        unit="Bohr",
        charge=0,
        spin=0,
        cart=True,
        verbose=0,
    )
    nao = ref_mol.nao_nr()
    weight = jnp.sin(jnp.arange(3 * nao * nao, dtype=jnp.float64)).reshape(
        3,
        nao,
        nao,
    )

    def objective(coords_bohr):
        dip = libcint_int1e_with_coords(
            coords_bohr,
            symbols,
            basis,
            0,
            0,
            True,
            0,
            "int1e_r",
            3,
            "analytic",
        )
        return jnp.sum(dip * weight)

    grad = jax.grad(objective)(coords0)

    def finite_difference_value(coords_np):
        mol = gto.M(
            atom=[
                (sym, tuple(float(x) for x in coords_np[ia]))
                for ia, sym in enumerate(symbols)
            ],
            basis=basis,
            unit="Bohr",
            charge=0,
            spin=0,
            cart=True,
            verbose=0,
        )
        dip = np.asarray(mol.intor_symmetric("int1e_r", comp=3), dtype=float)
        return float(np.sum(dip * np.asarray(weight)))

    step = 1e-5
    coords_np = np.asarray(coords0)
    fd_grad = np.zeros_like(coords_np)
    for ia in range(coords_np.shape[0]):
        for xyz in range(3):
            plus = coords_np.copy()
            minus = coords_np.copy()
            plus[ia, xyz] += step
            minus[ia, xyz] -= step
            fd_grad[ia, xyz] = (
                finite_difference_value(plus) - finite_difference_value(minus)
            ) / (2.0 * step)

    assert np.allclose(np.asarray(grad), fd_grad, atol=2e-6, rtol=2e-6)
