from __future__ import annotations

from dataclasses import dataclass
from jaxtyping import Array

from ..jax_libxc import (
    RestrictedFeatureBundle,
    eval_xc_energy_density,
    eval_xc_response_tensor,
    hybrid_coeff,
    parse_xc,
    semilocal_terms,
    xc_type,
)


CLASSIC_XC_SPECS = {
    "lda": "lda",
    "svwn": "svwn",
    "pbe": "pbe",
    "pbe0": "pbe0",
    "b3lyp": "b3lyp",
}


@dataclass(frozen=True)
class TraditionalXCFunctional:
    """Object wrapper for classic JAX XC specifications backed by jax_libxc."""

    spec: str
    name: str | None = None

    def __post_init__(self) -> None:
        parse_xc(self.spec)
        object.__setattr__(self, "name", self.spec if self.name is None else str(self.name))

    @property
    def exact_exchange_fraction(self) -> float:
        return float(hybrid_coeff(self.spec))

    @property
    def response_kind(self) -> str:
        return str(xc_type(self.spec))

    def terms(self):
        return tuple(parse_xc(self.spec))

    def semilocal_component_terms(self):
        return tuple(semilocal_terms(self.spec))

    def energy_density(self, features: RestrictedFeatureBundle) -> Array:
        return eval_xc_energy_density(self.spec, features)

    def local_energy_density(
        self,
        features: RestrictedFeatureBundle,
        *,
        density_floor: float = 1e-12,
    ) -> Array:
        del density_floor
        return self.energy_density(features)

    def response_tensor(
        self,
        rho: Array,
        *,
        grad: Array | None = None,
        tau: Array | None = None,
        density_floor: float = 1e-12,
    ) -> tuple[str, Array]:
        return eval_xc_response_tensor(
            self.spec,
            rho,
            grad=grad,
            tau=tau,
            density_floor=density_floor,
        )


def make_classic_xc_functional(spec: str, *, name: str | None = None) -> TraditionalXCFunctional:
    return TraditionalXCFunctional(spec=CLASSIC_XC_SPECS.get(spec, spec), name=name)


def make_lda_functional() -> TraditionalXCFunctional:
    return make_classic_xc_functional("lda")


def make_pbe_functional() -> TraditionalXCFunctional:
    return make_classic_xc_functional("pbe")


def make_pbe0_functional() -> TraditionalXCFunctional:
    return make_classic_xc_functional("pbe0")


def make_b3lyp_functional() -> TraditionalXCFunctional:
    return make_classic_xc_functional("b3lyp")
