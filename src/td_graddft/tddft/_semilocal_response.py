from __future__ import annotations

from dataclasses import dataclass
import weakref

from td_graddft.features import restricted_grid_response_variables
from td_graddft.jax_libxc import eval_xc_response_tensor, hybrid_coeff, xc_type


_GRID_RESPONSE_TENSOR_CACHE: dict[
    tuple[int, str],
    tuple[weakref.ReferenceType[object], object],
] = {}


@dataclass(frozen=True)
class SemilocalResponseFunctional:
    xc_spec: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "xc_spec", str(self.xc_spec).lower())
        object.__setattr__(self, "exact_exchange_fraction", float(hybrid_coeff(self.xc_spec)))
        object.__setattr__(self, "response_feature_kind", str(xc_type(self.xc_spec)))

    def grid_response_tensor(self, molecule):
        cache_key = (id(molecule), self.xc_spec)
        cached = _GRID_RESPONSE_TENSOR_CACHE.get(cache_key)
        if cached is not None:
            cached_ref, cached_tensor = cached
            if cached_ref() is molecule:
                return cached_tensor
        rho, grad_rho, tau, _ = restricted_grid_response_variables(
            molecule,
            feature_kind=self.response_feature_kind,
        )
        _, tensor = eval_xc_response_tensor(
            self.xc_spec,
            rho,
            grad=grad_rho,
            tau=tau,
        )
        _GRID_RESPONSE_TENSOR_CACHE[cache_key] = (weakref.ref(molecule), tensor)
        return tensor
