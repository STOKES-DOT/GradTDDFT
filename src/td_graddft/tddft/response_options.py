from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Literal


TwoElectronMode = Literal["auto", "direct", "df", "ris"]
RISFit = Literal["s", "sp", "spd"]


@dataclass(frozen=True)
class ResponseKernelOptions:
    """Options controlling the two-electron part of restricted TD response."""

    two_electron_mode: TwoElectronMode = "auto"
    ris_theta: float = 0.2
    ris_j_fit: RISFit = "sp"
    ris_k_fit: RISFit = "s"
    ris_aux_chunk_size: int = 256


_OPTION_FIELDS = {field.name for field in fields(ResponseKernelOptions)}
_TWO_ELECTRON_MODES = {"auto", "direct", "df", "ris"}
_RIS_FITS = {"s", "sp", "spd"}


def normalize_response_kernel_options(
    options: ResponseKernelOptions | dict[str, Any] | None,
) -> ResponseKernelOptions:
    if options is None:
        normalized = ResponseKernelOptions()
    elif isinstance(options, ResponseKernelOptions):
        normalized = options
    elif isinstance(options, dict):
        unknown = sorted(set(options) - _OPTION_FIELDS)
        if unknown:
            raise ValueError(f"Unknown response kernel option(s): {', '.join(unknown)}.")
        normalized = ResponseKernelOptions(**options)
    else:
        raise TypeError(
            "response_kernel_options must be None, a ResponseKernelOptions instance, "
            "or a mapping."
        )

    mode = str(normalized.two_electron_mode).lower()
    if mode not in _TWO_ELECTRON_MODES:
        raise ValueError(
            "two_electron_mode must be one of {'auto', 'direct', 'df', 'ris'}, "
            f"got {normalized.two_electron_mode!r}."
        )
    j_fit = str(normalized.ris_j_fit).lower()
    k_fit = str(normalized.ris_k_fit).lower()
    if j_fit not in _RIS_FITS:
        raise ValueError(f"ris_j_fit must be one of {{'s', 'sp', 'spd'}}, got {j_fit!r}.")
    if k_fit not in _RIS_FITS:
        raise ValueError(f"ris_k_fit must be one of {{'s', 'sp', 'spd'}}, got {k_fit!r}.")
    if float(normalized.ris_theta) <= 0.0:
        raise ValueError("ris_theta must be positive.")
    if int(normalized.ris_aux_chunk_size) <= 0:
        raise ValueError("ris_aux_chunk_size must be positive.")

    return ResponseKernelOptions(
        two_electron_mode=mode,  # type: ignore[arg-type]
        ris_theta=float(normalized.ris_theta),
        ris_j_fit=j_fit,  # type: ignore[arg-type]
        ris_k_fit=k_fit,  # type: ignore[arg-type]
        ris_aux_chunk_size=int(normalized.ris_aux_chunk_size),
    )
