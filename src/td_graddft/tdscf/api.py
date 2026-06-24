from __future__ import annotations

from typing import Any, Literal

from .. import spectra
from ..spectra import HARTREE_TO_EV
from ..tddft.eigensolvers import PYSCF_TD_DAVIDSON_MAX_CYCLE
from ..tddft.eigensolvers import PYSCF_TD_DAVIDSON_TOL
from ..tddft.eigensolvers import PYSCF_TD_POSITIVE_EIG_THRESHOLD
from ..tddft import RestrictedCasidaTDDFT, UnrestrictedCasidaTDDFT, UnrestrictedTDA
from ..tddft._semilocal_response import SemilocalResponseFunctional


def _molecule_from_source(source: Any) -> Any:
    if hasattr(source, "molecule"):
        molecule = getattr(source, "molecule")
        if molecule is None:
            ensure_molecule = getattr(source, "_ensure_molecule", None)
            if callable(ensure_molecule):
                return ensure_molecule()
    if hasattr(source, "reference"):
        molecule = getattr(source, "reference")
        if molecule is None:
            ensure_molecule = getattr(source, "_ensure_molecule", None)
            if callable(ensure_molecule):
                return ensure_molecule()
            ensure_reference = getattr(source, "_ensure_reference", None)
            if callable(ensure_reference):
                return ensure_reference()
            raise RuntimeError(
                "Run ground-state mf.kernel() or mf.run() before launching TD-SCF."
            )
        return molecule
    return source


_reference_from_source = _molecule_from_source


def _is_unrestricted_molecule(molecule: Any) -> bool:
    return hasattr(molecule, "nocc_alpha") or hasattr(molecule, "nocc_beta")


_is_unrestricted_reference = _is_unrestricted_molecule


def _xy_from_result(result: Any) -> Any:
    if hasattr(result, "amplitudes"):
        return result.amplitudes
    if hasattr(result, "x_amplitudes") and hasattr(result, "y_amplitudes"):
        return (result.x_amplitudes, result.y_amplitudes)
    if hasattr(result, "amplitudes_alpha") and hasattr(result, "amplitudes_beta"):
        return (result.amplitudes_alpha, result.amplitudes_beta)
    if (
        hasattr(result, "x_amplitudes_alpha")
        and hasattr(result, "x_amplitudes_beta")
        and hasattr(result, "y_amplitudes_alpha")
        and hasattr(result, "y_amplitudes_beta")
    ):
        return (
            (result.x_amplitudes_alpha, result.x_amplitudes_beta),
            (result.y_amplitudes_alpha, result.y_amplitudes_beta),
        )
    return None


class _BaseTD:
    """PySCF-style base object for TD-GradDFT excited-state facades."""

    nstates: int | None
    occupation_tolerance: float
    excitation_threshold: float
    matrix_eps: float
    eigensolver: Literal["auto", "davidson"]
    davidson_tol: float
    davidson_max_iter: int
    davidson_max_subspace: int | None
    davidson_initial_guess_count: int | None
    davidson_max_trial_vectors: int | None

    def __init__(
        self,
        mf_or_molecule: Any = None,
        *,
        xc_functional: Any | None = None,
        xc_params: Any | None = None,
        nstates: int | None = 3,
        occupation_tolerance: float = 1e-8,
        excitation_threshold: float = PYSCF_TD_POSITIVE_EIG_THRESHOLD,
        matrix_eps: float = 1e-10,
        eigensolver: Literal["auto", "davidson"] = "auto",
        davidson_tol: float = PYSCF_TD_DAVIDSON_TOL,
        davidson_max_iter: int = PYSCF_TD_DAVIDSON_MAX_CYCLE,
        davidson_max_subspace: int | None = None,
        davidson_initial_guess_count: int | None = None,
        davidson_max_trial_vectors: int | None = None,
        response_kernel_options: Any | None = None,
        **kwargs: Any,
    ) -> None:
        if mf_or_molecule is None and "mf_or_reference" in kwargs:
            mf_or_molecule = kwargs.pop("mf_or_reference")
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword argument(s): {unexpected}")
        self.mf = mf_or_molecule
        self.xc_functional = xc_functional
        self.xc_params = xc_params
        self.nstates = nstates
        self.occupation_tolerance = occupation_tolerance
        self.excitation_threshold = excitation_threshold
        self.matrix_eps = matrix_eps
        self.eigensolver = eigensolver
        self.davidson_tol = davidson_tol
        self.davidson_max_iter = davidson_max_iter
        self.davidson_max_subspace = davidson_max_subspace
        self.davidson_initial_guess_count = davidson_initial_guess_count
        self.davidson_max_trial_vectors = davidson_max_trial_vectors
        self.response_kernel_options = response_kernel_options

        self.result: Any | None = None
        self.e: Any | None = None
        self.e_ev: Any | None = None
        self.xy: Any | None = None
        self.converged: bool | None = None
        self._solver: Any | None = None

    @property
    def molecule(self) -> Any:
        return _molecule_from_source(self.mf)

    @property
    def reference(self) -> Any:
        return self.molecule

    def _resolved_xc_functional(self) -> Any | None:
        if self.xc_functional is not None:
            if isinstance(self.xc_functional, str):
                return SemilocalResponseFunctional(self.xc_functional)
            return self.xc_functional
        source_xc = getattr(self.mf, "xc", None)
        if source_xc is None:
            return None
        if isinstance(source_xc, str):
            return SemilocalResponseFunctional(source_xc)
        return source_xc

    def _common_solver_kwargs(self, molecule: Any) -> dict[str, Any]:
        return {
            "molecule": molecule,
            "xc_functional": self._resolved_xc_functional(),
            "xc_params": self.xc_params,
            "occupation_tolerance": self.occupation_tolerance,
            "excitation_threshold": self.excitation_threshold,
        }

    def _restricted_solver_kwargs(self, molecule: Any) -> dict[str, Any]:
        kwargs = self._common_solver_kwargs(molecule)
        kwargs.update(
            {
                "matrix_eps": self.matrix_eps,
                "eigensolver": self.eigensolver,
                "davidson_tol": self.davidson_tol,
                "davidson_max_iter": self.davidson_max_iter,
                "davidson_max_subspace": self.davidson_max_subspace,
                "davidson_initial_guess_count": self.davidson_initial_guess_count,
                "davidson_max_trial_vectors": self.davidson_max_trial_vectors,
                "response_kernel_options": self.response_kernel_options,
            }
        )
        return kwargs

    def _unrestricted_solver_kwargs(self, molecule: Any) -> dict[str, Any]:
        kwargs = self._common_solver_kwargs(molecule)
        kwargs.update(
            {
                "matrix_eps": self.matrix_eps,
                "eigensolver": self.eigensolver,
                "davidson_tol": self.davidson_tol,
                "davidson_max_iter": self.davidson_max_iter,
                "davidson_max_subspace": self.davidson_max_subspace,
            }
        )
        return kwargs

    def _set_result(self, result: Any) -> Any:
        self.result = result
        self.e = result.excitation_energies
        self.e_ev = self.e * HARTREE_TO_EV
        self.xy = _xy_from_result(result)
        converged = getattr(result, "converged", True)
        try:
            self.converged = bool(converged)
        except (TypeError, ValueError):
            self.converged = converged
        return result

    def _require_result(self) -> Any:
        if self.result is None:
            raise RuntimeError("Run td.kernel() before requesting excited-state properties.")
        return self.result

    def transition_dipole(self) -> Any:
        return spectra.transition_dipoles(
            self.molecule,
            self._require_result(),
            occupation_tolerance=self.occupation_tolerance,
        )

    def transition_dipoles(self) -> Any:
        return self.transition_dipole()

    def oscillator_strength(self) -> Any:
        return spectra.oscillator_strengths(
            self.molecule,
            self._require_result(),
            occupation_tolerance=self.occupation_tolerance,
        )


class TDA(_BaseTD):
    """PySCF-style TDA driver dispatching restricted and unrestricted molecules."""

    def _build_solver(self) -> Any:
        molecule = self.molecule
        if _is_unrestricted_molecule(molecule):
            kwargs = self._unrestricted_solver_kwargs(molecule)
            kwargs.pop("matrix_eps")
            return UnrestrictedTDA(**kwargs)
        return RestrictedCasidaTDDFT(**self._restricted_solver_kwargs(molecule))

    def kernel(self, nstates: int | None = None) -> Any:
        nroots = self.nstates if nstates is None else nstates
        self._solver = self._build_solver()
        try:
            if isinstance(self._solver, UnrestrictedTDA):
                result = self._solver.kernel(nstates=nroots)
            else:
                result = self._solver.tda(nstates=nroots)
        except Exception:
            self.converged = False
            raise
        return self._set_result(result)


class TDDFT(_BaseTD):
    """PySCF-style full Casida TDDFT driver."""

    def _build_solver(self) -> Any:
        molecule = self.molecule
        if _is_unrestricted_molecule(molecule):
            return UnrestrictedCasidaTDDFT(**self._unrestricted_solver_kwargs(molecule))
        return RestrictedCasidaTDDFT(**self._restricted_solver_kwargs(molecule))

    def kernel(self, nstates: int | None = None) -> Any:
        nroots = self.nstates if nstates is None else nstates
        self._solver = self._build_solver()
        try:
            result = self._solver.kernel(nstates=nroots)
        except Exception:
            self.converged = False
            raise
        return self._set_result(result)
