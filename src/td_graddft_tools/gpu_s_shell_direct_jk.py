from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from jaxtyping import Array

from td_graddft.data.basis import CartesianBasis


@dataclass(frozen=True)
class SShellSystem:
    """Minimal all-s shell data layout for the CUDA direct-J/K prototype."""

    centers: np.ndarray
    exponents: np.ndarray
    coefficients: np.ndarray
    nprims: np.ndarray

    @property
    def nshell(self) -> int:
        return int(self.centers.shape[0])

    @property
    def max_nprim(self) -> int:
        return int(self.exponents.shape[1])


@dataclass(frozen=True)
class SPAOSystem:
    """Minimal AO-level Cartesian data layout for the CUDA direct-J/K prototype."""

    centers: np.ndarray
    angulars: np.ndarray
    exponents: np.ndarray
    coefficients: np.ndarray
    nprims: np.ndarray

    @property
    def nao(self) -> int:
        return int(self.centers.shape[0])

    @property
    def max_nprim(self) -> int:
        return int(self.exponents.shape[1])


def extract_s_shell_system(basis: CartesianBasis) -> SShellSystem:
    """Extract a dense all-s-shell representation from a Cartesian basis."""

    shells = tuple(basis.shells)
    if not shells:
        return SShellSystem(
            centers=np.zeros((0, 3), dtype=np.float64),
            exponents=np.zeros((0, 0), dtype=np.float64),
            coefficients=np.zeros((0, 0), dtype=np.float64),
            nprims=np.zeros((0,), dtype=np.int32),
        )
    for shell in shells:
        if len(shell.angulars) != 1 or shell.angulars[0] != (0, 0, 0):
            raise NotImplementedError("The GPU prototype currently supports s-shell systems only.")

    nprims = np.asarray([int(shell.exponents.shape[0]) for shell in shells], dtype=np.int32)
    max_nprim = int(np.max(nprims))
    centers = np.asarray([np.asarray(shell.center, dtype=np.float64) for shell in shells], dtype=np.float64)
    exponents = np.zeros((len(shells), max_nprim), dtype=np.float64)
    coefficients = np.zeros((len(shells), max_nprim), dtype=np.float64)
    for idx, shell in enumerate(shells):
        nprim = int(nprims[idx])
        exponents[idx, :nprim] = np.asarray(shell.exponents, dtype=np.float64)
        coefficients[idx, :nprim] = np.asarray(shell.coefficients, dtype=np.float64)
    return SShellSystem(
        centers=centers,
        exponents=exponents,
        coefficients=coefficients,
        nprims=nprims,
    )


def extract_sp_ao_system(basis: CartesianBasis) -> SPAOSystem:
    """Extract a dense AO-level representation for s/p Cartesian AOs."""

    return extract_cartesian_ao_system(basis, max_l=1)


def extract_cartesian_ao_system(basis: CartesianBasis, *, max_l: int = 2) -> SPAOSystem:
    """Extract a dense AO-level representation up to a Cartesian angular limit."""

    aos = tuple(basis.aos)
    if not aos:
        return SPAOSystem(
            centers=np.zeros((0, 3), dtype=np.float64),
            angulars=np.zeros((0, 3), dtype=np.int32),
            exponents=np.zeros((0, 0), dtype=np.float64),
            coefficients=np.zeros((0, 0), dtype=np.float64),
            nprims=np.zeros((0,), dtype=np.int32),
        )
    for ao in aos:
        if sum(int(power) for power in ao.angular) > int(max_l):
            raise NotImplementedError(
                f"The GPU prototype currently supports Cartesian AOs up to l={int(max_l)}."
            )

    nprims = np.asarray([int(ao.exponents.shape[0]) for ao in aos], dtype=np.int32)
    max_nprim = int(np.max(nprims))
    centers = np.asarray([np.asarray(ao.center, dtype=np.float64) for ao in aos], dtype=np.float64)
    angulars = np.asarray([ao.angular for ao in aos], dtype=np.int32)
    exponents = np.zeros((len(aos), max_nprim), dtype=np.float64)
    coefficients = np.zeros((len(aos), max_nprim), dtype=np.float64)
    for idx, ao in enumerate(aos):
        nprim = int(nprims[idx])
        exponents[idx, :nprim] = np.asarray(ao.exponents, dtype=np.float64)
        coefficients[idx, :nprim] = np.asarray(ao.coefficients, dtype=np.float64)
    return SPAOSystem(
        centers=centers,
        angulars=angulars,
        exponents=exponents,
        coefficients=coefficients,
        nprims=nprims,
    )


def _primitive_s_norm(alpha: float) -> float:
    return float((2.0 * alpha / math.pi) ** 0.75)


def _primitive_sp_norm(alpha: float, angular: tuple[int, int, int]) -> float:
    ltot = sum(int(power) for power in angular)
    if ltot == 0:
        return _primitive_s_norm(alpha)
    if ltot == 1:
        return _primitive_s_norm(alpha) * math.sqrt(4.0 * alpha)

    numerator = (2.0 ** (2 * ltot + 3)) * float(math.factorial(ltot + 1))
    denominator = float(math.factorial(2 * ltot + 2)) * math.sqrt(math.pi)
    return math.sqrt(numerator * (2.0 * alpha) ** (ltot + 1.5) / denominator)


def _boys0(t: float) -> float:
    if t < 1e-8:
        term = 1.0
        total = 0.0
        factorial = 1.0
        for k in range(11):
            if k > 0:
                factorial *= k
                term *= -t
            total += term / (factorial * (2 * k + 1))
        return total
    sqrt_t = math.sqrt(t)
    return 0.5 * math.sqrt(math.pi) * math.erf(sqrt_t) / sqrt_t


def _boys_values(max_n: int, t: float) -> tuple[float, ...]:
    if t < 1e-8:
        values = []
        for n in range(max_n + 1):
            term = 1.0
            total = 0.0
            factorial = 1.0
            for k in range(16):
                if k > 0:
                    factorial *= k
                    term *= -t
                total += term / (factorial * (2 * n + 2 * k + 1))
            values.append(total)
        return tuple(values)

    values = [_boys0(t)]
    exp_neg_t = math.exp(-t)
    for n in range(1, max_n + 1):
        values.append(((2 * n - 1) * values[-1] - exp_neg_t) / (2.0 * t))
    return tuple(values)


def _primitive_ssss(
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    center_a: np.ndarray,
    center_b: np.ndarray,
    center_c: np.ndarray,
    center_d: np.ndarray,
) -> float:
    p = alpha + beta
    q = gamma + delta
    mu = alpha * beta / p
    nu = gamma * delta / q
    rab2 = float(np.dot(center_a - center_b, center_a - center_b))
    rcd2 = float(np.dot(center_c - center_d, center_c - center_d))
    center_p = (alpha * center_a + beta * center_b) / p
    center_q = (gamma * center_c + delta * center_d) / q
    rpq2 = float(np.dot(center_p - center_q, center_p - center_q))
    prefactor = 2.0 * math.pi**2.5 / (p * q * math.sqrt(p + q))
    t = (p * q / (p + q)) * rpq2
    return prefactor * math.exp(-mu * rab2 - nu * rcd2) * _boys0(t)


def _tuple_dec(angular: tuple[int, int, int], axis: int) -> tuple[int, int, int]:
    values = list(angular)
    values[axis] -= 1
    return tuple(values)


def _tuple_inc(angular: tuple[int, int, int], axis: int) -> tuple[int, int, int]:
    values = list(angular)
    values[axis] += 1
    return tuple(values)


def _primitive_sp_eri(
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    center_a: np.ndarray,
    center_b: np.ndarray,
    center_c: np.ndarray,
    center_d: np.ndarray,
    angular_a: tuple[int, int, int],
    angular_b: tuple[int, int, int],
    angular_c: tuple[int, int, int],
    angular_d: tuple[int, int, int],
) -> float:
    p = alpha + beta
    q = gamma + delta
    mu = alpha * beta / p
    nu = gamma * delta / q
    rab2 = float(np.dot(center_a - center_b, center_a - center_b))
    rcd2 = float(np.dot(center_c - center_d, center_c - center_d))
    center_p = (alpha * center_a + beta * center_b) / p
    center_q = (gamma * center_c + delta * center_d) / q
    center_w = (p * center_p + q * center_q) / (p + q)
    rpq2 = float(np.dot(center_p - center_q, center_p - center_q))
    prefactor = 2.0 * math.pi**2.5 / (p * q * math.sqrt(p + q))
    prefactor *= math.exp(-mu * rab2 - nu * rcd2)
    total_l = sum(angular_a) + sum(angular_b) + sum(angular_c) + sum(angular_d)
    boys = tuple(prefactor * value for value in _boys_values(total_l + 2, (p * q / (p + q)) * rpq2))
    bra_ratio = q / (p + q)
    ket_ratio = p / (p + q)

    @lru_cache(maxsize=None)
    def vrr(a: tuple[int, int, int], c: tuple[int, int, int], m: int) -> float:
        if min(a + c) < 0:
            return 0.0
        if sum(a) + sum(c) == 0:
            return boys[m]

        if sum(a) > 0:
            axis = next(idx for idx, power in enumerate(a) if power > 0)
            a1 = _tuple_dec(a, axis)
            out = (center_p[axis] - center_a[axis]) * vrr(a1, c, m)
            out += (center_w[axis] - center_p[axis]) * vrr(a1, c, m + 1)
            if a1[axis] > 0:
                a2 = _tuple_dec(a1, axis)
                coef = a1[axis] / (2.0 * p)
                out += coef * (vrr(a2, c, m) - bra_ratio * vrr(a2, c, m + 1))
            if c[axis] > 0:
                c1 = _tuple_dec(c, axis)
                coef = c[axis] / (2.0 * (p + q))
                out += coef * vrr(a1, c1, m + 1)
            return out

        axis = next(idx for idx, power in enumerate(c) if power > 0)
        c1 = _tuple_dec(c, axis)
        out = (center_q[axis] - center_c[axis]) * vrr(a, c1, m)
        out += (center_w[axis] - center_q[axis]) * vrr(a, c1, m + 1)
        if c1[axis] > 0:
            c2 = _tuple_dec(c1, axis)
            coef = c1[axis] / (2.0 * q)
            out += coef * (vrr(a, c2, m) - ket_ratio * vrr(a, c2, m + 1))
        if a[axis] > 0:
            a1 = _tuple_dec(a, axis)
            coef = a[axis] / (2.0 * (p + q))
            out += coef * vrr(a1, c1, m + 1)
        return out

    @lru_cache(maxsize=None)
    def hrr(
        a: tuple[int, int, int],
        b: tuple[int, int, int],
        c: tuple[int, int, int],
        d: tuple[int, int, int],
    ) -> float:
        if min(a + b + c + d) < 0:
            return 0.0
        if sum(b) > 0:
            axis = next(idx for idx, power in enumerate(b) if power > 0)
            b1 = _tuple_dec(b, axis)
            a1 = _tuple_inc(a, axis)
            return hrr(a1, b1, c, d) + (center_a[axis] - center_b[axis]) * hrr(a, b1, c, d)
        if sum(d) > 0:
            axis = next(idx for idx, power in enumerate(d) if power > 0)
            d1 = _tuple_dec(d, axis)
            c1 = _tuple_inc(c, axis)
            return hrr(a, b, c1, d1) + (center_c[axis] - center_d[axis]) * hrr(a, b, c, d1)
        return vrr(a, c, 0)

    return hrr(angular_a, angular_b, angular_c, angular_d)


def contracted_ssss(system: SShellSystem, i: int, j: int, k: int, l: int) -> float:
    """Return contracted `(ij|kl)` for one all-s shell quartet."""

    value = 0.0
    for ip in range(int(system.nprims[i])):
        alpha = float(system.exponents[i, ip])
        weight_i = float(system.coefficients[i, ip]) * _primitive_s_norm(alpha)
        for jp in range(int(system.nprims[j])):
            beta = float(system.exponents[j, jp])
            weight_j = float(system.coefficients[j, jp]) * _primitive_s_norm(beta)
            for kp in range(int(system.nprims[k])):
                gamma = float(system.exponents[k, kp])
                weight_k = float(system.coefficients[k, kp]) * _primitive_s_norm(gamma)
                for lp in range(int(system.nprims[l])):
                    delta = float(system.exponents[l, lp])
                    weight_l = float(system.coefficients[l, lp]) * _primitive_s_norm(delta)
                    value += (
                        weight_i
                        * weight_j
                        * weight_k
                        * weight_l
                        * _primitive_ssss(
                            alpha,
                            beta,
                            gamma,
                            delta,
                            system.centers[i],
                            system.centers[j],
                            system.centers[k],
                            system.centers[l],
                        )
                    )
    return value


def contracted_sp_eri(system: SPAOSystem, i: int, j: int, k: int, l: int) -> float:
    """Return contracted `(ij|kl)` for one s/p AO quartet."""

    angular_i = tuple(int(value) for value in system.angulars[i])
    angular_j = tuple(int(value) for value in system.angulars[j])
    angular_k = tuple(int(value) for value in system.angulars[k])
    angular_l = tuple(int(value) for value in system.angulars[l])
    value = 0.0
    for ip in range(int(system.nprims[i])):
        alpha = float(system.exponents[i, ip])
        weight_i = float(system.coefficients[i, ip]) * _primitive_sp_norm(alpha, angular_i)
        for jp in range(int(system.nprims[j])):
            beta = float(system.exponents[j, jp])
            weight_j = float(system.coefficients[j, jp]) * _primitive_sp_norm(beta, angular_j)
            for kp in range(int(system.nprims[k])):
                gamma = float(system.exponents[k, kp])
                weight_k = float(system.coefficients[k, kp]) * _primitive_sp_norm(gamma, angular_k)
                for lp in range(int(system.nprims[l])):
                    delta = float(system.exponents[l, lp])
                    weight_l = float(system.coefficients[l, lp]) * _primitive_sp_norm(delta, angular_l)
                    value += (
                        weight_i
                        * weight_j
                        * weight_k
                        * weight_l
                        * _primitive_sp_eri(
                            alpha,
                            beta,
                            gamma,
                            delta,
                            system.centers[i],
                            system.centers[j],
                            system.centers[k],
                            system.centers[l],
                            angular_i,
                            angular_j,
                            angular_k,
                            angular_l,
                        )
                    )
    return value


def cpu_sp_direct_jk(system: SPAOSystem, density: Array) -> tuple[np.ndarray, np.ndarray]:
    """Build exact s/p AO-level J/K without materializing a packed ERI matrix."""

    density_arr = np.asarray(density, dtype=np.float64)
    density_arr = 0.5 * (density_arr + density_arr.T)
    nao = system.nao
    if density_arr.shape != (nao, nao):
        raise ValueError(f"Density shape {density_arr.shape} does not match AO count {(nao, nao)}.")

    j_mat = np.zeros_like(density_arr)
    k_mat = np.zeros_like(density_arr)
    for i in range(nao):
        for j in range(nao):
            for k in range(nao):
                for l in range(nao):
                    eri = contracted_sp_eri(system, i, j, k, l)
                    j_mat[i, j] += eri * density_arr[k, l]
                    k_mat[i, k] += eri * density_arr[j, l]
    return 0.5 * (j_mat + j_mat.T), 0.5 * (k_mat + k_mat.T)


def cpu_sp_direct_jk_from_basis(
    basis: CartesianBasis,
    density: Array,
) -> tuple[np.ndarray, np.ndarray]:
    """Build exact s/p AO-level J/K directly from a Cartesian basis."""

    return cpu_sp_direct_jk(extract_sp_ao_system(basis), density)


def cpu_s_shell_direct_jk(system: SShellSystem, density: Array) -> tuple[np.ndarray, np.ndarray]:
    """Build exact all-s-shell J/K without materializing a packed ERI matrix."""

    density_arr = np.asarray(density, dtype=np.float64)
    density_arr = 0.5 * (density_arr + density_arr.T)
    nshell = system.nshell
    if density_arr.shape != (nshell, nshell):
        raise ValueError(
            f"Density shape {density_arr.shape} does not match all-s shell count {(nshell, nshell)}."
        )

    j_mat = np.zeros_like(density_arr)
    k_mat = np.zeros_like(density_arr)
    for i in range(nshell):
        for j in range(nshell):
            for k in range(nshell):
                for l in range(nshell):
                    eri = contracted_ssss(system, i, j, k, l)
                    j_mat[i, j] += eri * density_arr[k, l]
                    k_mat[i, k] += eri * density_arr[j, l]
    return 0.5 * (j_mat + j_mat.T), 0.5 * (k_mat + k_mat.T)
