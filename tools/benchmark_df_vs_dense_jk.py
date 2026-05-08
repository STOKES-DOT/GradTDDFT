from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from pyscf import gto

from td_graddft.data import basis_from_spec
from td_graddft.data.integrals import eri_tensor
from td_graddft.df import build_jk_from_df, eri_to_df_factors


_SYSTEMS: dict[str, list[tuple[str, tuple[float, float, float]]]] = {
    "ethylene": [
        ("C", (-0.6695, 0.0, 0.0)),
        ("C", (0.6695, 0.0, 0.0)),
        ("H", (-1.2321, 0.9239, 0.0)),
        ("H", (-1.2321, -0.9239, 0.0)),
        ("H", (1.2321, 0.9239, 0.0)),
        ("H", (1.2321, -0.9239, 0.0)),
    ],
}


@dataclass(frozen=True)
class DFBenchmarkResult:
    system: str
    basis: str
    df_tol: float
    df_max_rank: int | None
    factor_rank: int
    dense_jk_s: float
    df_factor_s: float
    df_jk_first_s: float
    df_jk_warm_s: float
    df_jk_speedup_vs_dense: float
    j_max_abs_diff: float
    k_max_abs_diff: float


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=sorted(_SYSTEMS), default="ethylene")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--df-tol", type=float, default=1e-10)
    p.add_argument("--df-max-rank", type=int, default=0)
    p.add_argument("--outdir", default="outputs/df_benchmark")
    p.add_argument("--max-l", type=int, default=3)
    return p.parse_args()


def _timed(fn):
    t0 = time.perf_counter()
    out = fn()
    if hasattr(out, "block_until_ready"):
        out.block_until_ready()
    return out, float(time.perf_counter() - t0)


def _build_dense_jk(eri: jnp.ndarray, density: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    j_mat = jnp.einsum("pqrs,rs->pq", eri, density, precision=jax.lax.Precision.HIGHEST)
    k_mat = jnp.einsum("prqs,rs->pq", eri, density, precision=jax.lax.Precision.HIGHEST)
    return j_mat, k_mat


def main() -> None:
    args = _parse_args()
    jax.config.update("jax_enable_x64", True)
    atom = _SYSTEMS[str(args.system)]
    _ = gto.M(
        atom=atom,
        basis=str(args.basis),
        unit="Angstrom",
        charge=0,
        spin=0,
        cart=True,
        verbose=0,
    )
    basis = basis_from_spec(
        atom,
        basis=str(args.basis),
        unit="Angstrom",
        charge=0,
        spin=0,
        max_l=int(args.max_l),
    )

    with jax.default_device(jax.devices("cpu")[0]):
        eri, _ = _timed(lambda: eri_tensor(basis, engine="jit"))
        rng = np.random.default_rng(0)
        density_np = rng.normal(size=(basis.nao, basis.nao))
        density_np = 0.5 * (density_np + density_np.T)
        density = jnp.asarray(density_np)

        (_, _), dense_jk_s = _timed(lambda: _build_dense_jk(eri, density))
        df_factors, df_factor_s = _timed(
            lambda: eri_to_df_factors(
                eri,
                tol=float(args.df_tol),
                max_rank=None if int(args.df_max_rank) <= 0 else int(args.df_max_rank),
            )
        )
        (j_ref, k_ref), _ = _timed(lambda: _build_dense_jk(eri, density))
        (j_first, k_first), df_jk_first_s = _timed(lambda: build_jk_from_df(df_factors, density))
        (j_warm, k_warm), df_jk_warm_s = _timed(lambda: build_jk_from_df(df_factors, density))

    result = DFBenchmarkResult(
        system=str(args.system),
        basis=str(args.basis),
        df_tol=float(args.df_tol),
        df_max_rank=None if int(args.df_max_rank) <= 0 else int(args.df_max_rank),
        factor_rank=int(np.asarray(df_factors).shape[0]),
        dense_jk_s=float(dense_jk_s),
        df_factor_s=float(df_factor_s),
        df_jk_first_s=float(df_jk_first_s),
        df_jk_warm_s=float(df_jk_warm_s),
        df_jk_speedup_vs_dense=float(dense_jk_s / df_jk_warm_s),
        j_max_abs_diff=float(np.max(np.abs(np.asarray(j_warm) - np.asarray(j_ref)))),
        k_max_abs_diff=float(np.max(np.abs(np.asarray(k_warm) - np.asarray(k_ref)))),
    )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    json_path = outdir / f"{args.system}_{args.basis}_df_summary.json"
    with json_path.open("w") as f:
        json.dump(asdict(result), f, indent=2)
    print(json.dumps(asdict(result), indent=2))
    print(json_path)


if __name__ == "__main__":
    main()
