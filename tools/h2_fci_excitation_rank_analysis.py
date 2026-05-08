from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from pyscf import ao2mo, fci, gto, scf
from pyscf.fci import cistring

HARTREE_TO_EV = 27.211386245988


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze FCI singlet excited-state wavefunctions of H2 by excitation rank "
            "relative to the RHF reference determinant."
        )
    )
    parser.add_argument("--bond-length", type=float, default=0.7414, help="H-H bond length in Angstrom")
    parser.add_argument("--basis", type=str, default="6-31g*", help="AO basis")
    parser.add_argument(
        "--n-singlet-excited",
        type=int,
        default=3,
        help="number of singlet excited states to report (S1..Sn)",
    )
    parser.add_argument(
        "--top-dets",
        type=int,
        default=10,
        help="top determinants by |CI coefficient| per state",
    )
    parser.add_argument(
        "--singlet-ss-tol",
        type=float,
        default=1e-6,
        help="singlet detection threshold on <S^2>",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="outputs/h2_fci_excitation_rank_analysis_631gstar_equilibrium",
        help="output directory",
    )
    return parser.parse_args()


def build_h2_mol(bond_length: float, basis: str) -> gto.Mole:
    mol = gto.Mole()
    mol.atom = f"""
    H 0.0000000000 0.0000000000 {-0.5 * bond_length:.10f}
    H 0.0000000000 0.0000000000 {+0.5 * bond_length:.10f}
    """
    mol.unit = "Angstrom"
    mol.basis = basis
    mol.spin = 0
    mol.charge = 0
    mol.verbose = 0
    mol.build()
    return mol


def bit_occ_list(bit_string: int, norb: int) -> list[int]:
    occ: list[int] = []
    x = int(bit_string)
    for orb in range(norb):
        if (x >> orb) & 1:
            occ.append(orb)
    return occ


def build_rank_labels(
    norb: int,
    nalpha: int,
    nbeta: int,
) -> tuple[np.ndarray, list[list[int]], list[list[int]]]:
    alpha_strings = np.asarray(cistring.make_strings(range(norb), nalpha), dtype=np.int64)
    beta_strings = np.asarray(cistring.make_strings(range(norb), nbeta), dtype=np.int64)

    alpha_occ = [bit_occ_list(s, norb) for s in alpha_strings]
    beta_occ = [bit_occ_list(s, norb) for s in beta_strings]

    ref_a = set(range(nalpha))
    ref_b = set(range(nbeta))

    rank_matrix = np.zeros((len(alpha_strings), len(beta_strings)), dtype=np.int32)
    for ia, occ_a in enumerate(alpha_occ):
        exa = len(ref_a - set(occ_a))
        for ib, occ_b in enumerate(beta_occ):
            exb = len(ref_b - set(occ_b))
            rank_matrix[ia, ib] = exa + exb
    return rank_matrix.reshape(-1), alpha_occ, beta_occ


def solve_fci_roots(
    h1_mo: np.ndarray,
    eri_mo: np.ndarray,
    norb: int,
    nelec: int,
    required_singlets: int,
    singlet_ss_tol: float,
) -> tuple[np.ndarray, list[np.ndarray], list[float]]:
    ndeta = cistring.num_strings(norb, nelec // 2)
    ndetb = cistring.num_strings(norb, nelec // 2)
    ndet = int(ndeta * ndetb)
    nroots = min(ndet, max(8, required_singlets * 4))

    solver = fci.direct_spin1.FCI()
    solver.conv_tol = 1e-12
    solver.max_cycle = 200

    while True:
        e_roots, ci_roots = solver.kernel(h1_mo, eri_mo, norb, nelec, nroots=nroots)
        e_arr = np.asarray(e_roots, dtype=np.float64).reshape(-1)
        ci_list = ci_roots if isinstance(ci_roots, (list, tuple)) else [ci_roots]

        ss_values = [
            float(fci.spin_op.spin_square0(ci_list[idx], norb, nelec)[0])
            for idx in range(len(ci_list))
        ]
        n_singlets = int(np.sum(np.asarray(ss_values) < singlet_ss_tol))
        if n_singlets >= required_singlets or nroots >= ndet:
            return e_arr, [np.asarray(ci, dtype=np.float64) for ci in ci_list], ss_values
        nroots = min(ndet, max(nroots + 4, int(np.ceil(1.5 * nroots))))


def build_hamiltonian_matrix(
    h1_mo: np.ndarray,
    eri_mo: np.ndarray,
    norb: int,
    nelec: int,
) -> np.ndarray:
    nalpha = nelec // 2
    nbeta = nelec // 2
    ndet = int(cistring.num_strings(norb, nalpha) * cistring.num_strings(norb, nbeta))
    addr, h_pspace = fci.direct_spin1.pspace(h1_mo, eri_mo, norb, nelec, np=ndet)
    addr = np.asarray(addr, dtype=np.int64).reshape(-1)
    if addr.size != ndet:
        raise RuntimeError(f"Unexpected pspace dimension: {addr.size} != {ndet}")
    perm = np.argsort(addr)
    if not np.all(addr[perm] == np.arange(ndet, dtype=np.int64)):
        raise RuntimeError("pspace addresses do not cover full determinant space.")
    return np.asarray(h_pspace, dtype=np.float64)[np.ix_(perm, perm)]


def state_rank_summary(
    *,
    ci_vec: np.ndarray,
    hmat: np.ndarray,
    ranks: np.ndarray,
) -> list[dict[str, float | int]]:
    c = np.asarray(ci_vec, dtype=np.float64).reshape(-1)
    hc = hmat @ c
    stats: list[dict[str, float | int]] = []
    for rank in sorted(int(v) for v in np.unique(ranks)):
        mask = ranks == rank
        c_rank = np.where(mask, c, 0.0)
        weight = float(np.sum(c[mask] ** 2))
        if weight < 1e-16:
            continue
        e_proj = float(c_rank @ (hmat @ c_rank))
        e_total = float(c_rank @ hc)
        e_coupling = e_total - e_proj
        abs_c = np.abs(c[mask])
        stats.append(
            {
                "rank": rank,
                "n_determinants": int(np.count_nonzero(mask)),
                "weight": weight,
                "weight_percent": 100.0 * weight,
                "coef_l2": float(np.sqrt(weight)),
                "coef_abs_max": float(abs_c.max()),
                "coef_abs_mean": float(abs_c.mean()),
                "energy_projected_h": e_proj,
                "energy_coupling_h": e_coupling,
                "energy_total_h": e_total,
                "energy_total_ev": e_total * HARTREE_TO_EV,
            }
        )
    return stats


def top_determinants(
    *,
    ci_vec: np.ndarray,
    ranks: np.ndarray,
    alpha_occ: list[list[int]],
    beta_occ: list[list[int]],
    ndetb: int,
    top_n: int,
) -> list[dict[str, float | int | list[int]]]:
    c = np.asarray(ci_vec, dtype=np.float64).reshape(-1)
    order = np.argsort(-np.abs(c))[:top_n]
    rows: list[dict[str, float | int | list[int]]] = []
    for det_idx in order.tolist():
        ia = det_idx // ndetb
        ib = det_idx % ndetb
        coef = float(c[det_idx])
        rows.append(
            {
                "det_index": int(det_idx),
                "alpha_string_index": int(ia),
                "beta_string_index": int(ib),
                "rank": int(ranks[det_idx]),
                "coefficient": coef,
                "coefficient_abs": abs(coef),
                "weight": coef * coef,
                "alpha_occ": alpha_occ[ia],
                "beta_occ": beta_occ[ib],
            }
        )
    return rows


def write_rank_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "state_label",
        "state_index",
        "state_total_energy_h",
        "excitation_energy_h",
        "excitation_energy_ev",
        "rank",
        "n_determinants",
        "weight",
        "weight_percent",
        "coef_l2",
        "coef_abs_max",
        "coef_abs_mean",
        "energy_projected_h",
        "energy_coupling_h",
        "energy_total_h",
        "energy_total_ev",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_topdet_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "state_label",
        "state_index",
        "det_index",
        "alpha_string_index",
        "beta_string_index",
        "rank",
        "coefficient",
        "coefficient_abs",
        "weight",
        "alpha_occ",
        "beta_occ",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            row_out = dict(row)
            row_out["alpha_occ"] = json.dumps(row_out["alpha_occ"])
            row_out["beta_occ"] = json.dumps(row_out["beta_occ"])
            writer.writerow(row_out)


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    mol = build_h2_mol(args.bond_length, args.basis)
    mf = scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.max_cycle = 200
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("RHF did not converge.")

    mo = np.asarray(mf.mo_coeff, dtype=np.float64)
    h1_mo = mo.T @ np.asarray(mf.get_hcore(), dtype=np.float64) @ mo
    eri_mo = ao2mo.restore(1, ao2mo.kernel(mol, mo), mo.shape[1])

    norb = int(mo.shape[1])
    nelec = int(mol.nelectron)
    nalpha = nelec // 2
    nbeta = nelec // 2

    required_singlets = int(args.n_singlet_excited) + 1
    e_roots, ci_roots, ss_values = solve_fci_roots(
        h1_mo,
        eri_mo,
        norb,
        nelec,
        required_singlets=required_singlets,
        singlet_ss_tol=float(args.singlet_ss_tol),
    )

    singlet_indices = [idx for idx, ss in enumerate(ss_values) if ss < float(args.singlet_ss_tol)]
    if len(singlet_indices) < required_singlets:
        raise RuntimeError(
            f"Only found {len(singlet_indices)} singlets, need {required_singlets}. "
            "Try smaller requested excited-state count."
        )
    chosen = singlet_indices[:required_singlets]
    ground_idx = chosen[0]

    rank_flat, alpha_occ, beta_occ = build_rank_labels(norb, nalpha, nbeta)
    hmat = build_hamiltonian_matrix(h1_mo, eri_mo, norb, nelec)

    rank_csv_rows: list[dict[str, object]] = []
    topdet_csv_rows: list[dict[str, object]] = []
    states_json: list[dict[str, object]] = []

    e_ground = float(e_roots[ground_idx] + mol.energy_nuc())
    ndetb = int(cistring.num_strings(norb, nbeta))

    for state_order, root_idx in enumerate(chosen[1:], start=1):
        label = f"S{state_order}"
        ci_vec = np.asarray(ci_roots[root_idx], dtype=np.float64).reshape(-1)
        ci_norm = float(np.sum(ci_vec * ci_vec))
        e_elec = float(e_roots[root_idx])
        e_total = float(e_elec + mol.energy_nuc())
        gap_h = float(e_roots[root_idx] - e_roots[ground_idx])
        gap_ev = gap_h * HARTREE_TO_EV

        rank_stats = state_rank_summary(ci_vec=ci_vec, hmat=hmat, ranks=rank_flat)
        for row in rank_stats:
            rank_csv_rows.append(
                {
                    "state_label": label,
                    "state_index": int(root_idx),
                    "state_total_energy_h": e_total,
                    "excitation_energy_h": gap_h,
                    "excitation_energy_ev": gap_ev,
                    **row,
                }
            )

        tops = top_determinants(
            ci_vec=ci_vec,
            ranks=rank_flat,
            alpha_occ=alpha_occ,
            beta_occ=beta_occ,
            ndetb=ndetb,
            top_n=int(args.top_dets),
        )
        for row in tops:
            topdet_csv_rows.append(
                {
                    "state_label": label,
                    "state_index": int(root_idx),
                    **row,
                }
            )

        states_json.append(
            {
                "state_label": label,
                "state_root_index": int(root_idx),
                "state_total_energy_h": e_total,
                "state_total_energy_ev": e_total * HARTREE_TO_EV,
                "excitation_energy_h": gap_h,
                "excitation_energy_ev": gap_ev,
                "ci_norm": ci_norm,
                "spin_square": float(ss_values[root_idx]),
                "rank_summary": rank_stats,
                "top_determinants": tops,
            }
        )

    report = {
        "system": {
            "molecule": "H2",
            "bond_length_angstrom": float(args.bond_length),
            "basis": str(args.basis),
            "norb": norb,
            "nelec": nelec,
            "nalpha": nalpha,
            "nbeta": nbeta,
            "nuclear_repulsion_h": float(mol.energy_nuc()),
            "rhf_total_energy_h": float(mf.e_tot),
        },
        "fci": {
            "requested_singlet_excited_states": int(args.n_singlet_excited),
            "selected_singlet_root_indices": [int(x) for x in chosen],
            "selected_ground_root_index": int(ground_idx),
            "ground_total_energy_h": e_ground,
            "ground_total_energy_ev": e_ground * HARTREE_TO_EV,
        },
        "states": states_json,
    }

    json_path = outdir / "h2_fci_excitation_rank_report.json"
    rank_csv_path = outdir / "h2_fci_excitation_rank_summary.csv"
    topdet_csv_path = outdir / "h2_fci_excitation_top_determinants.csv"

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_rank_csv(rank_csv_path, rank_csv_rows)
    write_topdet_csv(topdet_csv_path, topdet_csv_rows)

    print(f"[done] wrote JSON: {json_path}")
    print(f"[done] wrote rank CSV: {rank_csv_path}")
    print(f"[done] wrote top determinants CSV: {topdet_csv_path}")
    print("")
    for st in states_json:
        print(
            f"{st['state_label']}: E={st['state_total_energy_h']:.10f} Eh, "
            f"gap={st['excitation_energy_ev']:.6f} eV, <S^2>={st['spin_square']:.3e}"
        )
        for row in st["rank_summary"]:
            print(
                f"  rank={row['rank']}: weight={row['weight_percent']:.4f}%, "
                f"coef_max={row['coef_abs_max']:.6f}, E_total={row['energy_total_h']:.8f} Eh"
            )


if __name__ == "__main__":
    main()
