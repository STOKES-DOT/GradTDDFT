#!/usr/bin/env python3
"""GPU4PySCF-forward Neural XC CCSD(T) smoke training."""

from __future__ import annotations

import argparse
import copy
from dataclasses import is_dataclass, replace
import json
import os
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import h5py
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import optax
from pyscf.data import elements

from td_graddft import neural_xc
from td_graddft.data.molecule import MoleculeSpec, ANGSTROM_TO_BOHR
from td_graddft.jax_runtime import configure_jax_persistent_cache
from td_graddft.jax_libxc import b3lyp_component_basis
from td_graddft.scf import RKSConfig
from td_graddft.scf.builders import restricted_molecule_from_spec_with_gpu4pyscf_rks
from td_graddft.training import (
    GroundStateDatum,
    GroundStateTrainingConfig,
    create_train_state_from_molecule,
    make_ground_state_loss_and_grad,
    make_ground_state_predictor,
    make_ground_state_train_step,
)
from td_graddft.training.trainer import _tree_abs_max, _tree_l2_norm

HARTREE_TO_EV = 27.211386245988


class _H5DatumRef:
    __slots__ = ("index", "shape_signature")

    def __init__(self, *, index: int, shape_signature: str):
        self.index = int(index)
        self.shape_signature = str(shape_signature)


def _symbols_from_z(atomic_numbers: np.ndarray) -> tuple[str, ...]:
    return tuple(elements.ELEMENTS[int(z)] for z in atomic_numbers)


def _load_samples(
    path: Path,
    n_samples: int,
    max_atoms: int,
    *,
    shuffle: bool = True,
    seed: int = 42,
) -> list[dict[str, object]]:
    grouped_samples: list[list[dict[str, object]]] = []
    with h5py.File(path, "r") as f:
        for key in sorted(f.keys()):
            grp = f[key]
            atomic_numbers = np.asarray(grp["atomic_numbers"][:], dtype=np.int32)
            coords_all = np.asarray(grp["coordinates"][:], dtype=np.float64)
            energies = np.asarray(grp["ccsd(t)_cbs.energy"][:], dtype=np.float64)
            if coords_all.ndim != 3:
                continue
            if atomic_numbers.shape[0] != coords_all.shape[1]:
                continue
            if atomic_numbers.shape[0] > max_atoms:
                continue
            if int(atomic_numbers.sum()) % 2 != 0:
                continue
            finite = np.flatnonzero(np.isfinite(energies))
            group_samples: list[dict[str, object]] = []
            for idx in finite:
                group_samples.append(
                    {
                        "name": key,
                        "conformer_idx": int(idx),
                        "atomic_numbers": atomic_numbers,
                        "coordinates": coords_all[int(idx)],
                        "ccsdt_energy": float(energies[int(idx)]),
                    }
                )
            if group_samples:
                grouped_samples.append(group_samples)
    if shuffle:
        rng = np.random.default_rng(int(seed))
        rng.shuffle(grouped_samples)
        for group_samples in grouped_samples:
            rng.shuffle(group_samples)
    samples: list[dict[str, object]] = []
    conformer_slot = 0
    while len(samples) < n_samples:
        added = False
        for group_samples in grouped_samples:
            if conformer_slot >= len(group_samples):
                continue
            samples.append(group_samples[conformer_slot])
            added = True
            if len(samples) >= n_samples:
                return samples
        if not added:
            break
        conformer_slot += 1
    return samples


def _sample_group_count(samples: list[dict[str, object]]) -> int:
    return len({str(sample["name"]) for sample in samples})


def _molecule_spec(sample: dict[str, object]) -> MoleculeSpec:
    atomic_numbers = np.asarray(sample["atomic_numbers"], dtype=np.int32)
    coords_angstrom = np.asarray(sample["coordinates"], dtype=np.float64)
    return MoleculeSpec(
        symbols=_symbols_from_z(atomic_numbers),
        coords_bohr=jnp.asarray(coords_angstrom * ANGSTROM_TO_BOHR),
        charges=jnp.asarray(atomic_numbers, dtype=jnp.float64),
        charge=0,
        spin=0,
        unit="Bohr",
    )


def _mean_metric(metrics: dict[str, object], key: str) -> float | None:
    if key not in metrics:
        return None
    arr = np.asarray(jax.device_get(metrics[key]), dtype=float)
    if arr.size == 0:
        return None
    return float(np.mean(arr))


def _scalar_metric(metrics: dict[str, object], key: str) -> float:
    value = _mean_metric(metrics, key)
    return float("nan") if value is None else value


def _parse_hidden_dims(value: str) -> tuple[int, ...]:
    dims = tuple(int(item.strip()) for item in str(value).split(",") if item.strip())
    if not dims:
        raise argparse.ArgumentTypeError("--hidden-dims must contain at least one layer width.")
    if any(width <= 0 for width in dims):
        raise argparse.ArgumentTypeError("--hidden-dims entries must be positive integers.")
    return dims


def _parse_float_tuple(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not values:
        raise argparse.ArgumentTypeError("value must contain at least one float.")
    return values


def _build_learning_rate_schedule(
    *,
    learning_rate: float,
    lr_decay_epochs: int = 0,
    lr_decay_factor: float = 1.0,
    steps_per_epoch: int = 1,
):
    if int(lr_decay_epochs) <= 0 or float(lr_decay_factor) == 1.0:
        return float(learning_rate)
    if float(lr_decay_factor) <= 0.0:
        raise ValueError("--lr-decay-factor must be positive.")
    transition_steps = max(1, int(lr_decay_epochs)) * max(1, int(steps_per_epoch))
    return optax.exponential_decay(
        init_value=float(learning_rate),
        transition_steps=transition_steps,
        decay_rate=float(lr_decay_factor),
        staircase=True,
    )


def _optimizer_steps_per_epoch(update_mode: str, num_train_batches: int) -> int:
    if update_mode == "epoch_accum":
        return 1
    if update_mode == "per_batch":
        return max(1, int(num_train_batches))
    raise ValueError(f"Unsupported update_mode={update_mode!r}")


def _build_optimizer(
    *,
    learning_rate: float,
    gradient_clip_norm: float = 0.0,
    lr_decay_epochs: int = 0,
    lr_decay_factor: float = 1.0,
    steps_per_epoch: int = 1,
) -> optax.GradientTransformation:
    transforms: list[optax.GradientTransformation] = []
    if float(gradient_clip_norm) > 0.0:
        transforms.append(optax.clip_by_global_norm(float(gradient_clip_norm)))
    transforms.append(
        optax.adam(
            _build_learning_rate_schedule(
                learning_rate=learning_rate,
                lr_decay_epochs=lr_decay_epochs,
                lr_decay_factor=lr_decay_factor,
                steps_per_epoch=steps_per_epoch,
            )
        )
    )
    return optax.chain(*transforms) if len(transforms) > 1 else transforms[0]


def _tree_scale(tree, scale: float):
    return jax.tree_util.tree_map(
        lambda leaf: jnp.asarray(leaf) * jnp.asarray(scale, dtype=jnp.asarray(leaf).dtype),
        tree,
    )


def _tree_add(left, right):
    return jax.tree_util.tree_map(lambda a, b: jnp.asarray(a) + jnp.asarray(b), left, right)


def _accumulate_weighted_grads(grad_sum, grads, *, weight: float):
    weighted_grads = _tree_scale(grads, float(weight))
    if grad_sum is None:
        return weighted_grads
    return _tree_add(grad_sum, weighted_grads)


def _normalize_accumulated_grads(grad_sum, *, total_weight: float):
    if grad_sum is None:
        raise ValueError("Cannot normalize empty gradient accumulation.")
    return _tree_scale(grad_sum, 1.0 / max(float(total_weight), 1.0))


def _batch_loss_weight(batch: list[GroundStateDatum]) -> float:
    total = 0.0
    for datum in batch:
        total += float(jax.device_get(jnp.asarray(getattr(datum, "weight", 1.0))))
    return total


def _resolve_split_counts(
    *,
    n_samples: int,
    train_samples: int | None,
    test_samples: int,
) -> tuple[int, int]:
    total = int(n_samples)
    test_count = int(test_samples)
    if total <= 0:
        raise ValueError("--n-samples must be positive.")
    if test_count < 0:
        raise ValueError("--test-samples must be non-negative.")
    if train_samples is None:
        train_count = total - test_count
    else:
        train_count = int(train_samples)
        if train_count + test_count != total:
            raise ValueError(
                "Requested train/test split must satisfy "
                "--train-samples + --test-samples == --n-samples."
            )
    if train_count <= 0:
        raise ValueError("Train split must contain at least one sample.")
    if train_count + test_count != total:
        raise ValueError(
            "Requested train/test split must satisfy "
            "--train-samples + --test-samples == --n-samples."
        )
    return train_count, test_count


def _split_indices(
    *,
    n_samples: int,
    train_samples: int,
    test_samples: int,
    shuffle: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    indices = np.arange(int(n_samples), dtype=np.int32)
    if bool(shuffle):
        rng = np.random.default_rng(int(seed))
        rng.shuffle(indices)
    train_count = int(train_samples)
    test_count = int(test_samples)
    return indices[:train_count], indices[train_count : train_count + test_count]


def _mean_absolute_error_ev(predicted: list[float] | np.ndarray, targets: list[float] | np.ndarray) -> float:
    predicted_arr = np.asarray(predicted, dtype=float)
    target_arr = np.asarray(targets, dtype=float)
    if predicted_arr.size == 0 or target_arr.size == 0:
        return float("nan")
    if predicted_arr.shape != target_arr.shape:
        raise ValueError(
            f"Prediction and target shapes must match, got {predicted_arr.shape} and {target_arr.shape}."
        )
    return float(np.mean(np.abs(predicted_arr - target_arr)) * HARTREE_TO_EV)


def _host_numpy_array(value, *, dtype=np.float64) -> np.ndarray:
    return np.asarray(jax.device_get(value), dtype=dtype)


def _spin_summed_density_matrix_host(molecule) -> np.ndarray:
    density_matrix = _host_numpy_array(molecule.rdm1)
    if density_matrix.ndim == 3:
        return np.asarray(density_matrix.sum(axis=0), dtype=np.float64)
    return np.asarray(density_matrix, dtype=np.float64)


def _coulomb_energy_host(molecule, density_matrix: np.ndarray) -> float:
    density_matrix = 0.5 * (density_matrix + density_matrix.T)
    rep_tensor = getattr(molecule, "rep_tensor", None)
    if rep_tensor is not None:
        rep = _host_numpy_array(rep_tensor)
        if rep.size > 0:
            potential = np.einsum("pqrs,rs->pq", rep, density_matrix, optimize=True)
            return 0.5 * float(np.einsum("pq,pq->", density_matrix, potential, optimize=True))
    pair = getattr(molecule, "eri_pair_matrix", None)
    if pair is None:
        raise AttributeError("Molecule-like object must define rep_tensor or eri_pair_matrix.")
    pair_matrix = _host_numpy_array(pair)
    if pair_matrix.size == 0:
        raise ValueError("Coulomb energy requires full AO ERI or packed AO-pair ERI data.")
    nao = int(density_matrix.shape[0])
    rows, cols = np.tril_indices(nao)
    multiplicity = np.where(rows == cols, 1.0, 2.0).astype(np.float64)
    density_pair = density_matrix[rows, cols] * multiplicity
    j_pair = pair_matrix @ density_pair
    j_matrix = np.zeros_like(density_matrix)
    j_matrix[rows, cols] = j_pair
    j_matrix[cols, rows] = j_pair
    j_matrix = 0.5 * (j_matrix + j_matrix.T)
    return 0.5 * float(np.einsum("pq,pq->", density_matrix, j_matrix, optimize=True))


def _fixed_density_non_xc_energy(molecule) -> float:
    density_matrix = _spin_summed_density_matrix_host(molecule)
    one_body = float(np.einsum("pq,pq->", density_matrix, _host_numpy_array(molecule.h1e)))
    coulomb = _coulomb_energy_host(molecule, density_matrix)
    nuclear = float(np.asarray(jax.device_get(molecule.nuclear_repulsion)))
    return one_body + coulomb + nuclear


def _make_fixed_density_xc_predictor(functional):
    def predictor(params, molecule):
        if not hasattr(functional, "energy_from_molecule"):
            raise AttributeError("fixed-density XC residual training requires energy_from_molecule.")
        return functional.energy_from_molecule(params, molecule), molecule

    return predictor


def _is_gpu4pyscf_scf_nonconvergence(exc: BaseException) -> bool:
    message = str(exc)
    return "GPU4PySCF exact RKS SCF did not converge" in message


def _is_recoverable_build_resource_failure(exc: BaseException) -> bool:
    message = str(exc)
    return (
        "RESOURCE_EXHAUSTED" in message
        or "Out of memory" in message
        or "out of memory" in message
    )


def _is_skippable_build_failure(exc: BaseException) -> bool:
    return _is_gpu4pyscf_scf_nonconvergence(exc) or _is_recoverable_build_resource_failure(exc)


def _cache_array_leaf(value):
    if isinstance(value, (str, bytes)) or value is None:
        return value
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            return np.asarray(jax.device_get(value))
        except Exception:
            return value
    return value


def _hostify_for_cache(value):
    return jax.tree_util.tree_map(_cache_array_leaf, value)


def _release_build_device_memory() -> None:
    try:
        jax.clear_caches()
    except Exception:
        pass
    try:
        import cupy as cp

        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    except Exception:
        pass


def _built_data_cache_config(
    *,
    h5_path: str,
    n_samples: int,
    sample_buffer: int,
    shuffle_samples: bool,
    sample_seed: int,
    max_atoms: int,
    basis: str,
    ref_xc: str,
    grids_level: int,
    scf_max_cycle: int,
    scf_conv_tol: float,
    hfx_omega_values: tuple[float, ...],
    hfx_aux: bool,
    compute_response_eri_slices: bool,
) -> dict[str, object]:
    return {
        "h5_path": str(h5_path),
        "n_samples": int(n_samples),
        "sample_buffer": int(sample_buffer),
        "sample_selection": "round_robin_groups_v1",
        "shuffle_samples": bool(shuffle_samples),
        "sample_seed": int(sample_seed),
        "max_atoms": int(max_atoms),
        "basis": str(basis),
        "ref_xc": str(ref_xc),
        "grids_level": int(grids_level),
        "scf_max_cycle": int(scf_max_cycle),
        "scf_conv_tol": float(scf_conv_tol),
        "hfx_omega_values": [float(value) for value in hfx_omega_values],
        "hfx_aux": bool(hfx_aux),
        "compute_response_eri_slices": bool(compute_response_eri_slices),
    }


def _write_built_data_cache(
    path: Path,
    *,
    config: dict[str, object],
    data: list[GroundStateDatum],
    samples: list[dict[str, object]],
    built_samples: list[dict[str, object]],
    skipped_samples: list[dict[str, object]],
    reference_energies: list[float],
    build_elapsed: float,
    fixed_non_xc_offsets: list[float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "config": config,
        "data": _hostify_for_cache(data),
        "samples": samples,
        "built_samples": built_samples,
        "skipped_samples": skipped_samples,
        "reference_energies": reference_energies,
        "fixed_non_xc_offsets": [] if fixed_non_xc_offsets is None else fixed_non_xc_offsets,
        "build_elapsed": float(build_elapsed),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def _load_built_data_cache(path: Path, *, config: dict[str, object]) -> dict[str, object]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if payload.get("version") != 1:
        raise RuntimeError(f"Unsupported built-data cache version in {path}.")
    cached_config = payload.get("config")
    if cached_config != config:
        raise RuntimeError(
            f"Built-data cache config mismatch for {path}. "
            "Use --rebuild-built-data-cache to regenerate it."
        )
    data = list(payload.get("data", ()))
    if len(data) < int(config["n_samples"]):
        raise RuntimeError(
            f"Built-data cache {path} contains {len(data)} samples, "
            f"expected at least {config['n_samples']}."
        )
    return payload


def _h5_bytes_dtype():
    return h5py.vlen_dtype(np.dtype("uint8"))


def _h5_string_dtype():
    return h5py.string_dtype(encoding="utf-8")


def _pickle_to_uint8(value) -> np.ndarray:
    raw = pickle.dumps(_hostify_for_cache(value), protocol=pickle.HIGHEST_PROTOCOL)
    return np.frombuffer(raw, dtype=np.uint8)


def _unpickle_from_uint8(value):
    return pickle.loads(np.asarray(value, dtype=np.uint8).tobytes())


def _create_built_data_h5(path: Path, *, config: dict[str, object], samples: list[dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = h5py.File(path, "w")
    handle.attrs["version"] = 1
    handle.attrs["config_json"] = json.dumps(config, sort_keys=True)
    handle.attrs["candidate_count"] = int(len(samples))
    handle.attrs["candidate_group_count"] = int(_sample_group_count(samples))
    handle.create_dataset("datum_blobs", shape=(0,), maxshape=(None,), dtype=_h5_bytes_dtype())
    handle.create_dataset("shape_signature", shape=(0,), maxshape=(None,), dtype=_h5_string_dtype())
    handle.create_dataset("sample_name", shape=(0,), maxshape=(None,), dtype=_h5_string_dtype())
    handle.create_dataset("conformer_idx", shape=(0,), maxshape=(None,), dtype=np.int64)
    handle.create_dataset("natoms", shape=(0,), maxshape=(None,), dtype=np.int64)
    handle.create_dataset("target_total_energy", shape=(0,), maxshape=(None,), dtype=np.float64)
    handle.create_dataset("reference_energy", shape=(0,), maxshape=(None,), dtype=np.float64)
    handle.create_dataset("fixed_non_xc_offset", shape=(0,), maxshape=(None,), dtype=np.float64)
    handle.create_dataset("skipped_json", shape=(0,), maxshape=(None,), dtype=_h5_string_dtype())
    return handle


def _append_built_data_h5_sample(
    handle,
    *,
    datum: GroundStateDatum,
    sample: dict[str, object],
    reference_energy: float,
    fixed_non_xc_offset: float,
) -> _H5DatumRef:
    idx = int(handle["datum_blobs"].shape[0])
    for name in (
        "datum_blobs",
        "shape_signature",
        "sample_name",
        "conformer_idx",
        "natoms",
        "target_total_energy",
        "reference_energy",
        "fixed_non_xc_offset",
    ):
        handle[name].resize((idx + 1,))
    signature = _shape_signature_text(datum)
    handle["datum_blobs"][idx] = _pickle_to_uint8(datum)
    handle["shape_signature"][idx] = signature
    handle["sample_name"][idx] = str(sample["name"])
    handle["conformer_idx"][idx] = int(sample["conformer_idx"])
    handle["natoms"][idx] = int(len(sample["atomic_numbers"]))
    handle["target_total_energy"][idx] = float(sample["ccsdt_energy"])
    handle["reference_energy"][idx] = float(reference_energy)
    handle["fixed_non_xc_offset"][idx] = float(fixed_non_xc_offset)
    handle.flush()
    return _H5DatumRef(index=idx, shape_signature=signature)


def _append_built_data_h5_skip(handle, skipped_sample: dict[str, object]) -> None:
    idx = int(handle["skipped_json"].shape[0])
    handle["skipped_json"].resize((idx + 1,))
    handle["skipped_json"][idx] = json.dumps(skipped_sample, sort_keys=True)
    handle.flush()


def _load_built_data_h5(path: Path, *, config: dict[str, object]) -> dict[str, object]:
    with h5py.File(path, "r") as handle:
        if int(handle.attrs.get("version", -1)) != 1:
            raise RuntimeError(f"Unsupported built-data HDF5 cache version in {path}.")
        cached_config = json.loads(str(handle.attrs.get("config_json", "{}")))
        if cached_config != config:
            raise RuntimeError(
                f"Built-data HDF5 cache config mismatch for {path}. "
                "Use --rebuild-built-data-cache to regenerate it."
            )
        count = int(handle["datum_blobs"].shape[0])
        if count < int(config["n_samples"]):
            raise RuntimeError(
                f"Built-data HDF5 cache {path} contains {count} samples, "
                f"expected at least {config['n_samples']}."
            )
        names = [str(value) for value in handle["sample_name"].asstr()[:count]]
        conformers = np.asarray(handle["conformer_idx"][:count], dtype=np.int64)
        targets = np.asarray(handle["target_total_energy"][:count], dtype=float)
        built_samples = [
            {
                "name": names[i],
                "conformer_idx": int(conformers[i]),
                "ccsdt_energy": float(targets[i]),
            }
            for i in range(count)
        ]
        skipped = [
            json.loads(str(item))
            for item in handle["skipped_json"].asstr()[:]
        ]
        return {
            "refs": [
                _H5DatumRef(index=i, shape_signature=str(sig))
                for i, sig in enumerate(handle["shape_signature"].asstr()[:count])
            ],
            "samples": built_samples,
            "built_samples": built_samples,
            "skipped_samples": skipped,
            "reference_energies": np.asarray(handle["reference_energy"][:count], dtype=float),
            "fixed_non_xc_offsets": np.asarray(
                handle["fixed_non_xc_offset"][:count],
                dtype=float,
            ),
            "candidate_count": int(handle.attrs.get("candidate_count", count)),
            "candidate_group_count": int(handle.attrs.get("candidate_group_count", 0)),
        }


class _H5DatumStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self._handle = None

    def _file(self):
        if self._handle is None:
            self._handle = h5py.File(self.path, "r")
        return self._handle

    def read(self, index: int) -> GroundStateDatum:
        return _unpickle_from_uint8(self._file()["datum_blobs"][int(index)])

    def close(self) -> None:
        if self._handle is not None:
            self._handle.close()
            self._handle = None


def _materialize_datum(item, store: _H5DatumStore | None):
    if isinstance(item, _H5DatumRef):
        if store is None:
            raise RuntimeError("HDF5 datum reference requires an open datum store.")
        return store.read(item.index)
    return item


def _materialize_batch(batch: list[object], store: _H5DatumStore | None) -> list[GroundStateDatum]:
    return [_materialize_datum(item, store) for item in batch]


def _array_leaf_signature(value) -> tuple[tuple[int, ...], str]:
    if value is None:
        return ((), "None")
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return (tuple(int(dim) for dim in value.shape), str(value.dtype))
    try:
        arr = jnp.asarray(value)
    except Exception:
        return ((), type(value).__name__)
    return (tuple(int(dim) for dim in arr.shape), str(arr.dtype))


def _pytree_shape_signature(value) -> tuple[object, ...]:
    if isinstance(value, _H5DatumRef):
        return ("h5_datum_ref", value.shape_signature)
    leaves, treedef = jax.tree_util.tree_flatten(value)
    return (str(treedef), tuple(_array_leaf_signature(leaf) for leaf in leaves))


def _shape_signature_text(value) -> str:
    return json.dumps(_pytree_shape_signature(value), sort_keys=True)


def _make_batches(
    items: list[object],
    batch_size: int,
    *,
    bucket_by_shape: bool = False,
) -> list[list[object]]:
    if not items:
        return []
    if bool(bucket_by_shape):
        groups: dict[tuple[object, ...], list[object]] = {}
        for item in items:
            groups.setdefault(_pytree_shape_signature(item), []).append(item)
        size = int(batch_size)
        batches: list[list[object]] = []
        for group in groups.values():
            if size <= 0 or size >= len(group):
                batches.append(list(group))
            else:
                batches.extend(
                    list(group[start : start + size]) for start in range(0, len(group), size)
                )
        return batches
    if int(batch_size) <= 0 or int(batch_size) >= len(items):
        return [list(items)]
    size = int(batch_size)
    return [list(items[start : start + size]) for start in range(0, len(items), size)]


def _batches_for_epoch(
    batches: list[list[object]],
    *,
    epoch: int,
    shuffle: bool,
    seed: int,
) -> list[list[object]]:
    if not bool(shuffle) or len(batches) <= 1:
        return batches
    rng = np.random.default_rng(int(seed) + int(epoch))
    order = rng.permutation(len(batches))
    return [batches[int(idx)] for idx in order]


def _replace_object_fields(obj, **updates):
    if is_dataclass(obj):
        valid = {name for name in getattr(obj, "__dataclass_fields__", {})}
        return replace(obj, **{key: value for key, value in updates.items() if key in valid})
    copied = copy.copy(obj)
    for key, value in updates.items():
        if hasattr(copied, key):
            setattr(copied, key, value)
    return copied


def _strip_runtime_static_fields_for_fixed_density(
    data: list[GroundStateDatum],
    *,
    drop_ground_state_integrals: bool = False,
) -> list[GroundStateDatum]:
    stripped: list[GroundStateDatum] = []
    for datum in data:
        molecule = datum.molecule
        updates = {}
        if hasattr(molecule, "runtime_scf_backend"):
            updates["runtime_scf_backend"] = None
        if hasattr(molecule, "runtime_scf_options"):
            updates["runtime_scf_options"] = None
        if bool(drop_ground_state_integrals):
            dtype = jnp.asarray(getattr(molecule, "rdm1", jnp.asarray(0.0))).dtype
            for name in (
                "eri_pair_matrix",
                "df_factors",
                "eri_ovov",
                "eri_ovvo",
                "eri_oovv",
            ):
                if hasattr(molecule, name):
                    updates[name] = None
            if hasattr(molecule, "rep_tensor"):
                updates["rep_tensor"] = jnp.zeros((0,), dtype=dtype)
        if updates:
            molecule = _replace_object_fields(molecule, **updates)
        stripped.append(replace(datum, molecule=molecule))
    return stripped


def _to_fixed_density_xc_residual_data(
    data: list[GroundStateDatum],
) -> tuple[list[GroundStateDatum], np.ndarray]:
    offsets = np.asarray(
        [_fixed_density_non_xc_energy(datum.molecule) for datum in data],
        dtype=float,
    )
    residual_data: list[GroundStateDatum] = []
    for datum, offset in zip(data, offsets, strict=True):
        residual_target = jnp.asarray(datum.target_total_energy) - jnp.asarray(
            offset,
            dtype=jnp.asarray(datum.target_total_energy).dtype,
        )
        residual_data.append(replace(datum, target_total_energy=residual_target))
    residual_data = _strip_runtime_static_fields_for_fixed_density(
        residual_data,
        drop_ground_state_integrals=True,
    )
    return residual_data, offsets


def _build_training_config(
    *,
    training_mode: str,
    energy_normalization: str,
    energy_loss: str,
    train_scf_max_cycle: int,
    train_scf_damping: float,
    implicit_response_backend: str,
    implicit_diff_max_iter: int,
) -> GroundStateTrainingConfig:
    energy_mse_weight, energy_mae_weight = _energy_loss_weights(energy_loss)
    base_kwargs = {
        "energy_mse_weight": energy_mse_weight,
        "energy_mae_weight": energy_mae_weight,
        "energy_normalization": energy_normalization,
        "scf_max_cycle": train_scf_max_cycle,
        "scf_damping": train_scf_damping,
        "scf_conv_tol_density": 1e-8,
        "scf_vxc_clip": 20.0,
        "scf_require_convergence": False,
    }
    if training_mode == "fixed_density":
        return GroundStateTrainingConfig(
            mode="fixed_density",
            scf_runtime_forward_backend="auto",
            implicit_response_backend="jax",
            **base_kwargs,
        )
    if training_mode == "self_consistent":
        return GroundStateTrainingConfig(
            mode="self_consistent",
            scf_gradient_mode="impl",
            scf_implicit_forward_mode="input_state",
            scf_runtime_forward_backend="gpu4pyscf_rks",
            implicit_response_backend=implicit_response_backend,
            scf_implicit_diff_max_iter=implicit_diff_max_iter,
            **base_kwargs,
        )
    raise ValueError(f"Unsupported training mode: {training_mode!r}")


def _energy_loss_weights(energy_loss: str) -> tuple[float, float]:
    if energy_loss == "mae":
        return 0.0, 1.0
    if energy_loss == "mse":
        return 1.0, 0.0
    if energy_loss == "mae_mse":
        return 1.0, 1.0
    raise ValueError(f"Unsupported energy_loss={energy_loss!r}")


def _predict_final_energies(
    params,
    functional,
    data: list[object],
    training_config: GroundStateTrainingConfig,
    predictor=None,
    store: _H5DatumStore | None = None,
) -> list[float]:
    predictor = (
        make_ground_state_predictor(functional, training_config=training_config)
        if predictor is None
        else predictor
    )
    predicted: list[float] = []
    for item in data:
        datum = _materialize_datum(item, store)
        energy, _ = predictor(params, datum.molecule)
        predicted.append(float(jax.device_get(energy)))
    return predicted


def _evaluate_energy_mae_ev(
    params,
    functional,
    data: list[object],
    training_config: GroundStateTrainingConfig,
    targets: np.ndarray,
    predictor=None,
    store: _H5DatumStore | None = None,
) -> float:
    if not data:
        return float("nan")
    predicted = _predict_final_energies(
        params,
        functional,
        data,
        training_config,
        predictor=predictor,
        store=store,
    )
    return _mean_absolute_error_ev(predicted, targets)


def _build_functional(
    *,
    hidden_dims: tuple[int, ...],
    architecture: str,
    hf_input_mode: str = "spin_resolved",
    hfx_channels: int = 2,
    sigmoid_scale_factor: float = 2.0,
):
    return neural_xc.Functional(
        semilocal_xc=tuple(b3lyp_component_basis()),
        hidden_dims=hidden_dims,
        architecture=architecture,
        input_feature_mode="canonical",
        hf_input_mode=hf_input_mode,
        hfx_channels=hfx_channels,
        sigmoid_scale_factor=float(sigmoid_scale_factor),
        name="gpu4pyscf_ccsdt_smoke",
    )


def _epoch_monitor_row(
    *,
    epoch: int,
    epoch_elapsed: float,
    loss: float,
    metrics: dict[str, object],
) -> dict[str, float]:
    return {
        "epoch": float(epoch),
        "loss": float(loss),
        "epoch_time_s": float(epoch_elapsed),
        "energy_mae": _scalar_metric(metrics, "energy_mae"),
        "normalized_energy_mae": _scalar_metric(metrics, "normalized_energy_mae"),
        "grad_norm": _scalar_metric(metrics, "grad_norm"),
        "raw_grad_norm": _scalar_metric(metrics, "raw_grad_norm"),
        "grad_abs_max": _scalar_metric(metrics, "grad_abs_max"),
        "nonfinite_grad_fraction": _scalar_metric(metrics, "nonfinite_grad_fraction"),
        "param_update_norm": _scalar_metric(metrics, "param_update_norm"),
        "param_norm": _scalar_metric(metrics, "param_norm"),
        "mean_scf_cycles": _scalar_metric(metrics, "scf_cycles"),
        "mean_scf_final_rms": _scalar_metric(metrics, "scf_final_rms"),
    }


def _aggregate_epoch_rows(
    *,
    epoch: int,
    epoch_elapsed: float,
    batch_rows: list[dict[str, float]],
    batch_sizes: list[float],
) -> dict[str, float]:
    if not batch_rows:
        raise ValueError("Cannot aggregate an epoch with no batch rows.")
    weights = np.asarray(batch_sizes, dtype=float)
    total_weight = float(np.sum(weights))
    row = {
        "epoch": float(epoch),
        "epoch_time_s": float(epoch_elapsed),
    }
    for key in batch_rows[0]:
        if key in {"epoch", "epoch_time_s"}:
            continue
        values = np.asarray([float(batch_row[key]) for batch_row in batch_rows], dtype=float)
        if total_weight <= 0.0 or np.all(np.isnan(values)):
            row[key] = float("nan")
        else:
            row[key] = float(np.nansum(values * weights) / total_weight)
    return row


def _format_epoch_log(*, epoch: int, total_epochs: int, row: dict[str, float]) -> str:
    base = (
        f"epoch {epoch:04d}/{total_epochs}: loss={row['loss']:.8e} "
        f"mae_ha={row['energy_mae']:.3e} "
        f"time={row['epoch_time_s']:.3f}s grad_norm={row['grad_norm']:.3e} "
        f"raw_grad_norm={row['raw_grad_norm']:.3e} "
        f"grad_abs_max={row['grad_abs_max']:.3e} "
        f"update_norm={row['param_update_norm']:.3e} "
        f"nonfinite={row['nonfinite_grad_fraction']:.3e} "
        f"cycles={row['mean_scf_cycles']:.2f} rms={row['mean_scf_final_rms']:.3e}"
    )
    if "eval_train_mae_ev" in row:
        base += (
            f" eval_train_mae_ev={row['eval_train_mae_ev']:.4f} "
            f"eval_test_mae_ev={row['eval_test_mae_ev']:.4f}"
        )
    return base


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", default="datasets/ani/ani1x_ccsdt_10mol_smoke.h5")
    parser.add_argument("--n-samples", type=int, default=10)
    parser.add_argument("--train-samples", type=int, default=None)
    parser.add_argument("--test-samples", type=int, default=0)
    parser.add_argument(
        "--sample-buffer",
        type=int,
        default=0,
        help="Load this many extra candidate samples so skipped build failures can be replaced.",
    )
    parser.add_argument(
        "--shuffle-samples",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle HDF5 groups and conformers before round-robin sample selection.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Seed for deterministic HDF5 group/conformer sample shuffling.",
    )
    parser.add_argument(
        "--skip-unconverged-builds",
        action="store_true",
        help="Skip candidate molecules whose initial GPU4PySCF reference SCF does not converge.",
    )
    parser.add_argument("--shuffle-split", action="store_true")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--basis", default="6-31g*")
    parser.add_argument("--ref-xc", default="b3lyp")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Mini-batch size for train_data. Default 0 uses a full-batch update.",
    )
    parser.add_argument(
        "--update-mode",
        default="per_batch",
        choices=("epoch_accum", "per_batch"),
        help=(
            "per_batch matches GradDFT's train kernel: value_and_grad on the current "
            "batch, then one optimizer update. epoch_accum is a full-epoch gradient "
            "accumulation diagnostic."
        ),
    )
    parser.add_argument(
        "--bucket-by-shape",
        action="store_true",
        help="Group mini-batches by JAX pytree shape to reduce recompilation on heterogeneous molecules.",
    )
    parser.add_argument(
        "--shuffle-train-batches",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Shuffle train batches each epoch, matching GradDFT loader(randomize=True).",
    )
    parser.add_argument(
        "--train-batch-seed",
        type=int,
        default=0,
        help="Seed used for deterministic per-epoch train-batch shuffling.",
    )
    parser.add_argument(
        "--jit-train-step",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="JIT the fixed-density train step. Useful for small or shape-homogeneous batches.",
    )
    parser.add_argument(
        "--keep-runtime-static-fields",
        action="store_true",
        help="Keep GPU4PySCF runtime SCF static fields on fixed-density training molecules.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument(
        "--lr-decay-epochs",
        type=int,
        default=0,
        help="Halve/scale the learning rate every N epochs when positive.",
    )
    parser.add_argument(
        "--lr-decay-factor",
        type=float,
        default=1.0,
        help="Multiplicative learning-rate factor applied every --lr-decay-epochs.",
    )
    parser.add_argument(
        "--energy-normalization",
        default="none",
        choices=("none", "per_electron", "per_atom"),
        help="Normalization applied before energy MAE. Default none means raw total-energy MAE.",
    )
    parser.add_argument(
        "--training-mode",
        default="self_consistent",
        choices=("self_consistent", "fixed_density"),
        help="Use self-consistent implicit-SCF training or fixed-density energy fitting.",
    )
    parser.add_argument(
        "--fixed-density-loss-target",
        default="xc_residual",
        choices=("xc_residual", "total_energy"),
        help=(
            "For fixed-density mode, train either total energy directly or the "
            "equivalent XC residual target E_ref - E_non_xc."
        ),
    )
    parser.add_argument(
        "--energy-loss",
        default="mae",
        choices=("mae", "mse", "mae_mse"),
        help=(
            "Energy objective used for gradients. GradDFT examples optimize "
            "MSE and log MAE as a metric."
        ),
    )
    parser.add_argument(
        "--gradient-clip-norm",
        type=float,
        default=0.0,
        help="Apply optax.clip_by_global_norm before Adam when positive.",
    )
    parser.add_argument("--hidden-dims", default="64,64,64")
    parser.add_argument(
        "--sigmoid-scale-factor",
        type=float,
        default=2.0,
        help=(
            "Output coefficient transform scale. Positive values apply "
            "scale * sigmoid(raw / scale) before coefficient sanitization."
        ),
    )
    parser.add_argument(
        "--hf-input-mode",
        default="spin_resolved",
        choices=("spin_resolved", "total_only"),
    )
    parser.add_argument("--hfx-omega-values", default="0.0,0.4")
    parser.add_argument(
        "--hfx-aux",
        action="store_true",
        help="Store hfx_nu auxiliary integrals for experimental HFX-response paths.",
    )
    parser.add_argument(
        "--no-hfx-aux",
        action="store_true",
        help="Deprecated compatibility flag; cached local-HFX features are the default.",
    )
    parser.add_argument(
        "--compute-response-eri-slices",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Build MO ERI response slices. Disabled by default for ground-state energy training.",
    )
    parser.add_argument(
        "--architecture",
        default="residual",
        choices=("residual", "graddft_residual", "mlp", "simple_mlp"),
    )
    parser.add_argument("--max-atoms", type=int, default=8)
    parser.add_argument("--grids-level", type=int, default=0)
    parser.add_argument("--scf-max-cycle", type=int, default=40)
    parser.add_argument("--scf-conv-tol", type=float, default=1e-9)
    parser.add_argument("--train-scf-max-cycle", type=int, default=12)
    parser.add_argument("--train-scf-damping", type=float, default=0.25)
    parser.add_argument(
        "--implicit-response-backend",
        default="gpu4pyscf_jk",
        choices=("jax", "gpu4pyscf_jk"),
    )
    parser.add_argument("--implicit-diff-max-iter", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument(
        "--eval-first-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Evaluate train/test MAE after epoch 1. Disable for compile-time speed probes.",
    )
    parser.add_argument(
        "--jax-compilation-cache-dir",
        default=None,
        help="Optional persistent JAX compilation cache directory.",
    )
    parser.add_argument(
        "--built-data-cache",
        default=None,
        help="Optional pickle cache for built GroundStateDatum objects.",
    )
    parser.add_argument(
        "--built-data-h5",
        default=None,
        help=(
            "Optional HDF5 cache for built GroundStateDatum objects. In this mode "
            "training keeps only sample references in memory and reads each batch "
            "from the HDF5 input file."
        ),
    )
    parser.add_argument(
        "--rebuild-built-data-cache",
        action="store_true",
        help="Ignore and overwrite --built-data-cache.",
    )
    parser.add_argument("--outdir", default="outputs/ccsdt_gpu4pyscf_smoke")
    args = parser.parse_args()

    jax_cache_dir = configure_jax_persistent_cache(
        cache_dir=args.jax_compilation_cache_dir,
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    hfx_omega_values = _parse_float_tuple(args.hfx_omega_values)
    hfx_aux = bool(args.hfx_aux) and not bool(args.no_hfx_aux)
    train_count, test_count = _resolve_split_counts(
        n_samples=args.n_samples,
        train_samples=args.train_samples,
        test_samples=args.test_samples,
    )
    train_indices, test_indices = _split_indices(
        n_samples=args.n_samples,
        train_samples=train_count,
        test_samples=test_count,
        shuffle=args.shuffle_split,
        seed=args.split_seed,
    )
    eval_every = int(args.eval_every) if int(args.eval_every) > 0 else int(args.log_every)

    print("jax devices:", jax.devices())
    print(
        "allocator env: "
        f"XLA_PYTHON_CLIENT_PREALLOCATE={os.environ.get('XLA_PYTHON_CLIENT_PREALLOCATE')} "
        f"XLA_PYTHON_CLIENT_MEM_FRACTION={os.environ.get('XLA_PYTHON_CLIENT_MEM_FRACTION')} "
        f"XLA_PYTHON_CLIENT_ALLOCATOR={os.environ.get('XLA_PYTHON_CLIENT_ALLOCATOR')} "
        f"TF_GPU_ALLOCATOR={os.environ.get('TF_GPU_ALLOCATOR')}"
    )
    if jax_cache_dir is not None:
        print(f"jax compilation cache: {jax_cache_dir}")
    try:
        import gpu4pyscf

        print("gpu4pyscf:", getattr(gpu4pyscf, "__version__", "unknown"))
    except Exception as exc:
        print("gpu4pyscf import failed:", repr(exc))
        raise

    cache_config = _built_data_cache_config(
        h5_path=args.h5_path,
        n_samples=args.n_samples,
        sample_buffer=max(int(args.sample_buffer), 0),
        shuffle_samples=bool(args.shuffle_samples),
        sample_seed=args.sample_seed,
        max_atoms=args.max_atoms,
        basis=args.basis,
        ref_xc=args.ref_xc,
        grids_level=args.grids_level,
        scf_max_cycle=args.scf_max_cycle,
        scf_conv_tol=args.scf_conv_tol,
        hfx_omega_values=hfx_omega_values,
        hfx_aux=hfx_aux,
        compute_response_eri_slices=bool(args.compute_response_eri_slices),
    )
    cache_path = Path(args.built_data_cache) if args.built_data_cache else None
    h5_cache_path = Path(args.built_data_h5) if args.built_data_h5 else None
    if cache_path is not None and h5_cache_path is not None:
        raise ValueError("Use either --built-data-cache or --built-data-h5, not both.")
    cache_payload = None
    h5_cache_payload = None
    datum_store: _H5DatumStore | None = None
    fixed_density_uses_xc_residual = (
        args.training_mode == "fixed_density"
        and args.fixed_density_loss_target == "xc_residual"
    )
    strip_runtime_static_fields = (
        args.training_mode == "fixed_density" and not bool(args.keep_runtime_static_fields)
    )
    fixed_non_xc_offsets_from_cache_or_build: list[float] = []
    if cache_path is not None and cache_path.exists() and not bool(args.rebuild_built_data_cache):
        print(f"loading built-data cache: {cache_path}")
        cache_payload = _load_built_data_cache(cache_path, config=cache_config)
    if (
        h5_cache_path is not None
        and h5_cache_path.exists()
        and not bool(args.rebuild_built_data_cache)
    ):
        print(f"loading built-data HDF5 input: {h5_cache_path}")
        h5_cache_payload = _load_built_data_h5(h5_cache_path, config=cache_config)

    if h5_cache_payload is not None:
        data = list(h5_cache_payload["refs"])[: args.n_samples]
        samples = list(h5_cache_payload["samples"])
        built_samples = list(h5_cache_payload["built_samples"])[: args.n_samples]
        skipped_samples = list(h5_cache_payload["skipped_samples"])
        reference_energies = [
            float(value) for value in np.asarray(h5_cache_payload["reference_energies"])
        ][: args.n_samples]
        fixed_non_xc_offsets_from_cache_or_build = [
            float(value) for value in np.asarray(h5_cache_payload["fixed_non_xc_offsets"])
        ][: args.n_samples]
        build_elapsed = 0.0
        datum_store = _H5DatumStore(h5_cache_path)
        print(f"loaded {len(data)} built samples from HDF5 input")
    elif cache_payload is not None:
        data = list(cache_payload["data"])[: args.n_samples]
        samples = list(cache_payload["samples"])
        built_samples = list(cache_payload["built_samples"])[: args.n_samples]
        skipped_samples = list(cache_payload["skipped_samples"])
        reference_energies = [float(value) for value in cache_payload["reference_energies"]][
            : args.n_samples
        ]
        fixed_non_xc_offsets_from_cache_or_build = [
            float(value) for value in cache_payload.get("fixed_non_xc_offsets", ())
        ][: args.n_samples]
        build_elapsed = 0.0
        cached_build_elapsed = float(cache_payload["build_elapsed"])
        print(
            f"loaded {len(data)} built samples from cache "
            f"(original build {cached_build_elapsed:.2f} s)"
        )
    else:
        candidate_count = args.n_samples + max(int(args.sample_buffer), 0)
        samples = _load_samples(
            Path(args.h5_path),
            candidate_count,
            args.max_atoms,
            shuffle=bool(args.shuffle_samples),
            seed=args.sample_seed,
        )
        if len(samples) < args.n_samples:
            raise RuntimeError(
                f"Loaded {len(samples)} valid samples, expected at least {args.n_samples}."
            )

        print(
            f"loaded {len(samples)} candidate samples from {args.h5_path} "
            f"across {_sample_group_count(samples)} HDF5 groups"
        )
        for i, sample in enumerate(samples):
            print(
                f"  sample {i:02d}: {sample['name']} conf={sample['conformer_idx']} "
                f"natoms={len(sample['atomic_numbers'])} Eccsdt={sample['ccsdt_energy']:.10f}"
            )

        rks_config = RKSConfig(
            xc_spec=args.ref_xc,
            max_cycle=args.scf_max_cycle,
            conv_tol=args.scf_conv_tol,
            conv_tol_density=1e-8,
            damping=0.25,
        )
        build_t0 = time.perf_counter()
        h5_write_handle = None
        if h5_cache_path is not None:
            print(f"writing built-data HDF5 input incrementally: {h5_cache_path}")
            h5_write_handle = _create_built_data_h5(
                h5_cache_path,
                config=cache_config,
                samples=samples,
            )
        data: list[object] = []
        built_samples: list[dict[str, object]] = []
        skipped_samples: list[dict[str, object]] = []
        reference_energies: list[float] = []
        try:
            for i, sample in enumerate(samples):
                if len(data) >= args.n_samples:
                    break
                spec = _molecule_spec(sample)
                try:
                    mol = restricted_molecule_from_spec_with_gpu4pyscf_rks(
                        atom=spec,
                        basis=args.basis,
                        xc_spec=args.ref_xc,
                        unit="Bohr",
                        cart=True,
                        grids_level=args.grids_level,
                        rks_config=rks_config,
                        integral_backend="libcint",
                        compute_local_hfx_features=True,
                        compute_local_hfx_aux=hfx_aux,
                        hfx_omega_values=hfx_omega_values,
                        compute_local_pt2_features=False,
                        compute_response_eri_slices=bool(args.compute_response_eri_slices),
                    )
                except Exception as exc:
                    if bool(args.skip_unconverged_builds) and _is_skippable_build_failure(exc):
                        skipped = {
                            "candidate_index": int(i),
                            "name": str(sample["name"]),
                            "conformer_idx": int(sample["conformer_idx"]),
                            "reason": str(exc),
                        }
                        skipped_samples.append(skipped)
                        if h5_write_handle is not None:
                            _append_built_data_h5_skip(h5_write_handle, skipped)
                        print(
                            f"  skipped candidate {i + 1:02d}/{len(samples)}: "
                            f"{sample['name']} conf={sample['conformer_idx']} reason={exc}"
                        )
                        _release_build_device_memory()
                        continue
                    raise
                target_total_energy = float(sample["ccsdt_energy"])
                datum = GroundStateDatum(
                    molecule=mol,
                    target_total_energy=jnp.asarray(
                        target_total_energy,
                        dtype=jnp.float64,
                    ),
                    density_constraint_weight=0.0,
                )
                non_xc_offset = 0.0
                if fixed_density_uses_xc_residual:
                    non_xc_offset = _fixed_density_non_xc_energy(mol)
                    datum = replace(
                        datum,
                        target_total_energy=jnp.asarray(
                            target_total_energy - non_xc_offset,
                            dtype=jnp.float64,
                        ),
                    )
                    datum = _strip_runtime_static_fields_for_fixed_density(
                        [datum],
                        drop_ground_state_integrals=True,
                    )[0]
                    fixed_non_xc_offsets_from_cache_or_build.append(float(non_xc_offset))
                elif strip_runtime_static_fields:
                    datum = _strip_runtime_static_fields_for_fixed_density([datum])[0]
                datum = _hostify_for_cache(datum)
                if h5_write_handle is not None:
                    item = _append_built_data_h5_sample(
                        h5_write_handle,
                        datum=datum,
                        sample=sample,
                        reference_energy=float(mol.mf_energy),
                        fixed_non_xc_offset=float(non_xc_offset),
                    )
                else:
                    item = datum
                data.append(item)
                built_samples.append(sample)
                reference_energies.append(float(mol.mf_energy))
                print(
                    f"  built {len(data):02d}/{args.n_samples} "
                    f"(candidate {i + 1:02d}/{len(samples)}): backend={mol.runtime_scf_backend} "
                    f"ref_E={float(mol.mf_energy):.10f}"
                )
                del mol, datum
                _release_build_device_memory()
        finally:
            if h5_write_handle is not None:
                h5_write_handle.close()
        if len(data) != args.n_samples:
            raise RuntimeError(
                f"Built {len(data)} converged samples, expected {args.n_samples}; "
                f"loaded {len(samples)} candidates and skipped {len(skipped_samples)}."
            )
        build_elapsed = time.perf_counter() - build_t0
        if cache_path is not None:
            print(f"writing built-data cache: {cache_path}")
            _write_built_data_cache(
                cache_path,
                config=cache_config,
                data=data,
                samples=samples,
                built_samples=built_samples,
                skipped_samples=skipped_samples,
                reference_energies=reference_energies,
                build_elapsed=build_elapsed,
                fixed_non_xc_offsets=fixed_non_xc_offsets_from_cache_or_build,
            )
        if h5_cache_path is not None:
            datum_store = _H5DatumStore(h5_cache_path)
    target_energies = np.asarray([float(s["ccsdt_energy"]) for s in built_samples])
    reference_energies_arr = np.asarray(reference_energies, dtype=float)
    fixed_non_xc_offsets = np.zeros(len(data), dtype=float)
    if (
        fixed_density_uses_xc_residual
        and len(fixed_non_xc_offsets_from_cache_or_build) == len(data)
    ):
        fixed_non_xc_offsets = np.asarray(fixed_non_xc_offsets_from_cache_or_build, dtype=float)
        print("fixed-density training: using precomputed XC residual targets")
    elif fixed_density_uses_xc_residual:
        residual_t0 = time.perf_counter()
        data, fixed_non_xc_offsets = _to_fixed_density_xc_residual_data(data)
        residual_elapsed = time.perf_counter() - residual_t0
        print(
            "fixed-density training: using XC residual targets and stripped "
            f"ground-state ERI constants ({residual_elapsed:.2f} s)"
        )
    elif strip_runtime_static_fields:
        data = _strip_runtime_static_fields_for_fixed_density(data)
        print("fixed-density training: stripped GPU4PySCF runtime static fields from molecules")
    training_target_energies = (
        target_energies - fixed_non_xc_offsets
        if fixed_density_uses_xc_residual
        else target_energies
    )
    train_data = [data[int(idx)] for idx in train_indices]
    test_data = [data[int(idx)] for idx in test_indices]
    train_target_energies = training_target_energies[train_indices]
    test_target_energies = training_target_energies[test_indices]
    train_total_target_energies = target_energies[train_indices]
    test_total_target_energies = target_energies[test_indices]
    reference_mae_ev = _mean_absolute_error_ev(reference_energies_arr, target_energies)
    reference_train_mae_ev = _mean_absolute_error_ev(
        reference_energies_arr[train_indices],
        train_total_target_energies,
    )
    reference_test_mae_ev = _mean_absolute_error_ev(
        reference_energies_arr[test_indices],
        test_total_target_energies,
    )
    print(f"reference build time: {build_elapsed:.2f} s")
    print(f"{args.ref_xc} reference MAE vs CCSD(T): {reference_mae_ev:.6f} eV")
    print(
        f"split: train={len(train_data)} test={len(test_data)} "
        f"shuffle={bool(args.shuffle_split)} seed={args.split_seed}"
    )

    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    functional = _build_functional(
        hidden_dims=hidden_dims,
        architecture=args.architecture,
        hf_input_mode=args.hf_input_mode,
        hfx_channels=len(hfx_omega_values),
        sigmoid_scale_factor=args.sigmoid_scale_factor,
    )
    training_config = _build_training_config(
        training_mode=args.training_mode,
        energy_normalization=args.energy_normalization,
        energy_loss=args.energy_loss,
        train_scf_max_cycle=args.train_scf_max_cycle,
        train_scf_damping=args.train_scf_damping,
        implicit_response_backend=args.implicit_response_backend,
        implicit_diff_max_iter=args.implicit_diff_max_iter,
    )
    training_predictor = (
        _make_fixed_density_xc_predictor(functional)
        if fixed_density_uses_xc_residual
        else None
    )
    train_batches = _make_batches(
        train_data,
        int(args.batch_size),
        bucket_by_shape=bool(args.bucket_by_shape),
    )
    shape_bucket_count = len({_pytree_shape_signature(datum) for datum in train_data})
    steps_per_epoch = _optimizer_steps_per_epoch(args.update_mode, len(train_batches))
    sample_train_datum = _materialize_datum(train_data[0], datum_store)
    state = create_train_state_from_molecule(
        functional,
        jax.random.PRNGKey(args.seed),
        sample_train_datum.molecule,
        _build_optimizer(
            learning_rate=args.learning_rate,
            gradient_clip_norm=args.gradient_clip_norm,
            lr_decay_epochs=args.lr_decay_epochs,
            lr_decay_factor=args.lr_decay_factor,
            steps_per_epoch=steps_per_epoch,
        ),
    )
    train_step = make_ground_state_train_step(
        functional,
        training_config=training_config,
        predictor=training_predictor,
    )
    loss_and_grad = make_ground_state_loss_and_grad(
        functional,
        training_config=training_config,
        predictor=training_predictor,
    )
    jit_train_step = bool(args.jit_train_step) and args.training_mode == "fixed_density"
    if bool(args.jit_train_step) and not jit_train_step:
        print("--jit-train-step is only enabled for fixed-density mode; using eager train step")
    if jit_train_step:
        train_step = jax.jit(train_step)
        loss_and_grad = jax.jit(loss_and_grad)

    print(f"training mode: {args.training_mode}")
    print(
        f"training batches: {len(train_batches)} "
        f"batch_size={args.batch_size if int(args.batch_size) > 0 else len(train_data)} "
        f"update_mode={args.update_mode} "
        f"bucket_by_shape={bool(args.bucket_by_shape)} shape_buckets={shape_bucket_count} "
        f"steps_per_epoch={steps_per_epoch} jit_train_step={jit_train_step}"
    )
    if int(args.lr_decay_epochs) > 0 and float(args.lr_decay_factor) != 1.0:
        print(
            "learning-rate schedule: "
            f"initial={args.learning_rate:g} factor={args.lr_decay_factor:g} "
            f"every={args.lr_decay_epochs} epochs "
            f"({int(args.lr_decay_epochs) * steps_per_epoch} optimizer steps)"
        )
    else:
        print(f"learning-rate schedule: constant {args.learning_rate:g}")
    print("training start")
    epoch_rows: list[dict[str, float]] = []
    train_t0 = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_t0 = time.perf_counter()
        batch_rows: list[dict[str, float]] = []
        batch_sizes: list[float] = []
        epoch_train_batches = _batches_for_epoch(
            train_batches,
            epoch=epoch,
            shuffle=bool(args.shuffle_train_batches) and args.update_mode == "per_batch",
            seed=int(args.train_batch_seed),
        )
        if args.update_mode == "epoch_accum":
            epoch_params = state.params
            grad_sum = None
            total_loss_weight = 0.0
            for batch in epoch_train_batches:
                batch_data = _materialize_batch(batch, datum_store)
                loss, metrics, grads = loss_and_grad(epoch_params, batch_data)
                jax.block_until_ready(loss)
                batch_loss = float(jax.device_get(loss))
                batch_weight = _batch_loss_weight(batch_data)
                grad_sum = _accumulate_weighted_grads(
                    grad_sum,
                    grads,
                    weight=batch_weight,
                )
                total_loss_weight += batch_weight
                batch_rows.append(
                    _epoch_monitor_row(
                        epoch=epoch,
                        epoch_elapsed=0.0,
                        loss=batch_loss,
                        metrics=metrics,
                    )
                )
                batch_sizes.append(batch_weight)
            mean_grads = _normalize_accumulated_grads(
                grad_sum,
                total_weight=total_loss_weight,
            )
            new_state = state.apply_gradients(grads=mean_grads)
            param_delta = jax.tree_util.tree_map(
                lambda new, old: new - old,
                new_state.params,
                state.params,
            )
            grad_norm = _tree_l2_norm(mean_grads, sanitize=True)
            grad_abs_max = _tree_abs_max(mean_grads, sanitize=True)
            param_update_norm = _tree_l2_norm(param_delta, sanitize=True)
            param_norm = _tree_l2_norm(state.params, sanitize=True)
            state = new_state
        else:
            grad_norm = None
            grad_abs_max = None
            param_update_norm = None
            param_norm = None
            for batch in epoch_train_batches:
                batch_data = _materialize_batch(batch, datum_store)
                state, metrics = train_step(state, batch_data)
                jax.block_until_ready(metrics["loss"])
                batch_loss = float(jax.device_get(metrics["loss"]))
                batch_weight = _batch_loss_weight(batch_data)
                batch_rows.append(
                    _epoch_monitor_row(
                        epoch=epoch,
                        epoch_elapsed=0.0,
                        loss=batch_loss,
                        metrics=metrics,
                    )
                )
                batch_sizes.append(batch_weight)
        epoch_elapsed = time.perf_counter() - epoch_t0
        row = _aggregate_epoch_rows(
            epoch=epoch,
            epoch_elapsed=epoch_elapsed,
            batch_rows=batch_rows,
            batch_sizes=batch_sizes,
        )
        if args.update_mode == "epoch_accum":
            row["grad_norm"] = float(jax.device_get(grad_norm))
            row["grad_abs_max"] = float(jax.device_get(grad_abs_max))
            row["raw_grad_norm"] = row["grad_norm"]
            row["param_update_norm"] = float(jax.device_get(param_update_norm))
            row["param_norm"] = float(jax.device_get(param_norm))
        should_eval = (
            (bool(args.eval_first_epoch) and epoch == 1)
            or epoch == args.epochs
            or (eval_every > 0 and epoch % eval_every == 0)
        )
        if should_eval:
            row["eval_train_mae_ev"] = _evaluate_energy_mae_ev(
                state.params,
                functional,
                train_data,
                training_config,
                train_target_energies,
                predictor=training_predictor,
                store=datum_store,
            )
            row["eval_test_mae_ev"] = _evaluate_energy_mae_ev(
                state.params,
                functional,
                test_data,
                training_config,
                test_target_energies,
                predictor=training_predictor,
                store=datum_store,
            )
        epoch_rows.append(row)
        if epoch == 1 or epoch == args.epochs or epoch % args.log_every == 0:
            print(_format_epoch_log(epoch=epoch, total_epochs=args.epochs, row=row), flush=True)

    train_elapsed = time.perf_counter() - train_t0
    predicted_training_target = np.asarray(
        _predict_final_energies(
            state.params,
            functional,
            data,
            training_config,
            predictor=training_predictor,
            store=datum_store,
        ),
        dtype=float,
    )
    predicted = (
        predicted_training_target + fixed_non_xc_offsets
        if fixed_density_uses_xc_residual
        else predicted_training_target
    )
    final_params_path = outdir / "final_params.pkl"
    with open(final_params_path, "wb") as f:
        pickle.dump(jax.device_get(state.params), f, protocol=pickle.HIGHEST_PROTOCOL)
    final_mae_ev = _mean_absolute_error_ev(predicted, target_energies)
    final_train_mae_ev = _evaluate_energy_mae_ev(
        state.params,
        functional,
        train_data,
        training_config,
        train_target_energies,
        predictor=training_predictor,
        store=datum_store,
    )
    final_test_mae_ev = _evaluate_energy_mae_ev(
        state.params,
        functional,
        test_data,
        training_config,
        test_target_energies,
        predictor=training_predictor,
        store=datum_store,
    )
    result = {
        "n_samples": len(data),
        "candidate_samples": len(samples),
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "sample_buffer": int(args.sample_buffer),
        "sample_selection": "round_robin_groups_v1",
        "candidate_group_count": int(_sample_group_count(samples)),
        "built_group_count": int(_sample_group_count(built_samples)),
        "shuffle_samples": bool(args.shuffle_samples),
        "sample_seed": int(args.sample_seed),
        "skip_unconverged_builds": bool(args.skip_unconverged_builds),
        "skipped_samples": skipped_samples,
        "shuffle_split": bool(args.shuffle_split),
        "split_seed": args.split_seed,
        "train_indices": [int(idx) for idx in train_indices],
        "test_indices": [int(idx) for idx in test_indices],
        "basis": args.basis,
        "ref_xc": args.ref_xc,
        "epochs": args.epochs,
        "eval_first_epoch": bool(args.eval_first_epoch),
        "batch_size": int(args.batch_size),
        "update_mode": args.update_mode,
        "optimizer_steps_per_epoch": int(steps_per_epoch),
        "optimizer_steps": int(args.epochs) * int(steps_per_epoch),
        "bucket_by_shape": bool(args.bucket_by_shape),
        "shuffle_train_batches": bool(args.shuffle_train_batches),
        "train_batch_seed": int(args.train_batch_seed),
        "shape_bucket_count": int(shape_bucket_count),
        "num_train_batches": len(train_batches),
        "jit_train_step": bool(jit_train_step),
        "strip_runtime_static_fields": bool(strip_runtime_static_fields),
        "built_data_cache": str(cache_path) if cache_path is not None else None,
        "built_data_h5": str(h5_cache_path) if h5_cache_path is not None else None,
        "disk_backed_training_data": bool(datum_store is not None),
        "jax_compilation_cache_dir": jax_cache_dir,
        "final_params_path": str(final_params_path),
        "learning_rate": args.learning_rate,
        "lr_decay_epochs": int(args.lr_decay_epochs),
        "lr_decay_factor": float(args.lr_decay_factor),
        "lr_decay_steps": (
            int(args.lr_decay_epochs) * int(steps_per_epoch)
            if int(args.lr_decay_epochs) > 0 and float(args.lr_decay_factor) != 1.0
            else 0
        ),
        "training_mode": args.training_mode,
        "fixed_density_loss_target": args.fixed_density_loss_target,
        "fixed_density_uses_xc_residual": bool(fixed_density_uses_xc_residual),
        "energy_loss": args.energy_loss,
        "loss_type": (
            "energy_mse"
            if args.energy_loss == "mse"
            else "energy_mae"
            if args.energy_loss == "mae"
            else "energy_mae_plus_mse"
        ),
        "energy_mse_weight": float(training_config.energy_mse_weight),
        "energy_mae_weight": float(training_config.energy_mae_weight),
        "energy_normalization": args.energy_normalization,
        "gradient_clip_norm": args.gradient_clip_norm,
        "implicit_response_backend": training_config.implicit_response_backend,
        "implicit_diff_max_iter": training_config.scf_implicit_diff_max_iter,
        "hidden_dims": list(hidden_dims),
        "architecture": args.architecture,
        "coefficient_transform": "scale_sigmoid_then_sanitize",
        "sigmoid_scale_factor": float(args.sigmoid_scale_factor),
        "hf_input_mode": args.hf_input_mode,
        "hfx_omega_values": list(hfx_omega_values),
        "hfx_aux": hfx_aux,
        "compute_response_eri_slices": bool(args.compute_response_eri_slices),
        "grids_level": args.grids_level,
        "build_time_s": build_elapsed,
        "train_time_s": train_elapsed,
        "seconds_per_epoch": train_elapsed / max(args.epochs, 1),
        "seconds_per_epoch_per_molecule": train_elapsed / max(args.epochs * len(train_data), 1),
        "reference_mae_ev": reference_mae_ev,
        "reference_train_mae_ev": reference_train_mae_ev,
        "reference_test_mae_ev": reference_test_mae_ev,
        "final_mae_ev": final_mae_ev,
        "final_train_mae_ev": final_train_mae_ev,
        "final_test_mae_ev": final_test_mae_ev,
        "sample_names": [str(sample["name"]) for sample in built_samples],
        "sample_conformer_indices": [int(sample["conformer_idx"]) for sample in built_samples],
        "target_total_energies": [float(value) for value in target_energies],
        "reference_total_energies": [float(value) for value in reference_energies_arr],
        "predicted_total_energies": [float(value) for value in predicted],
        "training_target_energies": [float(value) for value in training_target_energies],
        "predicted_training_target_energies": [
            float(value) for value in predicted_training_target
        ],
        "train_target_total_energies": [
            float(target_energies[int(idx)]) for idx in train_indices
        ],
        "train_reference_total_energies": [
            float(reference_energies_arr[int(idx)]) for idx in train_indices
        ],
        "train_predicted_total_energies": [
            float(predicted[int(idx)]) for idx in train_indices
        ],
        "test_target_total_energies": [
            float(target_energies[int(idx)]) for idx in test_indices
        ],
        "test_reference_total_energies": [
            float(reference_energies_arr[int(idx)]) for idx in test_indices
        ],
        "test_predicted_total_energies": [
            float(predicted[int(idx)]) for idx in test_indices
        ],
        "epoch_rows": epoch_rows,
    }
    print("training done")
    print(json.dumps({k: v for k, v in result.items() if k != "epoch_rows"}, indent=2))
    with open(outdir / "gpu4pyscf_10mol_100epoch_results.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {outdir / 'gpu4pyscf_10mol_100epoch_results.json'}")


if __name__ == "__main__":
    main()
