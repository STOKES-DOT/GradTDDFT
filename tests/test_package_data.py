import tomllib
from pathlib import Path


def test_pyscf_basis_snapshot_is_included_as_package_data():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    package_data = pyproject["tool"]["setuptools"]["package-data"]
    data_patterns = package_data["td_graddft.data"]

    assert "pyscf_basis_snapshot/**/*" in data_patterns


def test_joltqc_port_cuda_sources_are_included_as_package_data():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    package_data = pyproject["tool"]["setuptools"]["package-data"]
    scf_patterns = package_data["td_graddft.scf"]

    assert "joltqc_port/*.cu" in scf_patterns
    assert "joltqc_port/*.py" in scf_patterns
    assert "joltqc_port/*.json" in scf_patterns


def test_joltqc_port_includes_full_rys_root_sources():
    source_dir = Path("src/td_graddft/scf/joltqc_port")

    for nroot in range(1, 10):
        assert (source_dir / f"rys_root{nroot}.cu").exists()
    assert (source_dir / "1q1t.cu").exists()
    assert (source_dir / "optimal_scheme_fp64.json").exists()


def test_joltqc_port_util_is_self_contained():
    from td_graddft.scf.joltqc_port import util

    table = util.generate_lookup_table(1, 0, 0, 0)

    assert "i_idx" in table


def test_joltqc_port_builds_static_1qnt_source_without_cupy():
    from td_graddft.scf.joltqc_port.codegen import build_1qnt_source

    source = build_1qnt_source((1, 0, 0, 0), (3, 3, 3, 3))

    assert "constexpr int li = 1;" in source
    assert "constexpr int npi = 3;" in source
    assert "constexpr __device__ uint32_t i_idx" in source
    assert "void rys_1qnt_vjk" in source
    assert "RawModule" not in source


def test_joltqc_port_builds_basis_specific_1qnt_dispatch_source():
    import numpy as np

    from td_graddft.scf.joltqc_port.codegen import build_1qnt_dispatch_source

    source = build_1qnt_dispatch_source(
        np.asarray([[0, 3], [2, 1]], dtype=np.int32),
        np.asarray([[0, 0, 0, 0], [1, 1, 1, 1]], dtype=np.int32),
        np.asarray([0, 1, 2], dtype=np.int32),
    )

    assert "TdGraddftLaunchJoltQC1qnt" in source
    assert "tdg_joltqc_1q1t_l_0_0_0_0_p_3_3_3_3" in source
    assert "tdg_joltqc_1qnt_l_2_2_2_2_p_1_1_1_1" in source
    assert "reinterpret_cast<const int4*>" in source
    assert "cudaErrorNotSupported" in source
    assert "cudaMemcpyAsync" in source
    assert "host_group_quartet_keys[" in source
    assert "host_group_quartet_offsets[" in source
    assert "TdGraddftJoltQCSignatureSupported" in source
    assert "const int start = 0;" not in source
    assert "const int stop = 1;" not in source


def test_joltqc_port_dispatch_uses_joltqc_1q1t_for_ssss():
    import numpy as np

    from td_graddft.scf.joltqc_port.codegen import build_1qnt_dispatch_source

    source = build_1qnt_dispatch_source(
        np.asarray([[0, 3]], dtype=np.int32),
        np.asarray([[0, 0, 0, 0]], dtype=np.int32),
        np.asarray([0, 1], dtype=np.int32),
    )

    assert "tdg_joltqc_1q1t_l_0_0_0_0_p_3_3_3_3" in source
    assert "tdg_joltqc_1qnt_l_0_0_0_0_p_3_3_3_3" not in source
    assert "dim3 block(256, 1, 1)" in source
    assert "host_group_quartet_offsets[index]" in source


def test_joltqc_port_dispatch_specializes_kernels_by_primitive_counts():
    import numpy as np

    from td_graddft.scf.joltqc_port.codegen import build_1qnt_dispatch_source

    source = build_1qnt_dispatch_source(
        np.asarray([[1, 1], [1, 3]], dtype=np.int32),
        np.asarray([[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 1, 0]], dtype=np.int32),
        np.asarray([0, 2, 5, 7], dtype=np.int32),
    )

    assert "constexpr int npi = 1;" in source
    assert "constexpr int npi = 3;" in source
    assert "constexpr int npj = 1;" in source
    assert "constexpr int npj = 3;" in source
    assert source.count("constexpr int li = 1;") >= 3


def test_joltqc_port_split_dispatch_signature_units_are_independent_of_group_offsets():
    import numpy as np

    from td_graddft.scf.joltqc_port.codegen import build_1qnt_dispatch_source_units

    group_keys = np.asarray([[0, 3], [1, 3]], dtype=np.int32)
    group_quartet_keys = np.asarray([[0, 0, 0, 0], [1, 0, 1, 0]], dtype=np.int32)
    reordered_quartet_keys = np.asarray([[1, 0, 1, 0], [0, 0, 0, 0]], dtype=np.int32)

    units = build_1qnt_dispatch_source_units(
        group_keys,
        group_quartet_keys,
        np.asarray([0, 1, 2], dtype=np.int32),
    )
    same_signature_units = build_1qnt_dispatch_source_units(
        group_keys,
        reordered_quartet_keys,
        np.asarray([0, 3, 4], dtype=np.int32),
    )

    signature_units = sorted((name, source) for name, source in units if name.startswith("signature_"))
    same_signatures = sorted(
        (name, source) for name, source in same_signature_units if name.startswith("signature_")
    )
    dispatch_source = dict(units)["dispatch.cu"]
    same_dispatch_source = dict(same_signature_units)["dispatch.cu"]

    assert signature_units == same_signatures
    assert dispatch_source == same_dispatch_source
    assert "const int start = 0;" not in "\n".join(source for _, source in signature_units)
    assert "TdGraddftLaunchJoltQC1qntSignature_l_0_0_0_0_p_3_3_3_3" in dispatch_source
    assert "cudaMemcpyAsync" in dispatch_source
    assert "host_group_quartet_offsets[index]" in dispatch_source
    assert "shell_quartets,\n            0,\n            1" not in dispatch_source
    assert "const int start = 0;" not in dispatch_source


def test_joltqc_port_dispatch_source_key_is_independent_of_runtime_quartet_mapping():
    import numpy as np

    from td_graddft.scf.joltqc_port.codegen import build_1qnt_dispatch_source_key

    group_keys = np.asarray([[0, 3], [1, 3]], dtype=np.int32)
    group_quartet_keys = np.asarray([[0, 0, 0, 0], [1, 0, 1, 0]], dtype=np.int32)
    reordered_quartet_keys = np.asarray([[1, 0, 1, 0], [0, 0, 0, 0]], dtype=np.int32)

    key = build_1qnt_dispatch_source_key(
        group_keys,
        group_quartet_keys,
        np.asarray([0, 1, 2], dtype=np.int32),
    )
    reordered_key = build_1qnt_dispatch_source_key(
        group_keys,
        reordered_quartet_keys,
        np.asarray([0, 3, 4], dtype=np.int32),
    )

    assert key == reordered_key


def test_joltqc_port_fixed_dispatch_arrays_cover_runtime_signature_universe():
    import numpy as np

    from td_graddft.scf.joltqc_port.codegen import (
        build_fixed_1qnt_dispatch_arrays,
        build_1qnt_dispatch_source_key,
    )

    group_keys, group_quartet_keys, group_quartet_offsets = build_fixed_1qnt_dispatch_arrays(
        max_l=1,
        nprim_max=2,
    )

    assert group_keys.tolist() == [[0, 2], [0, 1], [1, 2], [1, 1]]
    assert group_quartet_keys.shape == (55, 4)
    assert group_quartet_offsets.shape == (56,)
    assert np.array_equal(group_quartet_offsets, np.arange(56, dtype=np.int32))
    assert isinstance(
        build_1qnt_dispatch_source_key(group_keys, group_quartet_keys, group_quartet_offsets),
        str,
    )


def test_joltqc_port_split_dispatch_specializes_kernels_by_primitive_counts():
    import numpy as np

    from td_graddft.scf.joltqc_port.codegen import build_1qnt_dispatch_source_units

    units = build_1qnt_dispatch_source_units(
        np.asarray([[1, 1], [1, 3]], dtype=np.int32),
        np.asarray([[0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 1, 0]], dtype=np.int32),
        np.asarray([0, 2, 5, 7], dtype=np.int32),
    )
    generated_sources = "\n".join(source for name, source in units if name != "dispatch.cu")

    assert "constexpr int npi = 1;" in generated_sources
    assert "constexpr int npi = 3;" in generated_sources
    assert "constexpr int npj = 1;" in generated_sources
    assert "constexpr int npj = 3;" in generated_sources
    assert len([name for name, _ in units if name.startswith("signature_")]) >= 3
