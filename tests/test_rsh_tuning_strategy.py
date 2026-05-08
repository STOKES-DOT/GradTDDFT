from argparse import Namespace
import importlib.util
import numpy as np
from pathlib import Path
import sys


_TOOL_PATH = Path(__file__).resolve().parents[1] / "tools" / "tune_water_rsh_endpoint_koopmans.py"
_SPEC = importlib.util.spec_from_file_location("tune_water_rsh_endpoint_koopmans", _TOOL_PATH)
assert _SPEC is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
_build_stage_specs = _MODULE._build_stage_specs
_rsh_template_from_args = _MODULE._rsh_template_from_args
_initial_resolved_from_args = _MODULE._initial_resolved_from_args
_local_xc_spec_from_args = _MODULE._local_xc_spec_from_args
_coordinate_active_raw_dims_from_template = _MODULE._coordinate_active_raw_dims_from_template
parse_args = _MODULE.parse_args


def test_two_dimensional_tuning_strategy_builds_koopmans_then_curvature_stages():
    args = Namespace(
        strategy="2dt",
        steps=7,
        stage_a_steps=None,
        stage_b_steps=None,
        janak_weight=0.0,
        fractional_weight=4.0,
        koopmans_ip_weight=1.0,
        koopmans_ea_weight=1.0,
        koopmans_lumo_ea_weight=0.0,
        koopmans_loss_kind="absolute",
        long_range_correction_weight=2.5,
    )

    stages = _build_stage_specs(args)

    assert [stage.name for stage in stages] == ["koopmans_lc", "curvature_selection"]
    assert [stage.steps for stage in stages] == [3, 4]
    assert stages[0].fractional_weight == 0.0
    assert stages[0].long_range_correction_weight == 2.5
    assert stages[0].koopmans_ea_weight == 1.0
    assert stages[0].koopmans_lumo_ea_weight == 0.0
    assert stages[0].koopmans_loss_kind == "absolute"
    assert stages[1].fractional_weight == 4.0
    assert stages[1].long_range_correction_weight == 2.5


def test_single_tuning_strategy_uses_user_supplied_weights_in_one_stage():
    args = Namespace(
        strategy="single",
        steps=5,
        stage_a_steps=None,
        stage_b_steps=None,
        janak_weight=0.2,
        fractional_weight=0.3,
        koopmans_ip_weight=0.4,
        koopmans_ea_weight=0.5,
        koopmans_lumo_ea_weight=0.6,
        koopmans_loss_kind="squared",
        long_range_correction_weight=0.7,
    )

    stages = _build_stage_specs(args)

    assert [stage.name for stage in stages] == ["single"]
    assert stages[0].steps == 5
    assert stages[0].janak_weight == 0.2
    assert stages[0].fractional_weight == 0.3
    assert stages[0].koopmans_ip_weight == 0.4
    assert stages[0].koopmans_ea_weight == 0.5
    assert stages[0].koopmans_lumo_ea_weight == 0.6
    assert stages[0].koopmans_loss_kind == "squared"
    assert stages[0].long_range_correction_weight == 0.7


def test_water_tuning_defaults_use_neutral_homo_lumo_ip_ea_loss():
    args = parse_args([])
    stages = _build_stage_specs(args)

    assert args.koopmans_ip_weight == 1.0
    assert args.koopmans_ea_weight == 0.0
    assert args.koopmans_lumo_ea_weight == 1.0
    assert args.koopmans_loss_kind == "absolute"
    assert stages[0].koopmans_ip_weight == 1.0
    assert stages[0].koopmans_ea_weight == 0.0
    assert stages[0].koopmans_lumo_ea_weight == 1.0
    assert stages[0].koopmans_loss_kind == "absolute"


def test_water_tuning_can_request_optdftw_style_j_squared_loss():
    args = parse_args(
        [
            "--koopmans-ea-weight",
            "1",
            "--koopmans-lumo-ea-weight",
            "0",
            "--koopmans-loss-kind",
            "squared",
        ]
    )
    stages = _build_stage_specs(args)

    assert args.koopmans_ip_weight == 1.0
    assert args.koopmans_ea_weight == 1.0
    assert args.koopmans_lumo_ea_weight == 0.0
    assert args.koopmans_loss_kind == "squared"
    assert stages[0].koopmans_ea_weight == 1.0
    assert stages[0].koopmans_lumo_ea_weight == 0.0
    assert stages[0].koopmans_loss_kind == "squared"


def test_lc_wpbe_preset_builds_locked_template_and_initial_parameters():
    args = Namespace(rsh_preset="LC_WPBE")

    template = _rsh_template_from_args(args)
    initial = _initial_resolved_from_args(args)

    assert template is not None
    assert template.name == "lc-wpbe"
    assert template.sr_hf_bounds == (0.0, 0.0)
    assert template.lr_hf_bounds == (1.0, 1.0)
    assert template.default_omega == 0.4
    assert initial is not None
    assert float(initial.sr_hf_fraction) == 0.0
    assert float(initial.lr_hf_fraction) == 1.0
    assert np.isclose(float(initial.omega), 0.4)


def test_lc_wpbe_preset_selects_strict_local_spec_for_training_by_default():
    args = Namespace(rsh_preset="LC_WPBE", xc="pbe")

    assert _local_xc_spec_from_args(args) == "lc_wpbe_local"


def test_lc_wpbe_optxc_omega_source_uses_recommended_tuning_range():
    args = Namespace(rsh_preset="LC_WPBE", rsh_omega_source="optxc")

    template = _rsh_template_from_args(args)
    initial = _initial_resolved_from_args(args)

    assert template is not None
    assert template.omega_bounds == (0.13, 0.30)
    assert np.isclose(template.default_omega, 0.205)
    assert initial is not None
    assert np.isclose(float(initial.omega), 0.205)


def test_lc_wpbe_coordinate_tuning_only_moves_omega_raw_dimension():
    args = Namespace(rsh_preset="LC_WPBE", rsh_omega_source="optxc")

    template = _rsh_template_from_args(args)

    assert _coordinate_active_raw_dims_from_template(template) == (2,)
    assert _coordinate_active_raw_dims_from_template(None) == (0, 1, 2)
