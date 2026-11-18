"""End-to-end integration tests for pipeline, JSON telemetry export, CLI, and 7-bit ASCII purity."""

from __future__ import annotations

import json
from pathlib import Path

from cova_kinetix.cli import main
from cova_kinetix.pipeline import run_full_characterization
from cova_kinetix.visualizer import (
    assert_ascii,
    render_covalent_kinetics,
    render_eht_summary,
    render_metadynamics_fes,
    render_permeation_profile,
    render_pocket_slice_contour,
)


def test_full_pipeline_and_json_export(tmp_path: Path) -> None:
    json_out = tmp_path / "summary.json"
    res = run_full_characterization(
        smiles="C=CC(=O)Nc1ccc2ncnc(Nc3ccc(F)c(Cl)c3)c2c1OC",
        target_name="EGFR_C797S",
        ligand_name="Afatinib_analog",
        metad_steps=800,
        n_metad_trajectories=5,
        seed=42,
        export_json_path=json_out,
    )
    assert json_out.exists()
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["warhead_class"] == "acrylamide"
    assert payload["covalent_kinetics"]["k_inact_over_ki"] > 0.0

    # Strict 7-bit ASCII validation on every visual panel
    for text in (
        render_eht_summary(res.molecule, res.eht),
        render_pocket_slice_contour(res.pb_grid, res.docking),
        render_permeation_profile(res.permeation),
        render_covalent_kinetics(res.kinetics),
        render_metadynamics_fes(res.metadynamics),
    ):
        assert_ascii(text)


def test_all_cli_subcommands_exit_zero(capsys: object, tmp_path: Path) -> None:
    json_file = tmp_path / "demo.json"
    commands = [
        ["demo", "--metad-steps", "500", "--export-json", str(json_file)],
        ["quantum", "--smiles", "C=CC(=O)Nc1ccccc1"],
        ["screen", "--target", "BTK_C481", "--top-k", "4"],
        ["permeate", "--smiles", "C=CC(=O)Nc1ccc(NCc2ccccc2)cc1", "--ph", "7.4"],
        ["metad", "--steps", "500", "--n-trajectories", "4"],
        ["evolve", "--target", "EGFR_C797S", "--generations", "1", "--pop-size", "6"],
        ["calibrate", "--n-bootstrap", "8", "--n-sobol", "16"],
        ["benchmark", "--target", "EGFR_C797S", "--json-out"],
    ]
    for argv in commands:
        rc = main(argv)
        assert rc == 0
    assert json_file.exists()


def test_assert_ascii_rejects_non_ascii() -> None:
    import pytest

    assert assert_ascii("pure-ascii_123!@#") == "pure-ascii_123!@#"
    with pytest.raises(ValueError):
        assert_ascii("delta \u0394G")


def test_benchmark_on_mpro_c145_target() -> None:
    rc = main(["benchmark", "--target", "MPRO_C145"])
    assert rc == 0
