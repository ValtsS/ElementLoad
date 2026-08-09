import os
import subprocess
import sys

import pytest

from elementload.cli import main, _ascii_chart, render_report
from elementload.report import build_model, analyze
from elementload.stm import read_stm, element_tube_extents


def test_ascii_chart_renders_rows():
    rows = _ascii_chart([(0.0, 0.0), (1.0, 5.0), (2.0, 10.0)])
    assert len(rows) == 11
    assert all(len(r) == 40 for r in rows)
    assert any("#" in r for r in rows)


def test_render_report_contains_sections():
    tubes = [(0.030, 0.002, 0.6), (0.025, 0.002, 0.6), (0.020, 0.002, 0.6), (0.016, 0.0015, 0.6)]
    model = build_model(tubes, overlap=0.05, joint="rigid",
                        supports=[("center", "fixed")], wind_speed=40.0)
    result = analyze(model)
    text = render_report(model, result, 40.0)
    assert "Survival wind speed per failure mode" in text
    assert "LIMITING MODE" in text
    assert "max deflection" in text
    assert "yield" in text and "buckling" in text and "pin_shear" in text


def test_render_report_tube_table_matches_layout_materialize():
    from elementload.assembly import TubeExtent
    from elementload.cli import _materialize_table, _tubes_table
    tubes = [(0.030, 0.002, 0.6), (0.025, 0.002, 0.6), (0.020, 0.002, 0.6), (0.016, 0.0015, 0.6)]
    model = build_model(tubes, overlap=0.05, joint="rigid",
                        supports=[("center", "fixed")], wind_speed=40.0)
    result = analyze(model)
    text = render_report(model, result, 40.0)
    assert "tubes (centre -> tip):" in text
    assert "Dxwall" in text and "exposed" in text and "full" in text
    r = {"extents": [(t.outer_d, t.wall, t.inner_start, t.inner_end)
                     for t in model.geometry_extents]}
    assert _materialize_table(r) == _tubes_table(r["extents"])


def test_build_model_geometry_extents_match_telescoping():
    from elementload.assembly import extents_from_specs
    tubes = [(0.030, 0.002, 0.6), (0.025, 0.002, 0.6), (0.020, 0.002, 0.6), (0.016, 0.0015, 0.6)]
    model = build_model(tubes, overlap=0.05, joint="rigid",
                        supports=[("center", "fixed")], wind_speed=40.0)
    expected = extents_from_specs(tubes, 0.05)
    assert len(model.geometry_extents) == len(expected)
    for got, exp in zip(model.geometry_extents, expected):
        assert got.outer_d == exp.outer_d
        assert got.wall == exp.wall
        assert abs(got.inner_start - exp.inner_start) < 1e-12
        assert abs(got.inner_end - exp.inner_end) < 1e-12


def test_cli_subprocess_acceptance():
    code = ("from elementload.cli import main; "
            "main(['--tube','0.030,0.002,0.6',"
            "'--tube','0.025,0.002,0.6',"
            "'--tube','0.020,0.002,0.6',"
            "'--tube','0.016,0.0015,0.6',"
            "'--overlap','0.05',"
            "'--joint','rigid',"
            "'--support','center:fixed',"
            "'--wind','40',"
            "'--damping','0.02',"
            "'--height','15',"
            "'--z0','0.05'])")
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "Element load analysis" in result.stdout
    assert "Survival wind speed per failure mode" in result.stdout
    assert "LIMITING MODE" in result.stdout
    assert "m/s" in result.stdout
    assert "N.m" in result.stdout
    assert "MPa" in result.stdout
    assert "Dynamics" in result.stdout
    assert "damping zeta = 0.020" in result.stdout
    assert "lock-in at V_crit" in result.stdout
    assert "Scruton" in result.stdout
    assert "crosswind response" in result.stdout
    assert "y_max" in result.stdout
    assert "stress range" in result.stdout
    assert "lock-in window" in result.stdout
    assert "Along-wind dynamic factor" in result.stdout
    assert "reference height 15 m" in result.stdout
    assert "c_s*c_d" in result.stdout
    assert "delta_s" in result.stdout
    assert "Fatigue" in result.stdout
    assert "Miner" in result.stdout


def test_cli_invalid_tube_arg():
    with pytest.raises(SystemExit):
        main(["--tube", "0.030,0.002", "--overlap", "0.05", "--joint", "rigid",
              "--support", "center:fixed"])


def test_cli_stm_acceptance():
    code = ("from elementload.cli import main; "
            "main(['--stm', 'data/mechanical2.stm', '--element', '15m RE'])")
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "YagiMechanical file" in result.stdout
    assert "15m RE" in result.stdout
    assert "20.28 MHz" in result.stdout
    assert "Boom (B0)" in result.stdout
    assert "skipped" in result.stdout
    assert "Overlap per joint: 100, 100, 130 mm" in result.stdout
    assert "Survival wind speed per failure mode" in result.stdout
    assert "Material: al6060-t6" in result.stdout
    assert "Fatigue (Miner's rule, R = -1, S-N al6060-t6):" in result.stdout
    assert "combined life" in result.stdout


def test_cli_stm_support_distance_applied():
    code = ("from elementload.cli import main; "
            "main(['--stm', 'data/a1.stm', '--element', '40m RE'])")
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert "Material: al6060-t66" in result.stdout
    # ES = 216.53543 in = 5.5 m; a support is added on each side of centre
    assert "Supports: center:fixed, 5.5:pin, -5.5:pin" in result.stdout
    # the added truss supports must show up in the survival wind speed
    assert "yield:" in result.stdout


def test_cli_stm_es_sag_model():
    # --es-model sag treats the ES supports as tension-only rope: they carry
    # self-weight but not wind, so the survival wind speed drops vs the pin model
    code = ("from elementload.cli import main; "
            "main(['--stm', 'data/a1.stm', '--element', '40m RE'])")
    result_pin = subprocess.run([sys.executable, "-c", code],
                                capture_output=True, text=True, timeout=120)
    assert result_pin.returncode == 0, result_pin.stderr
    code_sag = ("from elementload.cli import main; "
                "main(['--stm', 'data/a1.stm', '--element', '40m RE', '--es-model', 'sag'])")
    result_sag = subprocess.run([sys.executable, "-c", code_sag],
                                capture_output=True, text=True, timeout=120)
    assert result_sag.returncode == 0, result_sag.stderr
    assert "Supports: center:fixed, 5.5:sag, -5.5:sag" in result_sag.stdout
    v_pin = float([l for l in result_pin.stdout.splitlines() if l.strip().startswith("yield:")][0].split("m/s")[0].split()[-1])
    v_sag = float([l for l in result_sag.stdout.splitlines() if l.strip().startswith("yield:")][0].split("m/s")[0].split()[-1])
    assert v_sag < v_pin


def test_cli_stm_es_model_both_default():
    # the default --es-model both reports rigid-pin and sag results side by side
    code = ("from elementload.cli import main; "
            "main(['--stm', 'data/a1.stm', '--element', '40m RE'])")
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "ES support model comparison" in out
    assert "rigid pin" in out and "sag rope" in out
    # both survival wind speeds are reported
    assert "yield" in out
    # the full detail report is the rigid-pin model
    assert "Supports: center:fixed, 5.5:pin, -5.5:pin" in out


def test_cli_plot_writes_file(tmp_path):
    out = str(tmp_path / "my.png").replace("\\", "/")
    code = ("from elementload.cli import main; "
            "main(['--tube','0.030,0.002,0.6',"
            "'--tube','0.025,0.002,0.6',"
            "'--tube','0.020,0.002,0.6',"
            "'--tube','0.016,0.0015,0.6',"
            "'--overlap','0.05',"
            "'--joint','rigid',"
            "'--support','center:fixed',"
            "'--wind','40',"
            "'--plot','stress',"
            "'--plot-file','%s'])")
    result = subprocess.run([sys.executable, "-c", code % out],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    assert os.path.exists(out)
    assert os.path.getsize(out) > 0


def test_cli_stm_unknown_element_rejected():
    code = ("from elementload.cli import main; "
            "main(['--stm', 'data/mechanical2.stm', '--element', 'nope'])")
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "no element named" in result.stderr


def test_cli_optimize_reports_stock_vs_optimized():
    code = ("from elementload.cli import main; "
            "main(['--stm', 'data/a1.stm', '--element', '20m RE', '--optimize'])")
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "Optimization at target wind" in out
    assert "stock" in out and "optimized" in out
    assert "mass" in out and "saving" in out
    # the 20x2.0 -> 20x1.5 clearance fix is flagged
    assert "20.0 x 1.50 mm" in out
    assert "survival wind speed (yield)" in out
    assert "m/s" in out


def test_cli_optimize_reports_tip_extension_auto():
    code = ("from elementload.cli import main; "
            "main(['--stm', 'data/a1.stm', '--element', '20m RE', '--optimize'])")
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    # auto tip extension is the default; it must be reported on the optimized run
    assert "tip extension" in out and "auto" in out
    assert "mm/side" in out


def test_cli_optimize_tip_extension_fixed_reported():
    code = ("from elementload.cli import main; "
            "main(['--stm', 'data/a1.stm', '--element', '20m RE', '--optimize', "
            "'--tip-extension', '0.05'])")
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "tip extension 50 mm/side (fixed)" in out


def test_cli_optimize_tip_extension_zero_disables():
    code = ("from elementload.cli import main; "
            "main(['--stm', 'data/a1.stm', '--element', '20m RE', '--optimize', "
            "'--tip-extension', '0'])")
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    # a plain run reports mass/survival but no tip extension line
    assert "Optimization at target wind" in out
    assert "tip extension" not in out


def test_cli_optimize_requires_single_element():
    code = ("from elementload.cli import main; "
            "main(['--stm', 'data/a1.stm', '--optimize'])")
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "--optimize requires exactly one element" in result.stderr


def test_cli_optimize_es_sweep_reports_mass_vs_position():
    code = ("from elementload.cli import main; "
            "main(['--stm', 'data/a1.stm', '--element', '40m RE', '--optimize', "
            "'--es-sweep', '--es-range', '4', '10', '--es-steps', '7', '--wind', '35'])")
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "ES position sweep" in out
    assert "mass [kg]" in out and "survival [m/s]" in out
    assert "infeasible" in out  # near-tip and near-center positions fail
    # some feasible row carries a mass close to the optimized 14.0 kg chain
    # (tip extension auto is on by default, so the chain is slightly heavier)
    rows = [l for l in out.splitlines() if "  " in l and l.strip() and
            l.strip()[0].isdigit()]
    feasible = [l for l in rows if "infeasible" not in l]
    assert feasible, "expected at least one feasible sweep row"
    assert any("14.0" in l for l in feasible)


def test_cli_optimize_layout_search_reports_frontier():
    code = ("from elementload.cli import main; "
            "main(['--stm', 'data/a1.stm', '--element', '20m RE', '--optimize', "
            "'--layout-search', '--layout-workers', '2'])")
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "Layout design-space search" in out
    assert "survival" in out and "mass" in out and "chain" in out
    assert "Pareto frontier" in out and "per root tube diameter" in out
    # the known lightest-at-design layout from the brute force is found
    assert "40x2;35x2;30x2;25x2;20x1.5;12x1" in out
    rows = [l for l in out.splitlines() if l.startswith(" ")]
    assert any("<- lightest at/above design wind" in l for l in rows)


def test_cli_layout_search_emits_materialized_extent_command():
    """The layout-search report materializes each Pareto design as an elementload
    --extent command that reproduces the exact geometry and material."""
    code = ("from elementload.cli import main; "
            "main(['--stm', 'data/a1.stm', '--element', '20m RE', '--optimize', "
            "'--layout-search', '--layout-workers', '2', '--wind', '30'])")
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    out = result.stdout
    assert "Materialized Pareto designs" in out
    assert "Per-side tube extents" in out
    assert "tubes (centre -> tip)" in out
    # the command that materializes the lightest design is marked
    mat_lines = [l.strip() for l in out.splitlines()
                 if l.startswith("      elementload") and "--extent" in l]
    assert mat_lines, "expected at least one materialized extent command"
    # verify the marked command reproduces the reported survival/mass by
    # rebuilding the model from the extents it encodes
    from elementload.cli import _materialize_command
    from elementload.optimize import search_layouts, read_catalog, element_mass
    from elementload.report import build_model_from_geometry, survival_wind_speed
    from elementload.assembly import TubeExtent
    import re
    stm = read_stm("data/a1.stm")
    el = [e for e in stm.elements if e.name == "20m RE"][0]
    extents = element_tube_extents(el)
    supports = [("center", "fixed")]
    catalog = read_catalog("data/tube_catalog.csv")
    results = search_layouts(
        extents, "rigid", supports, 30.0, el.material_key, None,
        catalog=catalog, n_range=(3, 8), overlaps=(0.10,), ratios=(1.2,),
        sleeves=(0.15,), target_winds=[30.0], workers=2,
        tip_extension="auto")
    assert results
    r = min(results, key=lambda x: x["mass"])
    cmd = _materialize_command(r, "rigid", supports, 30.0, material="al6060-t66")
    m = re.match(r".*?--extent\s+(.*?)(?=\s*--material)", cmd)
    assert m, "command must carry --extent args before --material"
    specs = [TubeExtent(*[float(v) for v in g.split(",")])
             for g in re.findall(r"--extent\s+([\d.\-e]+,[\d.\-e]+,[\d.\-e]+,[\d.\-e]+)", cmd)]
    assert specs, "expected per-tube extent args in the command"
    model = build_model_from_geometry(specs, "rigid", supports, 30.0,
                                      "al6060-t66", 0.004, 2)
    assert survival_wind_speed(model, "yield") == pytest.approx(r["survival"], rel=1e-3)
    # rounded positions shift mass <0.1%
    assert element_mass(specs) == pytest.approx(r["mass"], rel=1e-3)
