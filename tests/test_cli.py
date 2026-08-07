import subprocess
import sys

import pytest

from elementload.cli import main, _ascii_chart, render_report
from elementload.report import build_model, analyze


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
    assert "km/h" in result.stdout
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


def test_cli_stm_unknown_element_rejected():
    code = ("from elementload.cli import main; "
            "main(['--stm', 'data/mechanical2.stm', '--element', 'nope'])")
    result = subprocess.run([sys.executable, "-c", code],
                            capture_output=True, text=True, timeout=120)
    assert result.returncode != 0
    assert "no element named" in result.stderr
