import os
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
