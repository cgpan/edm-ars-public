"""src/preflight.py: what a run needs, checked before it spends anything."""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pytest

import src.preflight as preflight
from src.config import load_config
from src.preflight import (
    FAIL,
    WARN,
    Finding,
    check_run_prerequisites,
    has_failures,
    required_r_packages,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def config(monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setenv("LSAR_HOME", "")
    cfg = load_config(str(REPO_ROOT / "config.yaml"))
    cfg = copy.deepcopy(cfg)
    cfg["llm_provider"] = "deepseek"
    cfg["review_gate"]["enabled"] = False
    cfg["sandbox"]["enabled"] = False
    return cfg


@pytest.fixture
def data_file(tmp_path: Path) -> str:
    path = tmp_path / "hsls_17_student_pets_sr_v1_0.csv"
    path.write_text("X1SEX\nMale\n", encoding="utf-8")
    return str(path)


@pytest.fixture
def all_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight.shutil, "which", lambda name: f"/bin/{name}")


def _codes(findings: list[Finding]) -> set[str]:
    return {f.code for f in findings}


def test_clean_setup_has_no_findings(config: dict, data_file: str, all_tools: None) -> None:
    findings = check_run_prerequisites(
        config, "prediction", "hsls09_public", data_file, False
    )
    assert findings == []
    assert not has_failures(findings)


def test_findings_are_plain_tuples(config: dict, all_tools: None,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    findings = check_run_prerequisites(
        config, "prediction", "hsls09_public", "missing.csv", False
    )
    code, severity, message, fix = findings[0]
    assert (code, severity) == ("KEY_MISSING", FAIL)
    assert message and fix


def test_missing_key_fails_and_is_never_echoed(
    config: dict, data_file: str, all_tools: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    findings = check_run_prerequisites(
        config, "prediction", "hsls09_public", data_file, False
    )
    [key] = [f for f in findings if f.code == "KEY_MISSING"]
    assert key.severity == FAIL
    assert "DEEPSEEK_API_KEY" in key.message
    assert "$env:DEEPSEEK_API_KEY" in key.fix  # PowerShell users get a line too
    assert ".env" in key.fix


def test_key_value_never_appears(
    config: dict, data_file: str, all_tools: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-secret-value-123456")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config["per_stage_providers"] = {"analyst": {"provider": "openai", "model": "gpt-x"}}
    findings = check_run_prerequisites(
        config, "prediction", "hsls09_public", data_file, False
    )
    assert [f.code for f in findings] == ["KEY_MISSING"]
    assert "OPENAI_API_KEY" in findings[0].message
    assert "analyst" in findings[0].message
    assert all("sk-secret" not in " ".join(f) for f in findings)


def test_missing_raw_data_names_the_exact_file(
    config: dict, tmp_path: Path, all_tools: None
) -> None:
    (tmp_path / "HSLS_2017_PETS_SR.csv").write_text("x\n", encoding="utf-8")
    expected = tmp_path / "hsls_17_student_pets_sr_v1_0.csv"
    findings = check_run_prerequisites(
        config, "prediction", "hsls09_public", str(expected), False
    )
    [data] = [f for f in findings if f.code == "DATA_MISSING"]
    assert data.severity == FAIL
    assert str(expected) in data.message
    assert "hsls_17_student_pets_sr_v1_0.csv" in data.fix
    assert "nces.ed.gov/surveys/hsls09" in data.fix
    # A file saved under the provider's own name is pointed out.
    assert "HSLS_2017_PETS_SR.csv" in data.fix


def test_missing_panel_points_at_the_harmonizer(
    config: dict, tmp_path: Path, all_tools: None
) -> None:
    findings = check_run_prerequisites(
        config, "causal_did", "did_els_hsls_panel",
        str(tmp_path / "did_els_hsls_panel" / "panel.csv"), False,
    )
    [data] = [f for f in findings if f.code == "DATA_MISSING"]
    assert "harmonize_els_hsls.py" in data.fix


def test_missing_pdflatex_is_a_warning(
    config: dict, data_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preflight.shutil, "which", lambda name: None)
    findings = check_run_prerequisites(
        config, "prediction", "hsls09_public", data_file, False
    )
    [latex] = [f for f in findings if f.code == "LATEX_MISSING"]
    assert latex.severity == WARN
    assert "no PDF" in latex.message
    assert not has_failures(findings)


def test_journal_venue_needs_biber(
    config: dict, data_file: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    config["writer"]["venue_format"] = "journal"
    monkeypatch.setattr(
        preflight.shutil, "which",
        lambda name: None if name == "biber" else f"/bin/{name}",
    )
    findings = check_run_prerequisites(
        config, "prediction", "hsls09_public", data_file, False
    )
    assert [f.message.split()[0] for f in findings] == ["biber"]


def test_prompt_folder_missing_fails(
    config: dict, data_file: str, all_tools: None, tmp_path: Path
) -> None:
    config["paths"]["agent_prompts"] = str(tmp_path / "nowhere")
    findings = check_run_prerequisites(
        config, "prediction", "hsls09_public", data_file, False
    )
    assert "INSTALL_INCOMPLETE" in _codes(findings)
    assert has_failures(findings)


# ---------------------------------------------------------------------------
# R (psychometrics)
# ---------------------------------------------------------------------------


def test_r_packages_follow_the_method_battery() -> None:
    assert required_r_packages({"method_battery": ["P1", "P3", "P6"]}) == (
        ["jsonlite", "lavaan"], ["mirt", "CDM", "MASS"],
    )
    assert required_r_packages({"method_battery": ["P7"]})[0] == ["jsonlite", "CDM"]
    # CTT alone runs in Python: R is not needed at all.
    assert required_r_packages({"method_battery": ["P1"]}) == ([], [
        "jsonlite", "lavaan", "mirt", "CDM", "MASS"])
    # No battery (or an unknown id): assume every helper may run.
    assert required_r_packages(None)[0] == ["jsonlite", "lavaan", "mirt", "CDM", "MASS"]
    assert required_r_packages({"method_battery": ["P9"]})[1] == []


def test_missing_rscript_fails(
    config: dict, data_file: str, all_tools: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.r_bridge as rb

    def _nope(explicit: str | None = None) -> str:
        raise rb.RBridgeError("Rscript not found.")

    monkeypatch.setattr(rb, "find_rscript", _nope)
    findings = check_run_prerequisites(
        config, "psychometrics", "hsls09_public", data_file, False,
        locked_spec={"method_battery": ["P3"]},
    )
    [r] = [f for f in findings if f.code == "R_MISSING"]
    assert r.severity == FAIL
    assert "EDM_ARS_RSCRIPT" in r.fix


def test_missing_r_packages_split_by_need(
    config: dict, data_file: str, all_tools: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.r_bridge as rb

    calls: list[tuple] = []
    monkeypatch.setattr(rb, "find_rscript", lambda explicit=None: "/usr/bin/Rscript")

    def _probe(name: str, payload: dict, timeout_s: int = 600,
               rscript_path: str | None = None) -> dict:
        calls.append((name, rscript_path))
        return {"installed": ["jsonlite", "MASS"],
                "missing": ["lavaan", "mirt", "CDM"], "r_version": "4.4.1"}

    monkeypatch.setattr(rb, "run_r_script", _probe)
    findings = check_run_prerequisites(
        config, "psychometrics", "hsls09_public", data_file, False,
        locked_spec={"method_battery": ["P3", "P5"]},
    )
    assert calls == [("preflight_packages.R", "/usr/bin/Rscript")]
    by_sev = {f.severity: f for f in findings if f.code == "R_PACKAGES_MISSING"}
    assert "lavaan" in by_sev[FAIL].message and "mirt" not in by_sev[FAIL].message
    assert "mirt" in by_sev[WARN].message and "CDM" in by_sev[WARN].message
    assert "install.packages" in by_sev[FAIL].fix


def test_r_is_not_probed_for_other_task_types(
    config: dict, data_file: str, all_tools: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import src.r_bridge as rb

    monkeypatch.setattr(rb, "find_rscript", lambda explicit=None: pytest.fail("probed"))
    check_run_prerequisites(config, "causal_soo", "hsls09_public", data_file, False)


def test_r_probe_against_a_real_r_when_present(
    config: dict, data_file: str, all_tools: None
) -> None:
    from src.r_bridge import RBridgeError, find_rscript, run_r_script

    try:
        rscript = find_rscript()
    except RBridgeError:
        pytest.skip("R is not installed on this machine")
    result = run_r_script("preflight_packages.R", {}, timeout_s=180,
                          rscript_path=rscript)
    assert set(result) >= {"installed", "missing", "r_version"}
    assert set(result["installed"]) | set(result["missing"]) == {
        "jsonlite", "lavaan", "mirt", "CDM", "MASS"}


# ---------------------------------------------------------------------------
# LSAR (review gate)
# ---------------------------------------------------------------------------


@pytest.fixture
def _restore_lsar_modules() -> None:
    before = {k: v for k, v in sys.modules.items()
              if k == "lsar" or k.startswith("lsar.")}
    path_before = list(sys.path)
    yield
    for key in [k for k in sys.modules if k == "lsar" or k.startswith("lsar.")]:
        if key not in before:
            del sys.modules[key]
    sys.modules.update(before)
    sys.path[:] = path_before


def test_gate_with_no_lsar_warns(
    config: dict, data_file: str, all_tools: None, tmp_path: Path
) -> None:
    config["review_gate"]["lsar_project_path"] = str(tmp_path / "LSAR")
    findings = check_run_prerequisites(
        config, "prediction", "hsls09_public", data_file, True
    )
    [lsar] = [f for f in findings if f.code == "LSAR_NOT_FOUND"]
    assert lsar.severity == WARN
    assert "will not run" in lsar.message
    assert not has_failures(findings)


def test_gate_with_unimportable_lsar_warns(
    config: dict, data_file: str, all_tools: None, tmp_path: Path,
    _restore_lsar_modules: None,
) -> None:
    root = tmp_path / "LSAR-fake"
    (root / "lsar").mkdir(parents=True)
    (root / "lsar" / "__init__.py").write_text("", encoding="utf-8")
    (root / "lsar" / "pipeline.py").write_text(
        "import a_dependency_that_is_not_installed\n", encoding="utf-8")
    config["review_gate"]["lsar_project_path"] = str(root)
    sys.modules.pop("lsar", None)
    findings = check_run_prerequisites(
        config, "prediction", "hsls09_public", data_file, True
    )
    [lsar] = [f for f in findings if f.code == "LSAR_IMPORT_FAILED"]
    assert "a_dependency_that_is_not_installed" in lsar.message
    assert "requirements.txt" in lsar.fix
    assert str(root) not in sys.path


def test_gate_disabled_skips_lsar(
    config: dict, data_file: str, all_tools: None, tmp_path: Path
) -> None:
    config["review_gate"]["lsar_project_path"] = str(tmp_path / "LSAR")
    assert check_run_prerequisites(
        config, "prediction", "hsls09_public", data_file, False
    ) == []
