"""LSAR install: safe unpacking, conservative requirements, real import check."""

from __future__ import annotations

import io
import os
import json
import tarfile
from pathlib import Path
from typing import Any

import pytest

from tests.cli.infra_helpers import (  # noqa: F401 - fixtures
    FakeResponse,
    FakeSession,
    completed,
    edmars_home,
    no_network,
    serve_bytes,
)

from edmars import lsar, proc  # noqa: E402

pytestmark = pytest.mark.usefixtures("edmars_home", "no_network")


def _load_settings() -> dict[str, Any]:
    from edmars import settings

    return settings.load()

COMMIT = "0123456789abcdef0123456789abcdef01234567"
FAKE_DIST = "edmars-test-dist-that-does-not-exist"

REQUIREMENTS = f"""# LSAR Dependencies
PyYAML>=6.0.1
requests<1.0
{FAKE_DIST}>=1.0   # not installed anywhere
# Testing
pytest>=7.4.0
pytest-mock>=3.12.0
ruff>=0.5.0
"""

CONFIG = """llm:
  provider: "deepseek"
  model: "deepseek-v4-pro"
  stage_models:
    ingestion: "deepseek-v4-flash"
    review: "deepseek-v4-pro"
"""


def _tree(pipeline_body: str = "VALUE = 1\n", top: str = "LSAR-public-master",
          extra: dict[str, bytes] | None = None) -> dict[str, bytes]:
    files = {
        f"{top}/config.yaml": CONFIG.encode(),
        f"{top}/requirements.txt": REQUIREMENTS.encode(),
        f"{top}/calibration/anchors_edm.yaml": b"overall_p25_full: 6.3\nn_anchors: 15\n",
        f"{top}/lsar/__init__.py": b"",
        f"{top}/lsar/pipeline.py": pipeline_body.encode(),
    }
    files.update(extra or {})
    return files


def _targz(files: dict[str, bytes], *, commit: str | None = COMMIT,
           links: dict[str, str] | None = None) -> bytes:
    buf = io.BytesIO()
    pax = {"comment": commit} if commit else {}
    with tarfile.open(fileobj=buf, mode="w:gz", format=tarfile.PAX_FORMAT,
                      pax_headers=pax) as tf:
        dirs: set[str] = set()
        for name in files:
            parts = name.split("/")[:-1]
            for i in range(1, len(parts) + 1):
                dirs.add("/".join(parts[:i]))
        for d in sorted(dirs):
            info = tarfile.TarInfo(d)
            info.type = tarfile.DIRTYPE
            info.mode = 0o755
            tf.addfile(info)
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
        for name, target in (links or {}).items():
            info = tarfile.TarInfo(name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tf.addfile(info)
    return buf.getvalue()


class FakePip:
    """Answers pip calls; everything else runs for real (the import check)."""

    def __init__(self, real_run: Any, report: list[dict[str, Any]] | None = None,
                 install_rc: int = 0) -> None:
        self.real_run = real_run
        self.report = report or []
        self.install_rc = install_rc
        self.installed: list[list[str]] = []
        self.dry_runs: list[list[str]] = []
        self.envs: list[dict[str, str]] = []

    def __call__(self, args: list[str], **kwargs: Any) -> Any:
        if "pip" not in args:
            return self.real_run(args, **kwargs)
        self.envs.append(kwargs.get("env") or {})
        if "--version" in args:
            return completed(args, 0, "pip 25.1")
        req = Path(args[args.index("-r") + 1]).read_text(encoding="utf-8").split()
        if "--dry-run" in args:
            self.dry_runs.append(req)
            report = Path(args[args.index("--report") + 1])
            report.write_text(json.dumps({"install": self.report}), encoding="utf-8")
            return completed(args, 0)
        self.installed.append(req)
        return completed(args, self.install_rc, "", "ERROR: no matching distribution")


@pytest.fixture
def real_run() -> Any:
    return proc.run


def test_install_unpacks_installs_only_what_is_missing_and_verifies(
    monkeypatch: pytest.MonkeyPatch, real_run: Any
) -> None:
    settings_dict = _load_settings()
    pip = FakePip(real_run, report=[{"metadata": {"name": FAKE_DIST, "version": "1.2"}}])
    monkeypatch.setattr(proc, "run", pip)
    session = serve_bytes(_targz(_tree()))
    steps: list[str] = []

    home = lsar.install(settings_dict, session=session, on_step=steps.append)

    assert home == lsar.home_for_ref("master")
    assert session.calls[0]["url"] == "https://github.com/cgpan/LSAR-public/archive/master.tar.gz"
    for rel in lsar.REQUIRED_FILES:
        assert (home / rel).is_file()
    # dev tools and already-installed packages are never passed to pip
    assert pip.dry_runs == [[f"{FAKE_DIST}>=1.0"]]
    assert pip.installed == [[f"{FAKE_DIST}>=1.0"]]
    assert all(env.get("PIP_USER") == "0" for env in pip.envs)
    assert not any("API_KEY" in k for env in pip.envs for k in env)
    record = json.loads((home / lsar.INSTALL_RECORD).read_text(encoding="utf-8"))
    assert record["commit"] == COMMIT
    assert any(u.startswith("requests ") for u in record["unmet_pins"])
    assert settings_dict["lsar"]["home"] == str(home)
    assert settings_dict["lsar"]["ref"] == COMMIT
    from edmars import settings as settings_mod

    assert settings_mod.load()["lsar"]["home"] == str(home)
    assert not list(lsar.lsar_root().glob(".staging-*"))
    assert steps and steps[-1] == "Checking that LSAR loads"


def test_install_refuses_to_change_existing_packages(
    monkeypatch: pytest.MonkeyPatch, real_run: Any
) -> None:
    settings_dict = _load_settings()
    import importlib.metadata

    installed = importlib.metadata.version("PyYAML")
    pip = FakePip(real_run, report=[
        {"metadata": {"name": FAKE_DIST, "version": "1.2"}},
        {"metadata": {"name": "PyYAML", "version": installed + ".post999"}},
    ])
    monkeypatch.setattr(proc, "run", pip)
    with pytest.raises(lsar.LsarInstallError) as err:
        lsar.install(settings_dict, session=serve_bytes(_targz(_tree())))
    assert "PyYAML" in str(err.value) and "Nothing was installed" in str(err.value)
    assert err.value.plan is not None and err.value.plan.changes
    assert pip.installed == []
    assert not lsar.home_for_ref("master").exists()
    assert settings_dict["lsar"].get("home") in (None, "")


def test_allow_changes_passes_out_of_pin_requirements_too(
    monkeypatch: pytest.MonkeyPatch, real_run: Any
) -> None:
    settings_dict = _load_settings()
    pip = FakePip(real_run)
    monkeypatch.setattr(proc, "run", pip)
    lsar.install(settings_dict, session=serve_bytes(_targz(_tree())), allow_changes=True)
    assert "requests<1.0" in pip.installed[0]
    assert "PyYAML>=6.0.1" not in pip.installed[0]  # already satisfied


def test_a_copy_that_does_not_load_never_replaces_a_working_install(
    monkeypatch: pytest.MonkeyPatch, real_run: Any
) -> None:
    settings_dict = _load_settings()
    monkeypatch.setattr(proc, "run", FakePip(real_run))
    good = lsar.install(settings_dict, session=serve_bytes(_targz(_tree())))
    marker = good / "lsar" / "pipeline.py"
    before = marker.read_text(encoding="utf-8")

    broken = _tree("import edmars_no_such_module_xyz\n")
    with pytest.raises(lsar.LsarInstallError, match="edmars_no_such_module_xyz"):
        lsar.install(settings_dict, session=serve_bytes(_targz(broken)))
    assert marker.read_text(encoding="utf-8") == before


def test_a_failed_pip_install_is_reported(
    monkeypatch: pytest.MonkeyPatch, real_run: Any
) -> None:
    settings_dict = _load_settings()
    monkeypatch.setattr(proc, "run", FakePip(real_run, install_rc=1))
    with pytest.raises(lsar.LsarInstallError, match="no matching distribution"):
        lsar.install(settings_dict, session=serve_bytes(_targz(_tree())))


@pytest.mark.parametrize(
    "bad_name",
    ["LSAR-public-master/../../evil.txt", "/abs/evil.txt", "LSAR-public-master/C:evil",
     "LSAR-public-master\\..\\evil.txt"],
)
def test_unsafe_archive_paths_are_refused(tmp_path: Path, bad_name: str) -> None:
    archive = tmp_path / "a.tar.gz"
    archive.write_bytes(_targz(_tree(extra={bad_name: b"x"})))
    with pytest.raises(lsar.LsarInstallError, match="unsafe path"):
        lsar._safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_links_are_skipped_and_one_top_folder_is_required(tmp_path: Path) -> None:
    archive = tmp_path / "a.tar.gz"
    archive.write_bytes(_targz(_tree(), links={"LSAR-public-master/link": "../../outside"}))
    top, commit = lsar._safe_extract(archive, tmp_path / "out")
    assert top.name == "LSAR-public-master" and commit == COMMIT
    assert not (top / "link").exists()

    two = tmp_path / "b.tar.gz"
    two.write_bytes(_targz(_tree(extra={"second/x.txt": b"x"})))
    with pytest.raises(lsar.LsarInstallError, match="one folder"):
        lsar._safe_extract(two, tmp_path / "out2")


def test_download_failure_is_an_install_error() -> None:
    settings_dict = _load_settings()
    session = FakeSession(lambda u, h, p: FakeResponse(404, body=b"no such ref"))
    with pytest.raises(lsar.LsarInstallError, match="HTTP 404"):
        lsar.install(settings_dict, ref="v9.9.9", session=session)
    assert session.calls[0]["url"].endswith("/archive/v9.9.9.tar.gz")


def test_verify_names_missing_files_and_missing_modules(tmp_path: Path) -> None:
    assert lsar.verify(tmp_path / "nowhere")
    home = tmp_path / "home"
    for rel, data in _tree(top="x").items():
        target = home / rel.split("/", 1)[1]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    assert lsar.verify(home) == []
    (home / "calibration" / "anchors_edm.yaml").write_text("n_anchors: 1\n", encoding="utf-8")
    assert "overall_p25_full" in " ".join(lsar.verify(home))
    (home / "calibration" / "anchors_edm.yaml").write_text("overall_p25_full: 6\n",
                                                           encoding="utf-8")
    (home / "lsar" / "pipeline.py").write_text("import tenacity_missing_for_test\n",
                                               encoding="utf-8")
    problems = lsar.verify(home)
    assert problems and "tenacity_missing_for_test" in problems[0]
    (home / "config.yaml").unlink()
    assert "config.yaml is missing" in lsar.verify(home)[0]


def test_runtime_requirements_drop_dev_tools(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    assert lsar.runtime_requirements(tmp_path) == [
        "PyYAML>=6.0.1", "requests<1.0", f"{FAKE_DIST}>=1.0"]


def test_uv_dry_run_output_is_parsed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    output = ("Resolved 12 packages in 40ms\nWould uninstall 1 package\n"
              "Would install 2 packages\n - pymupdf4llm==1.27.2.2\n"
              " + pymupdf4llm==0.0.27\n + tenacity==9.1.2\n")
    monkeypatch.setattr(proc, "run", lambda args, **kw: completed(args, 0, "", output))
    req = tmp_path / "r.txt"
    req.write_text("tenacity\n", encoding="utf-8")
    changes, method = lsar._dry_run_changes(["uv", "pip", "install"], "uv", req)
    assert method == "uv"
    assert changes == ["pymupdf4llm 1.27.2.2 -> 0.0.27"]


def test_checks_when_not_installed() -> None:
    settings_dict = _load_settings()
    assert [c.status for c in lsar.checks(settings_dict)] == ["info"]
    settings_dict["lsar"]["enabled"] = True
    only = lsar.checks(settings_dict)
    assert [c.status for c in only] == ["fail"] and only[0].fix == "edmars setup lsar"


def test_checks_report_retired_models_missing_key_and_deep_import(
    monkeypatch: pytest.MonkeyPatch, real_run: Any
) -> None:
    settings_dict = _load_settings()
    monkeypatch.setattr(proc, "run", FakePip(real_run))
    lsar.install(settings_dict, session=serve_bytes(_targz(_tree())))
    settings_dict["lsar"]["enabled"] = True
    from edmars import secrets

    monkeypatch.setattr(secrets, "get_secret", lambda name: None)
    by_name = {c.name: c for c in lsar.checks(settings_dict, deep=True)}
    assert by_name["Automated reviewer (LSAR)"].status == "ok"
    assert COMMIT[:12] in by_name["Automated reviewer (LSAR)"].detail
    assert by_name["LSAR model settings"].status == "warn"
    assert "ingestion: deepseek-v4-flash" in by_name["LSAR model settings"].detail
    assert by_name["DeepSeek key for LSAR"].status == "fail"
    assert by_name["LSAR loads in Python"].status == "ok"
    assert by_name["LSAR's Python packages"].status in ("ok", "warn", "fail")

    monkeypatch.setattr(secrets, "get_secret", lambda name: "present")
    names = [c.name for c in lsar.checks(settings_dict)]
    assert "DeepSeek key for LSAR" not in names
    assert "LSAR loads in Python" not in names


# ---------------------------------------------------------------------------
# gate config and `edmars review RUN`
# ---------------------------------------------------------------------------

FAKE_REVIEW_SCRIPT = '''
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
out = Path(args[args.index("--output-dir") + 1])
venue = args[args.index("--venue") + 1]
print("Stage 1: PDF Ingestion for", Path(args[0]).name, "venue", venue)
print("using key", os.environ.get("DEEPSEEK_API_KEY"))
if os.environ.get("FAKE_LSAR_FAIL"):
    sys.exit(3)
out.mkdir(parents=True, exist_ok=True)
(out / "LSAR_Review_Report.json").write_text(json.dumps(
    {"scores": {"overall_score": 6.8, "recommendation": "Accept"}}), encoding="utf-8")
(out / "LSAR_Review_Report.md").write_text("# Review", encoding="utf-8")
'''


def _home(tmp_path: Path) -> Path:
    home = tmp_path / "lsar-home"
    for rel, data in _tree(top="x").items():
        target = home / rel.split("/", 1)[1]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (home / "calibration" / "anchors_edm.yaml").write_text(
        "overall_p25_full: 6.3\nvenues:\n  AERA_OPEN:\n    p25: 6.6\n  JEDM:\n    p25: 5.15\n",
        encoding="utf-8")
    (home / "scripts").mkdir()
    (home / "scripts" / "run_review.py").write_text(FAKE_REVIEW_SCRIPT, encoding="utf-8")
    return home


def _run_folder(tmp_path: Path, venue: str | None = "AERA Open") -> Path:
    run = tmp_path / "study"
    run.mkdir()
    (run / "paper.tex").write_text(r"\documentclass{article}", encoding="utf-8")
    (run / "paper.pdf").write_bytes(b"%PDF-1.5 fake")
    if venue:
        (run / "run_config.yaml").write_text(f"review_gate:\n  venue: {venue}\n",
                                             encoding="utf-8")
    return run


def test_gate_config_uses_absolute_paths(tmp_path: Path) -> None:
    home = _home(tmp_path)
    cfg = lsar.gate_config({"lsar": {"home": str(home)}}, "AERA Open")
    assert cfg == {
        "enabled": True,
        "lsar_project_path": str(home),
        "lsar_config_path": str(home / "config.yaml"),
        "calibration_path": str(home / "calibration" / "anchors_edm.yaml"),
        "venue": "AERA_OPEN",
    }
    assert "${" not in json.dumps(cfg)
    assert lsar.gate_config({"lsar": {}}) == {"enabled": False}


def test_benchmarks_come_from_the_calibration_file(tmp_path: Path) -> None:
    home = _home(tmp_path)
    assert lsar.benchmark_for(home, "EDM") == 6.3
    assert lsar.benchmark_for(home, "jedm") == 5.15
    assert lsar.benchmark_for(home, "AERA Open") == 6.6
    assert lsar.benchmark_for(home, "LAK") is None


def test_review_run_writes_only_its_own_folder_and_redacts_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = "sk" + "-" + "r3v13w" * 4
    monkeypatch.setenv("DEEPSEEK_API_KEY", key)
    from edmars import secrets

    monkeypatch.setattr(secrets, "get_secret", lambda name: os.environ.get(name))
    if hasattr(secrets, "child_secrets"):
        monkeypatch.setattr(secrets, "child_secrets",
                            lambda names: {n: os.environ[n] for n in names if os.environ.get(n)})
    home = _home(tmp_path)
    run = _run_folder(tmp_path)
    before = {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()}

    result = lsar.review_run(run, {"lsar": {"home": str(home)}}, timeout_s=300)

    assert result["venue"] == "AERA_OPEN"
    assert result["score"] == 6.8 and result["benchmark"] == 6.6 and result["passed"] is True
    assert result["output_dir"] == run / "lsar_review_manual"
    assert {p.name: p.read_bytes() for p in run.iterdir() if p.is_file()} == before
    log = (run / "lsar_review_manual" / "console.log").read_text(encoding="utf-8")
    assert "Stage 1" in log and key not in log


def test_review_run_failures_are_explained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = _home(tmp_path)
    settings = {"lsar": {"home": str(home)}}
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(lsar.LsarReviewError, match="no paper PDF"):
        lsar.review_run(empty, settings)
    with pytest.raises(lsar.LsarReviewError, match="not installed"):
        lsar.review_run(empty, {"lsar": {}})

    run = _run_folder(tmp_path, venue=None)
    from edmars import secrets

    monkeypatch.setattr(secrets, "get_secret", lambda name: None)
    if hasattr(secrets, "child_secrets"):
        monkeypatch.setattr(secrets, "child_secrets", lambda names: {})
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(lsar.LsarReviewError, match="DeepSeek key"):
        lsar.review_run(run, settings)

    monkeypatch.setattr(secrets, "get_secret", lambda name: "present")
    if hasattr(secrets, "child_secrets"):
        monkeypatch.setattr(secrets, "child_secrets",
                            lambda names: {"DEEPSEEK_API_KEY": "present"})
    monkeypatch.setenv("FAKE_LSAR_FAIL", "1")
    with pytest.raises(lsar.LsarReviewError, match="exit code 3"):
        lsar.review_run(run, settings, timeout_s=300)


def test_a_second_install_replaces_the_first(
    monkeypatch: pytest.MonkeyPatch, real_run: Any
) -> None:
    settings_dict = _load_settings()
    monkeypatch.setattr(proc, "run", FakePip(real_run))
    home = lsar.install(settings_dict, session=serve_bytes(_targz(_tree("VALUE = 1\n"))))
    home2 = lsar.install(settings_dict, session=serve_bytes(_targz(_tree("VALUE = 2\n"))))
    assert home2 == home
    assert (home / "lsar" / "pipeline.py").read_text(encoding="utf-8") == "VALUE = 2\n"
    assert not list(lsar.lsar_root().glob(".old-*"))
    assert not list(lsar.lsar_root().glob(".staging-*"))


def test_requirements_for_another_python_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text(
        'tenacity>=8\nbackports.zoneinfo; python_version < "3.9"\n', encoding="utf-8")
    assert lsar.runtime_requirements(tmp_path) == ["tenacity>=8"]
