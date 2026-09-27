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

COMMIT = lsar.LSAR_REF
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

    assert home == lsar.home_for_ref()
    assert session.calls[0]["url"] == (
        f"https://github.com/cgpan/LSAR-public/archive/{lsar.LSAR_REF}.tar.gz")
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
    assert not lsar.home_for_ref().exists()
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
    assert top == (tmp_path / "out").resolve() and commit == COMMIT
    assert not (top / "link").exists()

    two = tmp_path / "b.tar.gz"
    two.write_bytes(_targz(_tree(extra={"second/x.txt": b"x"})))
    with pytest.raises(lsar.LsarInstallError, match="one folder"):
        lsar._safe_extract(two, tmp_path / "out2")


def test_the_archive_top_folder_is_not_recreated_on_disk(tmp_path: Path) -> None:
    # Found on Windows: GitHub names the archive's folder LSAR-public-<40-hex
    # commit>; unpacked under the data folder's staging folder it pushed
    # lsar/stage4_review_generation/templates/aera_open_template.md to 260
    # characters, past Windows' path limit, and the install stopped with
    # "No such file or directory". Its contents now land in dest directly.
    top = f"LSAR-public-{COMMIT}"
    deep = "lsar/stage4_review_generation/templates/aera_open_template.md"
    archive = tmp_path / "a.tar.gz"
    archive.write_bytes(_targz(_tree(top=top, extra={f"{top}/{deep}": b"x"})))
    dest = tmp_path / "staging"
    unpacked, commit = lsar._safe_extract(archive, dest)
    assert unpacked == dest.resolve() and commit == COMMIT
    assert not (dest / top).exists()
    assert (dest / deep).read_bytes() == b"x"
    for rel in lsar.REQUIRED_FILES:
        assert (dest / rel).is_file()
    deepest = max(len(str(p)) for p in dest.rglob("*"))
    assert deepest == len(str(dest.resolve() / deep))


def test_install_puts_the_files_directly_in_the_home_folder(
    monkeypatch: pytest.MonkeyPatch, real_run: Any
) -> None:
    monkeypatch.setattr(proc, "run", FakePip(real_run))
    top = f"LSAR-public-{COMMIT}"
    home = lsar.install(_load_settings(), session=serve_bytes(_targz(_tree(top=top))))
    assert home == lsar.home_for_ref()
    assert not (home / top).exists()
    assert (home / "lsar" / "pipeline.py").is_file()
    assert not list(lsar.lsar_root().glob(".staging-*"))


def test_the_pinned_ref_is_a_commit_id_not_a_branch() -> None:
    assert lsar.is_commit_id(lsar.LSAR_REF)
    assert not lsar.is_commit_id("master")


@pytest.mark.parametrize("recorded", ["f" * 40, None])
def test_an_archive_of_another_commit_is_refused_before_pip_runs(
    monkeypatch: pytest.MonkeyPatch, real_run: Any, recorded: str | None
) -> None:
    settings_dict = _load_settings()
    pip = FakePip(real_run)
    monkeypatch.setattr(proc, "run", pip)
    with pytest.raises(lsar.LsarInstallError, match="not the version EDM-ARS was tested with"):
        lsar.install(settings_dict, session=serve_bytes(_targz(_tree(), commit=recorded)))
    assert pip.dry_runs == [] and pip.installed == []
    assert not lsar.home_for_ref().exists()
    assert settings_dict["lsar"].get("home") in (None, "")


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


DLL_ERROR = ("DLL load failed while importing onnxruntime_pybind11_state: "
             "The specified module could not be found.")


def _home_with_converter(tmp_path: Path, layout_body: str) -> Path:
    """An LSAR home whose folder also shadows pymupdf4llm and pymupdf.layout.

    The import check puts the home first on sys.path, so these stand-ins
    are what the child Python imports.
    """
    home = tmp_path / "home"
    for rel, data in _tree(top="x").items():
        target = home / rel.split("/", 1)[1]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (home / "pymupdf4llm").mkdir()
    (home / "pymupdf4llm" / "__init__.py").write_text("", encoding="utf-8")
    (home / "pymupdf" / "layout").mkdir(parents=True)
    (home / "pymupdf" / "__init__.py").write_text("", encoding="utf-8")
    (home / "pymupdf" / "layout" / "__init__.py").write_text(layout_body, encoding="utf-8")
    return home


def test_verify_reports_a_pdf_layout_model_that_cannot_load(tmp_path: Path) -> None:
    # pymupdf4llm swallows this ImportError and switches to its classic
    # converter, so a review would run on different text than LSAR's
    # benchmark was calibrated with. On Windows the cause is onnxruntime's
    # MSVCP140.dll / MSVCP140_1.dll, which no wheel ships.
    home = _home_with_converter(tmp_path, f"raise ImportError({DLL_ERROR!r})\n")
    problems = lsar.verify(home)
    assert len(problems) == 1
    assert "Visual C++ Redistributable" in problems[0]
    assert lsar.VC_REDIST_URL in problems[0]
    assert "edmars setup reviewer" in problems[0]


def test_verify_passes_when_the_pdf_layout_model_loads(tmp_path: Path) -> None:
    assert lsar.verify(_home_with_converter(tmp_path, "LOADED = True\n")) == []


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


def test_not_ready_is_always_a_failure(tmp_path: Path) -> None:
    """Callers ask before switching LSAR on; "no fail" must mean "ready"."""
    settings_dict = _load_settings()
    for enabled in (False, True):
        settings_dict["lsar"]["enabled"] = enabled
        only = lsar.checks(settings_dict)
        assert [c.status for c in only] == ["fail"]
        assert only[0].fix == "edmars setup reviewer"
    settings_dict["lsar"]["enabled"] = False
    settings_dict["lsar"]["home"] = str(tmp_path / "gone")
    assert [c.status for c in lsar.checks(settings_dict)] == ["fail"]


def test_checks_report_retired_models_and_deep_import(
    monkeypatch: pytest.MonkeyPatch, real_run: Any
) -> None:
    settings_dict = _load_settings()
    monkeypatch.setattr(proc, "run", FakePip(real_run))
    lsar.install(settings_dict, session=serve_bytes(_targz(_tree())))
    settings_dict["lsar"]["enabled"] = True
    _all_packages_present(monkeypatch)
    by_name = {c.name: c for c in lsar.checks(settings_dict, deep=True)}
    assert by_name["Automated reviewer (LSAR)"].status == "ok"
    assert COMMIT[:12] in by_name["Automated reviewer (LSAR)"].detail
    assert by_name["LSAR model settings"].status == "warn"
    assert "ingestion: deepseek-v4-flash" in by_name["LSAR model settings"].detail
    assert by_name["LSAR loads in Python"].status == "ok"
    assert by_name["LSAR's Python packages"].status in ("ok", "warn", "fail")
    assert "LSAR loads in Python" not in [c.name for c in lsar.checks(settings_dict)]

    monkeypatch.setattr(lsar.importlib.util, "find_spec",
                        lambda name, *a: None if name == "tenacity" else object())
    headline = {c.name: c for c in lsar.checks(settings_dict)}["Automated reviewer (LSAR)"]
    assert headline.status == "fail" and "tenacity" in headline.detail


MAC_MISSING = ("tenacity", "pymupdf4llm", "arxiv")
MAC_IMPORT_ERROR = ("LSAR needs the Python package 'tenacity', which is not installed "
                    "for the Python that runs EDM-ARS.")


def _all_packages_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """The quick check sees every LSAR package, whatever this Python has."""
    real = lsar.importlib.util.find_spec
    monkeypatch.setattr(lsar.importlib.util, "find_spec",
                        lambda name, *a: object() if name in lsar.RUNTIME_MODULES else real(name, *a))


def _installed_and_on(monkeypatch: pytest.MonkeyPatch, real_run: Any) -> dict[str, Any]:
    settings_dict = _load_settings()
    monkeypatch.setattr(proc, "run", FakePip(real_run))
    lsar.install(settings_dict, session=serve_bytes(_targz(_tree())))
    settings_dict["lsar"]["enabled"] = True
    _all_packages_present(monkeypatch)
    return settings_dict


def test_a_reviewer_whose_packages_are_gone_is_one_problem_not_on_plus_failures(
    monkeypatch: pytest.MonkeyPatch, real_run: Any
) -> None:
    # The Mac test after an update: doctor said "Automated reviewer (LSAR):
    # on" and "[x] LSAR's Python packages: Missing: tenacity, pymupdf4llm,
    # arxiv"; --deep added "[x] LSAR loads in Python" for the same package,
    # so plain doctor counted 1 problem and --deep 2.
    settings_dict = _installed_and_on(monkeypatch, real_run)
    monkeypatch.setattr(lsar.importlib.util, "find_spec",
                        lambda name, *a: None if name in MAC_MISSING else object())
    monkeypatch.setattr(lsar, "verify", lambda home: [MAC_IMPORT_ERROR])
    for deep in (False, True):
        found = lsar.checks(settings_dict, deep=deep)
        failing = [c for c in found if c.status == "fail"]
        assert len(failing) == 1, found
        [line] = failing
        assert line.name == "Automated reviewer (LSAR)"
        assert line.detail.startswith("Turned on, but not ready: LSAR's Python packages are "
                                      "missing (tenacity, pymupdf4llm, arxiv)")
        assert "every review is skipped" in line.detail
        assert line.detail.count("tenacity") == 1  # the import check does not repeat it
        assert line.fix == "edmars setup reviewer"
        assert not any(c.status == "ok" for c in found)  # never "on" beside the failure
        assert {c.name for c in found} <= {"Automated reviewer (LSAR)", "LSAR model settings"}
    assert lsar.unavailable_reason(settings_dict) == (
        "LSAR's Python packages are missing (tenacity, pymupdf4llm, arxiv), so every "
        "review is skipped.")


def test_a_problem_only_the_deep_check_finds_becomes_the_reviewers_own_line(
    monkeypatch: pytest.MonkeyPatch, real_run: Any
) -> None:
    settings_dict = _installed_and_on(monkeypatch, real_run)
    settings_dict["lsar"]["enabled"] = False
    dll = "The PDF layout model LSAR's scores were calibrated with cannot load (DLL load failed)."
    monkeypatch.setattr(lsar, "verify", lambda home: [dll])
    assert lsar.checks(settings_dict)[0].status == "ok"
    found = lsar.checks(settings_dict, deep=True)
    assert [c.status for c in found if c.status != "warn"] == ["fail"]
    assert found[0].name == "Automated reviewer (LSAR)"
    assert found[0].detail == "Installed (turned off), but not ready: " + dll
    assert lsar.unavailable_reason(settings_dict) is None  # only the import finds it


# ---------------------------------------------------------------------------
# after an update: `edmars after-install` (the installer's last step)
# ---------------------------------------------------------------------------

FAKE_MODULE = "edmars_fake_lsar_dep"


class FakeUv:
    """The installed environment: no pip, only the uv that built it.

    ``uv pip install`` "installs" by writing FAKE_MODULE into the LSAR
    folder, which the import check puts on sys.path, so the real import
    check in a child Python sees the package appear. Every other call
    (the import check) runs for real.
    """

    def __init__(self, real_run: Any, home: Path, *, install_rc: int = 0,
                 dry_run_output: str = "Would install 1 package\n + fake==1.2\n") -> None:
        self.real_run, self.home, self.install_rc = real_run, home, install_rc
        self.dry_run_output = dry_run_output
        self.calls: list[list[str]] = []
        self.requirements: list[list[str]] = []

    def __call__(self, args: list[str], **kwargs: Any) -> Any:
        args = [str(a) for a in args]
        if args[1:4] == ["-m", "pip", "--version"]:
            return completed(args, 1, "", "No module named pip")
        if args[0] != "/fake/uv":
            return self.real_run(args, **kwargs)
        self.calls.append(args)
        self.requirements.append(Path(args[args.index("-r") + 1]).read_text(encoding="utf-8").split())
        if "--dry-run" in args:
            return completed(args, 0, self.dry_run_output)
        if self.install_rc == 0:
            (self.home / f"{FAKE_MODULE}.py").write_text("OK = True\n", encoding="utf-8")
        return completed(args, self.install_rc, "", "error: network unreachable")


def _set_up_reviewer(*, enabled: bool = True, finished: bool = True) -> Path:
    """Settings as `edmars setup` leaves them, with an LSAR folder in place
    whose lsar.pipeline needs a package the new environment lacks."""
    from edmars import settings as settings_mod

    home = lsar.home_for_ref()
    for name, data in _tree(f"import {FAKE_MODULE}\n").items():
        path = home / name.split("/", 1)[1]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (home / lsar.INSTALL_RECORD).write_text(json.dumps({"ref": COMMIT, "unmet_pins": ["old"]}),
                                           encoding="utf-8")
    current = settings_mod.load()
    current["lsar"].update(enabled=enabled, auto_review=enabled, home=str(home), ref=COMMIT)
    current["setup_progress"]["last_completed_screen"] = "S11" if finished else "S5"
    settings_mod.save(current)
    return home


def _state(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text(encoding="utf-8").splitlines())


@pytest.mark.parametrize("enabled", [True, False])
def test_after_install_puts_lsars_packages_back_into_the_rebuilt_environment(
    monkeypatch: pytest.MonkeyPatch, real_run: Any, tmp_path: Path, enabled: bool,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # The Mac test's update: the installer rebuilt venv-0.1.0 from scratch,
    # LSAR's packages were gone, and every review was skipped without a
    # word from the installer. A reviewer that is only turned off is kept
    # working too: `edmars review` uses it.
    from edmars import maintenance

    home = _set_up_reviewer(enabled=enabled)
    uv = FakeUv(real_run, home)
    monkeypatch.setattr(proc, "run", uv)
    monkeypatch.setenv("EDMARS_UV", "/fake/uv")
    assert lsar.verify(home)  # the new environment cannot load LSAR yet

    state = tmp_path / "state.txt"
    assert maintenance.after_install(state) == 0
    assert _state(state) == {"setup": "done", "reviewer": "repaired"}
    # uv installs into this Python, dry run first; dev tools and packages
    # EDM-ARS already has are left out, and the import check now passes.
    assert [c[:5] for c in uv.calls] == [["/fake/uv", "pip", "install", "--python", lsar.sys.executable]] * 2
    assert "--dry-run" in uv.calls[0] and "--dry-run" not in uv.calls[1]
    assert uv.requirements == [[f"{FAKE_DIST}>=1.0"]] * 2
    assert lsar.verify(home) == []
    record = json.loads((home / lsar.INSTALL_RECORD).read_text(encoding="utf-8"))
    assert record["requirements_reinstalled_at"] and record["unmet_pins"] != ["old"]
    assert "The automated reviewer is ready (1 of its packages installed again)." in capsys.readouterr().out


def test_after_install_reports_a_reviewer_it_could_not_repair(
    monkeypatch: pytest.MonkeyPatch, real_run: Any, tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from edmars import maintenance

    home = _set_up_reviewer(finished=False)
    monkeypatch.setattr(proc, "run", FakeUv(real_run, home, install_rc=2))
    monkeypatch.setenv("EDMARS_UV", "/fake/uv")
    state = tmp_path / "state.txt"
    assert maintenance.after_install(state) == 1
    assert _state(state) == {"setup": "partial", "reviewer": "failed"}
    said = capsys.readouterr()
    out = " ".join((said.out + said.err).split())
    assert "could not be made ready: Installing LSAR's Python packages failed" in out
    assert "network unreachable" in out and "`edmars setup reviewer`" in out

    # A package EDM-ARS uses would change: nothing is installed.
    import importlib.metadata

    installed = importlib.metadata.version("PyYAML")
    uv = FakeUv(real_run, home, dry_run_output=f" - pyyaml=={installed}\n + pyyaml==1.0\n")
    monkeypatch.setattr(proc, "run", uv)
    assert maintenance.after_install(state) == 1
    assert _state(state)["reviewer"] == "failed"
    assert len(uv.calls) == 1 and "--dry-run" in uv.calls[0]
    said = capsys.readouterr()
    assert "would change packages EDM-ARS already uses" in " ".join((said.out + said.err).split())

    # The LSAR folder itself is gone.
    import shutil

    shutil.rmtree(home)
    assert maintenance.after_install(state) == 1
    assert _state(state)["reviewer"] == "failed"


def test_after_install_without_a_reviewer_only_reports_setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    from edmars import maintenance
    from edmars import settings as settings_mod

    def no_process(args: list[str], **kwargs: Any) -> Any:
        raise AssertionError(f"nothing should run: {args}")

    monkeypatch.setattr(proc, "run", no_process)
    state = tmp_path / "state.txt"
    assert maintenance.after_install(state) == 0
    assert _state(state) == {"setup": "none", "reviewer": "none"}
    settings_mod.save(settings_mod.load())  # setup started, never finished
    assert maintenance.after_install(state) == 0
    assert _state(state) == {"setup": "partial", "reviewer": "none"}


def test_after_install_is_a_hidden_command(tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from edmars.cli import app

    cli = CliRunner()
    assert "after-install" not in cli.invoke(app, ["--help"]).output
    state = tmp_path / "state.txt"
    result = cli.invoke(app, ["after-install", "--plain", "--state-file", str(state)])
    assert result.exit_code == 0, result.output
    assert _state(state) == {"setup": "none", "reviewer": "none"}


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

    result = lsar.review_paper(run, {"lsar": {"home": str(home)}}, timeout_s=300)

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
        lsar.review_paper(empty, settings)
    with pytest.raises(lsar.LsarReviewError, match="not installed"):
        lsar.review_paper(empty, {"lsar": {}})

    run = _run_folder(tmp_path, venue=None)
    from edmars import secrets

    monkeypatch.setattr(secrets, "get_secret", lambda name: None)
    if hasattr(secrets, "child_secrets"):
        monkeypatch.setattr(secrets, "child_secrets", lambda names: {})
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(lsar.LsarReviewError, match="DeepSeek key"):
        lsar.review_paper(run, settings)

    monkeypatch.setattr(secrets, "get_secret", lambda name: "present")
    if hasattr(secrets, "child_secrets"):
        monkeypatch.setattr(secrets, "child_secrets",
                            lambda names: {"DEEPSEEK_API_KEY": "present"})
    monkeypatch.setenv("FAKE_LSAR_FAIL", "1")
    with pytest.raises(lsar.LsarReviewError, match="exit code 3"):
        lsar.review_paper(run, settings, timeout_s=300)


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


def test_edmars_review_shows_the_result_and_returns_an_exit_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from edmars import secrets, ui

    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(ui, "panel", lambda title, body: shown.append((title, body)))
    monkeypatch.setattr(ui, "ok", lambda msg: shown.append(("ok", msg)))
    monkeypatch.setattr(ui, "fail", lambda msg: shown.append(("fail", msg)))
    monkeypatch.setattr(secrets, "get_secret", lambda name: "present" if "DEEPSEEK" in name else None)
    if hasattr(secrets, "child_secrets"):
        monkeypatch.setattr(secrets, "child_secrets", lambda names: {"DEEPSEEK_API_KEY": "present"})
    home = _home(tmp_path)
    run = _run_folder(tmp_path, venue="EDM")

    assert lsar.review_run(run, {"lsar": {"home": str(home)}}) == 0
    body = next(b for t, b in shown if t == "Automated review (LSAR)")
    assert "Score: 6.8 / 10 (Accept)" in body
    assert "Benchmark for this venue: 6.30 (the score is at or above it)" in body
    assert "about 2 points" in body

    shown.clear()
    empty = tmp_path / "empty-run"
    empty.mkdir()
    assert lsar.review_run(empty, {"lsar": {"home": str(home)}}) == 1
    assert shown and shown[0][0] == "fail" and "no paper PDF" in shown[0][1]


def test_requirements_for_another_python_are_skipped(tmp_path: Path) -> None:
    (tmp_path / "requirements.txt").write_text(
        'tenacity>=8\nbackports.zoneinfo; python_version < "3.9"\n', encoding="utf-8")
    assert lsar.runtime_requirements(tmp_path) == ["tenacity>=8"]



def _stopped_study(tmp_path: Path, *, running: bool = False) -> Path:
    from tests.cli._run_support import alive_pid, log_lines, make_run, v2_status

    if running:
        return make_run(tmp_path, pdf=False, pid=alive_pid(),
                        log=log_lines((0, "Starting FORMULATING stage")))
    return make_run(tmp_path, pdf=False,
                    status=v2_status("ABORTED", released=False, reason_code="ABORTED",
                                     abort={"stage": "CRITIQUING", "code": "PRE_CRITIC_ABORT",
                                            "message": "pcc_07: promised a test", "resumable": False}))


@pytest.mark.parametrize("running", [False, True])
def test_edmars_review_of_a_study_without_a_paper_says_so_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, running: bool
) -> None:
    # The Mac study stopped at the automatic checks. `edmars review` first
    # announced "This usually takes 10-40 minutes", then advised checking
    # the PDF tools with `edmars doctor` and resuming a study that cannot
    # be resumed.
    from edmars import ui

    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(ui, "fail", lambda msg: shown.append(("fail", msg)))
    monkeypatch.setattr(ui, "info", lambda msg: shown.append(("info", msg)))
    entered: list[str] = []

    class _Status:
        def __init__(self, note: str) -> None:
            entered.append(note)

        def __enter__(self) -> None:
            return None

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(ui, "status", _Status)
    home = _home(tmp_path)
    run = _stopped_study(tmp_path / "studies", running=running)

    assert lsar.review_run(run, {"lsar": {"home": str(home)}}) == 1
    assert entered == []  # no "10-40 minutes" note before the check
    [(kind, message)] = shown
    assert kind == "fail" and "no paper PDF" in message
    assert "doctor" not in message and "edmars resume" not in message
    if running:
        assert "still running" in message and "edmars status" in message
    else:
        assert "stopped before its paper was written (Automatic checks stopped the study)" in message
        assert "edmars results" in message
