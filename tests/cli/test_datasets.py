"""Dataset download/import/validation with tiny local fixtures and no network."""

from __future__ import annotations

import dataclasses
import hashlib
import io
import os
import sys
import zipfile
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

from edmars import datasets, fetch  # noqa: E402

pytestmark = pytest.mark.usefixtures("edmars_home", "no_network")


def _load_settings() -> dict[str, Any]:
    from edmars import settings

    return settings.load()

HSLS_HEADER = ["STU_ID", "X1SEX", "X1RACE", "X1TXMTSCOR", "X3TGPAACAD", "X4EVRATNDCLG"]
HSLS_LABELLED = [
    ["10001", "Male", "White, non-Hispanic", "52.3", "3.1", "Yes"],
    ["10002", "Female", "Hispanic, race specified", "48.0", "2.7", "No"],
    ["10003", "Female", "Asian, non-Hispanic", "61.9", "3.9", "Unit non-response"],
]
HSLS_NUMERIC = [
    ["10001", "1", "8", "52.3", "3.1", "1"],
    ["10002", "2", "5", "48.0", "2.7", "0"],
]
ELS_HEADER = ["STU_ID", "BYSES1QU", "BYSES1", "BYSEX", "BYRACE", "BYPARED",
              "BYSTEXP", "BYTXMSTD", "F1TXMSTD"]


def _csv(header: list[str], rows: list[list[str]]) -> bytes:
    import csv

    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def _zip(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buf.getvalue()


@pytest.fixture
def tiny_hsls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat a few-row labelled CSV as a full-size HSLS file."""
    info = dataclasses.replace(datasets.CATALOG["hsls09_public"], min_bytes=0,
                               disk_needed_bytes=10_000)
    monkeypatch.setitem(datasets.CATALOG, "hsls09_public", info)


def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fetch, "_sleep", lambda s: None)


# ---------------------------------------------------------------------------
# validate_file
# ---------------------------------------------------------------------------


def test_labelled_hsls_passes_and_numeric_codes_fail(tmp_path: Path, tiny_hsls: None) -> None:
    good = tmp_path / "good.csv"
    good.write_bytes(_csv(HSLS_HEADER, HSLS_LABELLED))
    check = datasets.validate_file("hsls09_public", good)
    assert check.status == "ok", check.detail

    numeric = tmp_path / "numeric.csv"
    numeric.write_bytes(_csv(HSLS_HEADER, HSLS_NUMERIC))
    check = datasets.validate_file("hsls09_public", numeric)
    assert check.status == "fail"
    assert "numeric codes" in check.detail
    assert check.fix == "edmars data install hsls09_public"


def test_small_hsls_file_is_flagged_as_possibly_cut_short(tmp_path: Path) -> None:
    good = tmp_path / "good.csv"
    good.write_bytes(_csv(HSLS_HEADER, HSLS_LABELLED))
    check = datasets.validate_file("hsls09_public", good)
    assert check.status == "warn"
    assert "cut short" in check.detail


def test_missing_columns_are_named(tmp_path: Path) -> None:
    path = tmp_path / "other.csv"
    path.write_bytes(_csv(["STU_ID", "X1SEX"], [["1", "Male"]]))
    check = datasets.validate_file("hsls09_public", path)
    assert check.status == "fail"
    assert "X3TGPAACAD" in check.detail and "X4EVRATNDCLG" in check.detail


def test_els_expects_numeric_codes(tmp_path: Path) -> None:
    numeric = tmp_path / "els.csv"
    numeric.write_bytes(_csv(ELS_HEADER, [["1", "1", "-0.5", "2", "7", "3", "5", "50", "52"]]))
    assert datasets.validate_file("els_2002", numeric).status == "ok"
    labelled = tmp_path / "els_labels.csv"
    labelled.write_bytes(_csv(ELS_HEADER, [["1", "1", "-0.5", "Female", "White", "3", "5",
                                            "50", "52"]]))
    check = datasets.validate_file("els_2002", labelled)
    assert check.status == "fail"
    assert "text labels" in check.detail


def test_missing_file_folder_and_non_csv(tmp_path: Path) -> None:
    assert datasets.validate_file("hsls09_public", tmp_path / "nope.csv").status == "fail"
    assert datasets.validate_file("hsls09_public", tmp_path).status == "fail"
    junk = tmp_path / "junk.csv"
    junk.write_bytes(b"\x00\x01binary")
    assert datasets.validate_file("hsls09_public", junk).status == "fail"


def test_header_only_file_fails(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_bytes(_csv(HSLS_HEADER, []))
    assert datasets.validate_file("hsls09_public", path).status == "fail"


def test_unknown_dataset_is_a_dataset_error(tmp_path: Path) -> None:
    with pytest.raises(datasets.DatasetError):
        datasets.validate_file("pisa", tmp_path / "x.csv")


# ---------------------------------------------------------------------------
# download
# ---------------------------------------------------------------------------


def _hsls_zip(rows: list[list[str]] | None = None) -> tuple[bytes, bytes]:
    csv_bytes = _csv(HSLS_HEADER, rows or HSLS_LABELLED)
    body = _zip({
        "HSLS_2017_PETS_SR_v1_0_CSV_Datasets/hsls_17_student_pets_sr_v1_0.csv": csv_bytes,
        "HSLS_2017_PETS_SR_v1_0_CSV_Datasets/hsls_09_school_v1_0.csv": b"SCH_ID\n1\n",
    })
    return body, csv_bytes


def test_download_extracts_only_the_member_and_records_the_hash(
    tiny_hsls: None, tmp_path: Path
) -> None:
    settings_dict = _load_settings()
    body, csv_bytes = _hsls_zip()
    session = serve_bytes(body)
    datasets.accept_terms("hsls09_public", settings_dict)
    phases: list[str] = []
    raw = datasets.raw_data_dir(settings_dict)

    path = datasets.download("hsls09_public", raw, lambda d, t, phase: phases.append(phase),
                             settings=settings_dict, session=session)

    assert path == raw / "hsls_17_student_pets_sr_v1_0.csv"
    assert path.read_bytes() == csv_bytes
    assert not (raw / "hsls_09_school_v1_0.csv").exists()
    assert not (raw / ".downloads").exists(), "the zip is removed after extraction"
    digest = hashlib.sha256(csv_bytes).hexdigest()
    assert settings_dict["datasets"]["hsls09_public"]["sha256"] == digest
    assert settings_dict["datasets"]["hsls09_public"]["terms_accepted_at"]
    assert {"download", "extract", "verify"} <= set(phases)
    assert session.calls[0]["url"] == datasets.CATALOG["hsls09_public"].url
    assert session.calls[0]["headers"]["User-Agent"].startswith("edm-ars-cli/")
    # persisted, not just mutated
    from edmars import settings as settings_mod

    assert settings_mod.load()["datasets"]["hsls09_public"]["sha256"] == digest
    assert datasets.status("hsls09_public", settings_dict).status == "ok"


def test_two_argument_progress_callbacks_work_too(tiny_hsls: None, tmp_path: Path) -> None:
    body, _ = _hsls_zip()
    seen: list[tuple[int, Any]] = []
    datasets.download("hsls09_public", tmp_path / "raw", lambda d, t: seen.append((d, t)),
                      session=serve_bytes(body))
    assert seen and seen[-1][0] == seen[-1][1]
    # Without settings the record still lands in the sidecar ...
    sidecar = tmp_path / "raw" / "hsls_17_student_pets_sr_v1_0.csv.edmars.json"
    assert sidecar.is_file()


def test_download_resumes_with_range_after_a_dropped_connection(
    tiny_hsls: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_sleep(monkeypatch)
    body, csv_bytes = _hsls_zip()
    session = serve_bytes(body, fail_first_after=len(body) // 2)
    path = datasets.download("hsls09_public", tmp_path, session=session)
    assert path.read_bytes() == csv_bytes
    assert len(session.calls) == 2
    retry = session.calls[1]["headers"]
    assert retry["Range"] == f"bytes={len(body) // 2}-"
    assert retry["If-Range"] == '"v1"'


def test_download_restarts_when_the_server_ignores_range(
    tiny_hsls: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_sleep(monkeypatch)
    body, csv_bytes = _hsls_zip()
    session = serve_bytes(body, fail_first_after=40, honour_range=False)
    path = datasets.download("hsls09_public", tmp_path, session=session)
    assert path.read_bytes() == csv_bytes


def test_an_interrupted_download_resumes_on_the_next_call(
    tiny_hsls: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_sleep(monkeypatch)
    body, csv_bytes = _hsls_zip()
    dropping = FakeSession(lambda u, h, p: FakeResponse(
        200, body=body, fail_after=50, headers={"Content-Length": str(len(body)), "ETag": '"v1"'}))
    with pytest.raises(fetch.DownloadError, match="resume"):
        fetch.download_file(datasets.CATALOG["hsls09_public"].url,
                            tmp_path / ".downloads" / "HSLS_2017_PETS_SR_v1_0_CSV_Datasets.zip",
                            session=dropping, attempts=2)
    part = tmp_path / ".downloads" / "HSLS_2017_PETS_SR_v1_0_CSV_Datasets.zip.part"
    assert part.is_file() and part.stat().st_size == 50

    session = serve_bytes(body)
    path = datasets.download("hsls09_public", tmp_path, session=session)
    assert path.read_bytes() == csv_bytes
    assert session.calls[0]["headers"]["Range"] == "bytes=50-"


def test_an_html_page_instead_of_a_zip_is_refused(tiny_hsls: None, tmp_path: Path) -> None:
    session = serve_bytes(b"<html>Access denied by your network filter</html>")
    with pytest.raises(datasets.DatasetError, match="not a zip"):
        datasets.download("hsls09_public", tmp_path, session=session)
    assert not (tmp_path / "hsls_17_student_pets_sr_v1_0.csv").exists()
    assert not list((tmp_path / ".downloads").glob("*.zip"))


def test_a_404_is_not_retried(tiny_hsls: None, tmp_path: Path) -> None:
    session = FakeSession(lambda u, h, p: FakeResponse(404, body=b"gone"))
    with pytest.raises(fetch.DownloadError, match="HTTP 404"):
        datasets.download("hsls09_public", tmp_path, session=session)
    assert len(session.calls) == 1


def test_the_wrong_file_inside_the_zip_is_rejected(tiny_hsls: None, tmp_path: Path) -> None:
    body, _ = _hsls_zip(HSLS_NUMERIC)
    with pytest.raises(datasets.DatasetError, match="numeric codes"):
        datasets.download("hsls09_public", tmp_path, session=serve_bytes(body))
    assert not (tmp_path / "hsls_17_student_pets_sr_v1_0.csv").exists()
    assert (tmp_path / "hsls_17_student_pets_sr_v1_0.csv.rejected").exists()


def test_a_zip_without_the_member_names_what_it_has(tiny_hsls: None, tmp_path: Path) -> None:
    body = _zip({"readme.txt": b"hi", "other.csv": b"a,b\n1,2\n"})
    with pytest.raises(datasets.DatasetError, match="other.csv"):
        datasets.download("hsls09_public", tmp_path, session=serve_bytes(body))


def test_terms_must_be_accepted_when_settings_are_given(
    tiny_hsls: None, tmp_path: Path
) -> None:
    settings_dict = _load_settings()
    session = serve_bytes(_hsls_zip()[0])
    with pytest.raises(datasets.DatasetError, match="terms"):
        datasets.download("hsls09_public", tmp_path, settings=settings_dict, session=session)
    assert session.calls == []
    assert "NCES" in (datasets.terms_text("hsls09_public") or "")
    assert datasets.terms_accepted("did_els_hsls_panel", settings_dict)


def test_not_enough_disk_space_stops_before_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    monkeypatch.setattr(shutil, "disk_usage",
                        lambda p: shutil._ntuple_diskusage(10**12, 10**12 - 1000, 1000))
    session = serve_bytes(_hsls_zip()[0])
    with pytest.raises(datasets.DatasetError, match="Not enough free disk space"):
        datasets.download("hsls09_public", tmp_path, session=session)
    assert session.calls == []


def test_an_installed_file_is_not_downloaded_again(
    tiny_hsls: None, tmp_path: Path
) -> None:
    settings_dict = _load_settings()
    body, _ = _hsls_zip()
    datasets.accept_terms("hsls09_public", settings_dict)
    raw = datasets.raw_data_dir(settings_dict)
    datasets.download("hsls09_public", raw, settings=settings_dict, session=serve_bytes(body))
    again = serve_bytes(body)
    datasets.download("hsls09_public", raw, settings=settings_dict, session=again)
    assert again.calls == []


def test_installing_again_repairs_a_damaged_copy(tiny_hsls: None, tmp_path: Path) -> None:
    settings_dict = _load_settings()
    body, csv_bytes = _hsls_zip()
    datasets.accept_terms("hsls09_public", settings_dict)
    raw = datasets.raw_data_dir(settings_dict)
    path = datasets.download("hsls09_public", raw, settings=settings_dict,
                             session=serve_bytes(body))
    damaged = csv_bytes.replace(b"52.3", b"99.9")  # same size, different bytes
    path.write_bytes(damaged)
    assert datasets.verify("hsls09_public", settings_dict).status == "fail"

    again = serve_bytes(body)
    datasets.download("hsls09_public", raw, settings=settings_dict, session=again)
    assert again.calls, "a damaged copy is fetched again"
    assert path.read_bytes() == csv_bytes
    assert datasets.verify("hsls09_public", settings_dict).status == "ok"


def test_a_big_file_without_line_breaks_is_read_only_in_part(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(datasets, "HEAD_BYTES", 1024)
    blob = tmp_path / "data.csv"
    blob.write_bytes(b"X1SEX," * 5000)  # 30 KB, one endless "line"
    check = datasets.validate_file("hsls09_public", blob)
    assert check.status == "fail"


def test_manual_and_derived_datasets_cannot_be_downloaded(tmp_path: Path) -> None:
    with pytest.raises(datasets.DatasetError, match="import"):
        datasets.download("assistments_0910", tmp_path)
    with pytest.raises(datasets.DatasetError, match="install did_els_hsls_panel"):
        datasets.download("did_els_hsls_panel", tmp_path)


# ---------------------------------------------------------------------------
# import, status, verify
# ---------------------------------------------------------------------------


def test_import_places_the_file_and_trust_on_first_use_catches_changes(
    tiny_hsls: None, tmp_path: Path
) -> None:
    settings_dict = _load_settings()
    src = tmp_path / "my copy.csv"
    src.write_bytes(_csv(HSLS_HEADER, HSLS_LABELLED))
    assert datasets.status("hsls09_public", settings_dict).status == "warn"  # not installed

    dest = datasets.import_file("hsls09_public", src, settings_dict)
    assert dest == datasets.expected_path("hsls09_public", settings_dict)
    assert dest.read_bytes() == src.read_bytes()
    record = settings_dict["datasets"]["hsls09_public"]
    assert record["sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    assert record["source"] == "import:my copy.csv"
    assert datasets.status("hsls09_public", settings_dict).status == "ok"
    assert datasets.verify("hsls09_public", settings_dict).status == "ok"

    # Same size, different bytes: only a full verify can tell.
    data = bytearray(dest.read_bytes())
    data[-3:-1] = b"No"[::-1]
    os.remove(dest)
    dest.write_bytes(bytes(data))
    assert datasets.status("hsls09_public", settings_dict).status == "ok"
    check = datasets.verify("hsls09_public", settings_dict)
    assert check.status == "fail"
    assert "changed since it was first verified" in check.detail

    # A different size is visible to the cheap status check.
    dest.write_bytes(bytes(data) + b"10004,Male,Other,1,1,No\n")
    assert datasets.status("hsls09_public", settings_dict).status == "warn"


def test_importing_the_installed_file_itself_is_fine(tiny_hsls: None) -> None:
    settings_dict = _load_settings()
    dest = datasets.expected_path("hsls09_public", settings_dict)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_csv(HSLS_HEADER, HSLS_LABELLED))
    assert datasets.import_file("hsls09_public", dest, settings_dict) == dest
    assert dest.read_bytes() == _csv(HSLS_HEADER, HSLS_LABELLED)


def test_import_refuses_an_invalid_file(tmp_path: Path) -> None:
    settings_dict = _load_settings()
    src = tmp_path / "numeric.csv"
    src.write_bytes(_csv(HSLS_HEADER, HSLS_NUMERIC))
    with pytest.raises(datasets.DatasetError, match="numeric codes"):
        datasets.import_file("hsls09_public", src, settings_dict)
    assert not datasets.expected_path("hsls09_public", settings_dict).exists()


def test_pinned_hash_mismatch_is_reported(
    tiny_hsls: None, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_dict = _load_settings()
    monkeypatch.setitem(datasets.EXPECTED_SHA256, "hsls09_public", "0" * 64)
    src = tmp_path / "copy.csv"
    src.write_bytes(_csv(HSLS_HEADER, HSLS_LABELLED))
    with pytest.raises(datasets.DatasetError, match="published SHA-256"):
        datasets.import_file("hsls09_public", src, settings_dict)
    assert datasets.verify("hsls09_public", settings_dict).status == "fail"


def test_status_of_missing_optional_datasets() -> None:
    settings_dict = _load_settings()
    assert datasets.status("assistments_0910", settings_dict).status == "info"
    panel = datasets.status("did_els_hsls_panel", settings_dict)
    assert panel.status == "info"
    assert "HSLS:09" in panel.detail and "ELS:2002" in panel.detail
    els = datasets.status("els_2002", settings_dict)
    assert els.status == "info" and els.fix == "edmars data install els_2002"
    assert [c.name for c in datasets.catalog_checks(settings_dict)] == [
        info.label for info in datasets.CATALOG.values()]


def test_raw_data_dir_matches_the_harmonizer_layout() -> None:
    settings_dict = _load_settings()
    raw = datasets.raw_data_dir(settings_dict)
    assert raw.parts[-2:] == ("data", "raw")
    for info in datasets.CATALOG.values():
        from src.dataset_adapter import create_dataset_adapter

        assert info.filename == create_dataset_adapter(info.name).get_raw_data_filename()


# ---------------------------------------------------------------------------
# DiD panel
# ---------------------------------------------------------------------------


def _install_sources(raw: Path) -> None:
    hsls_rows = []
    labels = ["First quintile (lowest)", "Fifth quintile (highest)"]
    races = ["White, non-Hispanic", "Black/African-American, non-Hispanic"]
    for i in range(12):
        hsls_rows.append([
            str(20000 + i), labels[i % 2], f"{(i % 5) - 1.5:.2f}",
            "Female" if i % 3 else "Male", races[i % 2],
            "Bachelor's degree" if i % 2 else "High school diploma or GED",
            "Complete a Bachelor's degree" if i % 4 else "Start an Associate's degree",
            f"{40 + i * 1.7:.1f}", f"{42 + i * 1.3:.1f}", "3.0", "Yes",
        ])
    header = ["STU_ID", "X1SESQ5", "X1SES", "X1SEX", "X1RACE", "X1PAREDU",
              "X1STUEDEXPCT", "X1TXMTSCOR", "X2TXMTSCOR", "X3TGPAACAD", "X4EVRATNDCLG"]
    (raw / "hsls_17_student_pets_sr_v1_0.csv").write_bytes(_csv(header, hsls_rows))
    els_rows = []
    for i in range(12):
        els_rows.append([str(i), "1" if i % 2 else "4", f"{(i % 5) - 1.2:.2f}",
                         str(1 + i % 2), str([7, 3, 4, 2][i % 4]), str(1 + i % 8),
                         str(1 + i % 7), f"{45 + i:.1f}", f"{47 + i:.1f}"])
    (raw / "els_2002").mkdir(parents=True, exist_ok=True)
    (raw / "els_2002" / "els_02_12_byf3pststu_v1_0.csv").write_bytes(_csv(ELS_HEADER, els_rows))


def test_build_did_panel_runs_the_real_harmonizer_against_the_user_data_dir() -> None:
    settings_dict = _load_settings()
    raw = datasets.raw_data_dir(settings_dict)
    raw.mkdir(parents=True, exist_ok=True)
    _install_sources(raw)

    panel = datasets.build_did_panel(settings_dict, timeout_s=600)

    assert panel == raw / "did_els_hsls_panel" / "panel.csv"
    header = panel.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert tuple(header) == datasets.PANEL_COLUMNS
    assert len(panel.read_text(encoding="utf-8").splitlines()) == 1 + 24
    assert settings_dict["datasets"]["did_els_hsls_panel"]["sha256"]
    assert datasets.status("did_els_hsls_panel", settings_dict).status == "ok"
    # the app checkout itself was not written to
    from edmars import paths

    assert not (Path(paths.app_root()) / "data" / "raw" / "did_els_hsls_panel"
                / "panel.csv.previous").exists()


def test_build_did_panel_needs_both_sources() -> None:
    settings_dict = _load_settings()
    with pytest.raises(datasets.DatasetError, match="install hsls09_public"):
        datasets.build_did_panel(settings_dict)


def test_a_failed_build_keeps_the_previous_panel(
    monkeypatch: pytest.MonkeyPatch
) -> None:
    settings_dict = _load_settings()
    raw = datasets.raw_data_dir(settings_dict)
    raw.mkdir(parents=True, exist_ok=True)
    _install_sources(raw)
    old = raw / "did_els_hsls_panel" / "panel.csv"
    old.parent.mkdir(parents=True, exist_ok=True)
    old.write_bytes(_csv(list(datasets.PANEL_COLUMNS), [["0"] * 9]))
    from edmars import proc

    calls: list[dict[str, Any]] = []

    def fake_run(args: list[str], **kwargs: Any) -> Any:
        calls.append({"args": args, **kwargs})
        (raw / "did_els_hsls_panel" / "panel.csv").write_text("partial", encoding="utf-8")
        return completed(args, 1, "", "KeyError: 'X1SESQ5'")

    monkeypatch.setattr(proc, "run", fake_run)
    with pytest.raises(datasets.DatasetError, match="exit code 1"):
        datasets.build_did_panel(settings_dict)
    assert old.read_bytes() == _csv(list(datasets.PANEL_COLUMNS), [["0"] * 9])
    args = calls[0]["args"]
    assert args[0] == sys.executable and args[1] == "-c"
    assert args[3].endswith("harmonize_els_hsls.py")
    assert Path(args[4]) / "data" / "raw" == raw
    env = calls[0]["env"]
    assert not any("API_KEY" in k for k in env)
    assert env["PYTHONUTF8"] == "1"
