"""Dataset download/import/validation with tiny local fixtures and no network."""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import io
import json
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
    """A zip with fixed timestamps, so the same members give the same bytes."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            entry = zipfile.ZipInfo(name, date_time=(2020, 8, 17, 12, 0, 0))
            archive.writestr(entry, data, compress_type=zipfile.ZIP_DEFLATED)
    return buf.getvalue()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# A fake NCES release, shaped like the real one: the zip holds the
# numeric-code CSV (header CRLF, rows LF, some values quoted), and the
# conversion must produce the labelled CSV below byte for byte: labels in
# place of codes, the quote_all columns quoted, labels with a comma quoted,
# PSU renamed psu, and every row ending CRLF.
NUMERIC_CSV = (
    b'"STU_ID","PSU","X1SEX","X1RACE","X1TXMTSCOR","X3TGPAACAD","X4EVRATNDCLG","X2UNIV1"\r\n'
    b'"10001",-5,1,8,52.3,3.1,1,"11"\n'
    b'"10002",-5,2,5,48.0,2.7,0,"01"\n'
    b'"10003",-5,2,2,61.9,3.9,-8,"10"\n'
)
LABELLED_CSV = (
    b'"STU_ID","psu","X1SEX","X1RACE","X1TXMTSCOR","X3TGPAACAD","X4EVRATNDCLG","X2UNIV1"\r\n'
    b'"10001",-5,"Male","White, non-Hispanic",52.3,3.1,"Yes","BYR, F1R"\r\n'
    b'"10002",-5,"Female","Hispanic, race specified",48,2.7,"No","BYNR, F1R"\r\n'
    b'"10003",-5,"Female","Asian, non-Hispanic",61.9,3.9,"Unit non-response","BYR, F1NR"\r\n'
)
FAKE_TABLE: dict[str, Any] = {
    "format": 1,
    "header_renames": {"PSU": "psu"},
    "quote_all": ["STU_ID", "X1SEX", "X4EVRATNDCLG", "X2UNIV1"],
    "labels": {
        "X1SEX": {"1": "Male", "2": "Female"},
        "X1RACE": {"2": "Asian, non-Hispanic", "5": "Hispanic, race specified",
                   "8": "White, non-Hispanic"},
        "X1TXMTSCOR": {"48.0": "48"},
        "X4EVRATNDCLG": {"-8": "Unit non-response", "0": "No", "1": "Yes"},
        "X2UNIV1": {"01": "BYNR, F1R", "10": "BYR, F1NR", "11": "BYR, F1R"},
    },
    "numeric_csv_sha256": _sha(NUMERIC_CSV),
}
MEMBER = "HSLS_2017_PETS_SR_v1_0_CSV_Datasets/hsls_17_student_pets_sr_v1_0.csv"
NCES_ZIP = _zip({
    MEMBER: NUMERIC_CSV,
    "HSLS_2017_PETS_SR_v1_0_CSV_Datasets/hsls_09_school_v1_0.csv": b"SCH_ID\n1\n",
})


@pytest.fixture
def tiny_hsls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Treat a few-row labelled CSV as a full-size, unpinned HSLS file."""
    info = dataclasses.replace(datasets.CATALOG["hsls09_public"], min_bytes=0,
                               disk_needed_bytes=10_000)
    monkeypatch.setitem(datasets.CATALOG, "hsls09_public", info)
    monkeypatch.setitem(datasets.EXPECTED_SHA256, "hsls09_public", None)
    # A developer's own stand-in for the download must not reach these tests.
    monkeypatch.delenv("EDMARS_TEST_ZIP_HSLS09_PUBLIC", raising=False)


@pytest.fixture
def fake_nces(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch,
              tiny_hsls: None) -> bytes:
    """Make the catalog's HSLS:09 the fake release above; returns its zip.

    The zip's SHA-256, the label table and the labelled file's SHA-256 are
    pinned exactly as the real ones are.
    """
    table = tmp_path_factory.mktemp("table") / "fake.labels.json.gz"
    table.write_bytes(gzip.compress(json.dumps(FAKE_TABLE).encode("utf-8")))
    info = dataclasses.replace(datasets.CATALOG["hsls09_public"],
                               zip_sha256=_sha(NCES_ZIP), label_table=str(table))
    monkeypatch.setitem(datasets.CATALOG, "hsls09_public", info)
    monkeypatch.setitem(datasets.EXPECTED_SHA256, "hsls09_public", _sha(LABELLED_CSV))
    return NCES_ZIP


@pytest.fixture
def unpinned(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real HSLS:09 entry with trust-on-first-use instead of the pin."""
    monkeypatch.setitem(datasets.EXPECTED_SHA256, "hsls09_public", None)


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
    # It says where this file comes from and that install converts it,
    # not that the NCES zip holds a labelled CSV (it does not).
    assert "as it comes in the NCES zip" in check.detail
    assert "`edmars data install hsls09_public`" in check.detail
    assert "`edmars data import hsls09_public <path to the .zip>`" in check.detail
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


def test_download_converts_the_numeric_csv_to_the_labelled_file(
    fake_nces: bytes, tmp_path: Path
) -> None:
    settings_dict = _load_settings()
    session = serve_bytes(fake_nces)
    datasets.accept_terms("hsls09_public", settings_dict)
    phases: list[str] = []
    raw = datasets.raw_data_dir(settings_dict)

    path = datasets.download("hsls09_public", raw, lambda d, t, phase: phases.append(phase),
                             settings=settings_dict, session=session)

    assert path == raw / "hsls_17_student_pets_sr_v1_0.csv"
    assert path.read_bytes() == LABELLED_CSV  # byte for byte
    assert sorted(p.name for p in raw.iterdir()) == [
        "hsls_17_student_pets_sr_v1_0.csv", "hsls_17_student_pets_sr_v1_0.csv.edmars.json"
    ], "only the labelled file (and its record) is left: no numeric CSV, no zip, no .part"
    digest = _sha(LABELLED_CSV)
    assert settings_dict["datasets"]["hsls09_public"]["sha256"] == digest
    assert settings_dict["datasets"]["hsls09_public"]["terms_accepted_at"]
    # The zip is checked before it is converted, and the conversion is reported.
    assert phases.index("verify") < phases.index("convert")
    assert {"download", "verify", "convert"} == set(phases)
    assert session.calls[0]["url"] == datasets.CATALOG["hsls09_public"].url
    assert session.calls[0]["headers"]["User-Agent"].startswith("edm-ars-cli/")
    # persisted, not just mutated
    from edmars import settings as settings_mod

    assert settings_mod.load()["datasets"]["hsls09_public"]["sha256"] == digest
    assert datasets.status("hsls09_public", settings_dict).status == "ok"
    assert datasets.verify("hsls09_public", settings_dict).status == "ok"


def test_two_argument_progress_callbacks_work_too(fake_nces: bytes, tmp_path: Path) -> None:
    seen: list[tuple[int, Any]] = []
    datasets.download("hsls09_public", tmp_path / "raw", lambda d, t: seen.append((d, t)),
                      session=serve_bytes(fake_nces))
    assert seen and seen[-1][0] == seen[-1][1]
    # Without settings the record still lands in the sidecar ...
    sidecar = tmp_path / "raw" / "hsls_17_student_pets_sr_v1_0.csv.edmars.json"
    assert sidecar.is_file()


def test_download_resumes_with_range_after_a_dropped_connection(
    fake_nces: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_sleep(monkeypatch)
    session = serve_bytes(fake_nces, fail_first_after=len(fake_nces) // 2)
    path = datasets.download("hsls09_public", tmp_path, session=session)
    assert path.read_bytes() == LABELLED_CSV
    assert len(session.calls) == 2
    retry = session.calls[1]["headers"]
    assert retry["Range"] == f"bytes={len(fake_nces) // 2}-"
    assert retry["If-Range"] == '"v1"'


def test_download_restarts_when_the_server_ignores_range(
    fake_nces: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_sleep(monkeypatch)
    session = serve_bytes(fake_nces, fail_first_after=40, honour_range=False)
    path = datasets.download("hsls09_public", tmp_path, session=session)
    assert path.read_bytes() == LABELLED_CSV


def test_an_interrupted_download_resumes_on_the_next_call(
    fake_nces: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _no_sleep(monkeypatch)
    body = fake_nces
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
    assert path.read_bytes() == LABELLED_CSV
    assert session.calls[0]["headers"]["Range"] == "bytes=50-"


def test_an_html_page_instead_of_a_zip_is_refused(fake_nces: bytes, tmp_path: Path) -> None:
    session = serve_bytes(b"<html>Access denied by your network filter</html>")
    with pytest.raises(datasets.DatasetError, match="not a zip") as caught:
        datasets.download("hsls09_public", tmp_path, session=session)
    # The way round a filter: fetch the zip in a browser and import the zip
    # (the CSV inside it is the numeric one, which cannot be imported).
    assert "`edmars data import hsls09_public <the .zip>`" in str(caught.value)
    assert not (tmp_path / "hsls_17_student_pets_sr_v1_0.csv").exists()
    assert not list((tmp_path / ".downloads").glob("*.zip"))


def test_a_404_is_not_retried(fake_nces: bytes, tmp_path: Path) -> None:
    session = FakeSession(lambda u, h, p: FakeResponse(404, body=b"gone"))
    with pytest.raises(fetch.DownloadError, match="HTTP 404"):
        datasets.download("hsls09_public", tmp_path, session=session)
    assert len(session.calls) == 1


def test_a_zip_that_is_not_the_known_release_is_not_converted(
    fake_nces: bytes, tmp_path: Path
) -> None:
    # A new NCES release: same layout, one value changed. The label table was
    # made for the old release, so converting it could give a wrong file.
    changed = _zip({MEMBER: NUMERIC_CSV.replace(b"52.3", b"52.4")})
    raw = tmp_path / "raw"
    with pytest.raises(datasets.UnknownReleaseError) as caught:
        datasets.download("hsls09_public", raw, session=serve_bytes(changed))
    message = str(caught.value)
    assert "SHA-256 differs" in message and "did not convert" in message
    assert "The downloaded zip was deleted." in message
    assert "edmars data import hsls09_public <path to the .csv file>" in message
    assert "github.com/cgpan/edm-ars-public/issues" in message
    # Nothing unverified is left behind, and the unknown zip does not fill the disk.
    assert list(raw.rglob("*")) == []


def test_a_conversion_that_misses_the_pinned_hash_installs_nothing(
    fake_nces: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(datasets.EXPECTED_SHA256, "hsls09_public", "0" * 64)
    with pytest.raises(datasets.DatasetError, match="expected SHA-256"):
        datasets.download("hsls09_public", tmp_path, session=serve_bytes(fake_nces))
    assert not (tmp_path / "hsls_17_student_pets_sr_v1_0.csv").exists()
    assert not list(tmp_path.glob("*.part"))
    # The zip matched its pin, so it is kept: an EDM-ARS with a fixed table
    # can convert it without downloading it again.
    assert (tmp_path / ".downloads" / "HSLS_2017_PETS_SR_v1_0_CSV_Datasets.zip").is_file()


def test_a_zip_without_the_member_names_what_it_has(
    fake_nces: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = _zip({"readme.txt": b"hi", "other.csv": b"a,b\n1,2\n"})
    info = dataclasses.replace(datasets.CATALOG["hsls09_public"], zip_sha256=_sha(body))
    monkeypatch.setitem(datasets.CATALOG, "hsls09_public", info)
    with pytest.raises(datasets.DatasetError, match="other.csv"):
        datasets.download("hsls09_public", tmp_path, session=serve_bytes(body))
    assert not list(tmp_path.glob("*.csv*"))


def test_terms_must_be_accepted_when_settings_are_given(
    fake_nces: bytes, tmp_path: Path
) -> None:
    settings_dict = _load_settings()
    session = serve_bytes(fake_nces)
    with pytest.raises(datasets.DatasetError, match="terms"):
        datasets.download("hsls09_public", tmp_path, settings=settings_dict, session=session)
    assert session.calls == []
    assert datasets.terms_accepted("did_els_hsls_panel", settings_dict)


def test_the_terms_say_how_each_nces_file_is_used() -> None:
    hsls = " ".join((datasets.terms_text("hsls09_public") or "").split())
    assert "NCES" in hsls
    # HSLS:09 is converted, so its terms must not say it arrives unchanged.
    assert "straight from" not in hsls
    assert "converts the numeric codes" in hsls and "SHA-256" in hsls
    els = datasets.terms_text("els_2002") or ""
    assert "comes straight from the NCES website" in els


def test_not_enough_disk_space_stops_before_downloading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import shutil

    monkeypatch.setattr(shutil, "disk_usage",
                        lambda p: shutil._ntuple_diskusage(10**12, 10**12 - 1000, 1000))
    session = serve_bytes(NCES_ZIP)
    with pytest.raises(datasets.DatasetError, match="Not enough free disk space"):
        datasets.download("hsls09_public", tmp_path, session=session)
    assert session.calls == []


def test_an_installed_file_is_not_downloaded_again(
    fake_nces: bytes, tmp_path: Path
) -> None:
    settings_dict = _load_settings()
    datasets.accept_terms("hsls09_public", settings_dict)
    raw = datasets.raw_data_dir(settings_dict)
    datasets.download("hsls09_public", raw, settings=settings_dict, session=serve_bytes(fake_nces))
    again = serve_bytes(fake_nces)
    datasets.download("hsls09_public", raw, settings=settings_dict, session=again)
    assert again.calls == []


def test_installing_again_repairs_a_damaged_copy(fake_nces: bytes, tmp_path: Path) -> None:
    settings_dict = _load_settings()
    datasets.accept_terms("hsls09_public", settings_dict)
    raw = datasets.raw_data_dir(settings_dict)
    path = datasets.download("hsls09_public", raw, settings=settings_dict,
                             session=serve_bytes(fake_nces))
    damaged = LABELLED_CSV.replace(b"52.3", b"99.9")  # same size, different bytes
    path.write_bytes(damaged)
    check = datasets.verify("hsls09_public", settings_dict)
    assert check.status == "fail" and "expected SHA-256" in check.detail

    again = serve_bytes(fake_nces)
    datasets.download("hsls09_public", raw, settings=settings_dict, session=again)
    assert again.calls, "a damaged copy is fetched again"
    assert path.read_bytes() == LABELLED_CSV
    assert datasets.verify("hsls09_public", settings_dict).status == "ok"


def test_a_local_zip_can_stand_in_for_the_download_in_tests(
    fake_nces: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "local copy.zip"
    local.write_bytes(fake_nces)
    monkeypatch.setenv("EDMARS_TEST_ZIP_HSLS09_PUBLIC", str(local))
    settings_dict = _load_settings()
    datasets.accept_terms("hsls09_public", settings_dict)
    raw = datasets.raw_data_dir(settings_dict)
    session = serve_bytes(b"never requested")

    path = datasets.download("hsls09_public", raw, settings=settings_dict, session=session)

    assert session.calls == []
    assert path.read_bytes() == LABELLED_CSV
    assert settings_dict["datasets"]["hsls09_public"]["source"] == "local zip:local copy.zip"
    assert local.read_bytes() == fake_nces, "a zip EDM-ARS did not download is not deleted"
    assert not (raw / ".downloads").exists()


def test_a_local_zip_is_still_checked_against_the_pin(
    fake_nces: bytes, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "other.zip"
    local.write_bytes(_zip({MEMBER: LABELLED_CSV}))
    monkeypatch.setenv("EDMARS_TEST_ZIP_HSLS09_PUBLIC", str(local))
    with pytest.raises(datasets.UnknownReleaseError):
        datasets.download("hsls09_public", tmp_path / "raw", session=serve_bytes(b""))
    assert local.is_file()
    assert not (tmp_path / "raw" / "hsls_17_student_pets_sr_v1_0.csv").exists()


def test_a_local_zip_is_refused_where_nothing_pins_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "els.zip"
    local.write_bytes(_zip({"els_02_12_byf3pststu_v1_0.csv": b"x\n"}))
    monkeypatch.setenv("EDMARS_TEST_ZIP_ELS_2002", str(local))
    with pytest.raises(datasets.DatasetError, match="EDMARS_TEST_ZIP_ELS_2002 is set"):
        datasets.download("els_2002", tmp_path / "raw", session=serve_bytes(b""))
    monkeypatch.setenv("EDMARS_TEST_ZIP_HSLS09_PUBLIC", str(tmp_path / "missing.zip"))
    with pytest.raises(datasets.DatasetError, match="not a file"):
        datasets.download("hsls09_public", tmp_path / "raw", session=serve_bytes(b""))


REAL_ZIP_ENV = "EDMARS_REAL_HSLS_ZIP"


@pytest.mark.skipif(not os.environ.get(REAL_ZIP_ENV),
                    reason=f"set {REAL_ZIP_ENV} to the real NCES HSLS:09 zip to run this")
def test_the_real_nces_zip_installs_as_the_pinned_labelled_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end on the real 297 MB zip: `edmars data install`, no network.

    Writes the 2.0 GB labelled file into this test's EDMARS_HOME and takes
    about a minute.
    """
    from typer.testing import CliRunner

    from edmars.cli import app

    real = Path(os.environ[REAL_ZIP_ENV])
    monkeypatch.setenv("EDMARS_TEST_ZIP_HSLS09_PUBLIC", str(real))
    result = CliRunner().invoke(
        app, ["data", "install", "hsls09_public", "--accept-terms", "--plain"])
    assert result.exit_code == 0, result.output
    assert "converted 100%" in result.output
    settings_dict = _load_settings()
    path = datasets.expected_path("hsls09_public", settings_dict)
    assert path.stat().st_size == 1_998_219_907
    assert fetch.sha256_file(path) == datasets.EXPECTED_SHA256["hsls09_public"]
    record = settings_dict["datasets"]["hsls09_public"]
    assert record["sha256"] == datasets.EXPECTED_SHA256["hsls09_public"]
    assert record["terms_accepted_at"]
    assert real.is_file()
    assert datasets.validate_file("hsls09_public", path).status == "ok"


def test_els_is_still_extracted_as_it_comes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A dataset without a label table is unpacked, not converted."""
    info = dataclasses.replace(datasets.CATALOG["els_2002"], disk_needed_bytes=10_000)
    monkeypatch.setitem(datasets.CATALOG, "els_2002", info)
    csv_bytes = _csv(ELS_HEADER, [["1", "1", "-0.5", "2", "7", "3", "5", "50", "52"]])
    body = _zip({"ELS/els_02_12_byf3pststu_v1_0.csv": csv_bytes})
    phases: list[str] = []
    path = datasets.download("els_2002", tmp_path, lambda d, t, phase: phases.append(phase),
                             session=serve_bytes(body))
    assert path.read_bytes() == csv_bytes
    assert "extract" in phases and "convert" not in phases


def test_the_real_hsls_entry_is_pinned_to_the_checked_release() -> None:
    from edmars import relabel

    info = datasets.CATALOG["hsls09_public"]
    labelled = "b4400425b11294f7fc79f3a5f6a71a7b9bb9f77756abd3bb948a3b3b7ffa65d8"
    assert datasets.EXPECTED_SHA256["hsls09_public"] == labelled
    assert info.zip_sha256 == "770b2e64d509d8ed82f2ed1cf2a6983ebed969058f982e4cd22e12f4738e8639"
    table_file = relabel.table_path(str(info.label_table))
    # Package data: column, code and label metadata only, and small.
    assert table_file.parent.name == "data" and table_file.parent.parent.name == "edmars"
    assert table_file.stat().st_size < 100_000
    table = relabel.load_table(str(info.label_table))
    assert table.labelled_csv_sha256 == labelled
    assert table.numeric_csv_sha256 == (
        "987b609784978e273ac6821a665ae0cc72b6ce2edb2ff7543e1e82f8aec36a49")
    assert table.header_renames == {"PSU": "psu"}
    assert table.labels["X1SEX"] == {b"1": b"Male", b"2": b"Female", b"-9": b"Missing"}
    assert len(table.labels) == 3241
    assert sum(len(pairs) for pairs in table.labels.values()) == 16732
    assert {"STU_ID", "X1SEX", "X1RACE"} <= table.quote_all


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


def test_a_cut_short_import_is_never_shown_as_verified(tmp_path: Path, unpinned: None) -> None:
    settings_dict = _load_settings()
    src = tmp_path / "hsls fake.csv"
    src.write_bytes(_csv(HSLS_HEADER, HSLS_LABELLED))
    assert datasets.validate_file("hsls09_public", src).status == "warn"  # the import warns
    datasets.import_file("hsls09_public", src, settings_dict)
    assert settings_dict["datasets"]["hsls09_public"]["sha256"]  # a hash was recorded
    check = datasets.status("hsls09_public", settings_dict)
    assert check.status == "warn"
    assert "incomplete" in check.detail and "verified" not in check.detail
    assert check.fix == "edmars data install hsls09_public"
    assert datasets.verify("hsls09_public", settings_dict).status == "warn"


def test_importing_the_installed_file_itself_is_fine(tiny_hsls: None) -> None:
    settings_dict = _load_settings()
    dest = datasets.expected_path("hsls09_public", settings_dict)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(_csv(HSLS_HEADER, HSLS_LABELLED))
    assert datasets.import_file("hsls09_public", dest, settings_dict) == dest
    assert dest.read_bytes() == _csv(HSLS_HEADER, HSLS_LABELLED)


def test_the_nces_zip_can_be_imported_and_is_converted(fake_nces: bytes, tmp_path: Path) -> None:
    """A zip downloaded in a browser gives the same file as `data install`."""
    settings_dict = _load_settings()
    zip_file = tmp_path / "HSLS_2017_PETS_SR_v1_0_CSV_Datasets.zip"
    zip_file.write_bytes(fake_nces)

    dest = datasets.import_file("hsls09_public", zip_file, settings_dict)

    assert dest == datasets.expected_path("hsls09_public", settings_dict)
    assert dest.read_bytes() == LABELLED_CSV
    record = settings_dict["datasets"]["hsls09_public"]
    assert record["sha256"] == _sha(LABELLED_CSV)
    assert record["source"] == "import:HSLS_2017_PETS_SR_v1_0_CSV_Datasets.zip"
    assert zip_file.read_bytes() == fake_nces, "the user's own zip is left alone"
    assert datasets.status("hsls09_public", settings_dict).status == "ok"
    assert datasets.verify("hsls09_public", settings_dict).status == "ok"


def test_an_imported_zip_of_another_release_is_refused_and_kept(
    fake_nces: bytes, tmp_path: Path
) -> None:
    settings_dict = _load_settings()
    zip_file = tmp_path / "newer.zip"
    zip_file.write_bytes(_zip({MEMBER: NUMERIC_CSV + b'"10004",-5,1,8,50.0,3.0,1,"11"\n'}))
    with pytest.raises(datasets.UnknownReleaseError, match="did not convert") as caught:
        datasets.import_file("hsls09_public", zip_file, settings_dict)
    assert "was deleted" not in str(caught.value)
    assert zip_file.is_file()
    assert not datasets.expected_path("hsls09_public", settings_dict).exists()


def test_data_import_command_converts_the_nces_zip(fake_nces: bytes, tmp_path: Path) -> None:
    from typer.testing import CliRunner

    from edmars.cli import app

    zip_file = tmp_path / "HSLS_2017_PETS_SR_v1_0_CSV_Datasets.zip"
    zip_file.write_bytes(fake_nces)
    result = CliRunner().invoke(app, ["data", "import", "hsls09_public", str(zip_file), "--plain"])
    assert result.exit_code == 0, result.output
    assert datasets.expected_path("hsls09_public", _load_settings()).read_bytes() == LABELLED_CSV


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
    with pytest.raises(datasets.DatasetError, match="differs from the one EDM-ARS expects"):
        datasets.import_file("hsls09_public", src, settings_dict)
    check = datasets.verify("hsls09_public", settings_dict)
    assert check.status == "fail" and "expected SHA-256" in check.detail


def test_a_labelled_copy_other_than_the_pinned_one_is_placed_but_not_verified(
    tmp_path: Path,
) -> None:
    """With the real pin, a labelled CSV from elsewhere is usable but flagged."""
    settings_dict = _load_settings()
    src = tmp_path / "other export.csv"
    src.write_bytes(_csv(HSLS_HEADER, HSLS_LABELLED))
    with pytest.raises(datasets.DatasetError, match="not marked verified"):
        datasets.import_file("hsls09_public", src, settings_dict)
    assert datasets.expected_path("hsls09_public", settings_dict).is_file()
    assert "sha256" not in (settings_dict.get("datasets", {}).get("hsls09_public") or {})


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
