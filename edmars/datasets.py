"""Dataset catalog: download, import, validate and verify public-use files.

Every dataset lives in ONE raw-data folder (``raw_data_dir(settings)``)
under the file name the pipeline's dataset adapter expects
(``src/dataset_adapter.py``), because the run config points
``paths.raw_data`` at that folder and the adapters append their own
relative file names.

Verification is trust-on-first-use unless a SHA-256 is pinned in
``EXPECTED_SHA256``: the first time a file passes validation its hash is
recorded (in settings, and in a ``<file>.edmars.json`` sidecar so the
record survives a caller that forgot to save settings), and
``verify()`` later tells the user when the bytes changed.

HSLS:09 is not used as it comes: the NCES zip holds the numeric-code
CSV, and the pipeline needs the labelled one. For a dataset with a
``label_table`` the zip's SHA-256 is checked first, the CSV inside it is
converted while it is unpacked (``edmars.relabel``), and the result must
match the pinned SHA-256 of the labelled file; otherwise nothing is
installed.

Nothing here touches the network except :func:`download`, which the
caller runs only after the user accepted the dataset's terms.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import sys
import zipfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Literal

from edmars import fetch, relabel
from edmars.fetch import ProgressFn, emit_progress, human_bytes
from edmars.model import Check

#: Pinned SHA-256 of each installed CSV. ``None`` means trust-on-first-use:
#: the hash of the first file that passes validation is recorded and later
#: files are compared against it. Pin a value only after checking it
#: against a download made on a second machine.
EXPECTED_SHA256: dict[str, str | None] = {
    # The labelled CSV that edmars.relabel makes from the NCES zip. Checked
    # 2026-09-26: the conversion of a fresh NCES download reproduced the
    # labelled copy the pipeline was built on (March 2026) byte for byte.
    "hsls09_public": "b4400425b11294f7fc79f3a5f6a71a7b9bb9f77756abd3bb948a3b3b7ffa65d8",
    "els_2002": None,
}

_ISSUES_URL = "https://github.com/cgpan/edm-ars-public/issues"

_NCES_TERMS_COMMON = (
    "This is a public-use file published by the National Center for "
    "Education Statistics (NCES). By downloading it you agree to use it for "
    "statistical research only, to make no attempt to identify any "
    "individual student, school or family, and to cite NCES as the source "
    "in anything you publish. EDM-ARS is not affiliated with or endorsed by "
    "NCES or IES"
)

#: Summary of the NCES public-use terms, shown before an NCES download
#: that is installed as it comes.
NCES_TERMS = _NCES_TERMS_COMMON + "; the file comes straight from the NCES website."

#: The same terms for a file EDM-ARS converts (HSLS:09).
NCES_TERMS_CONVERTED = _NCES_TERMS_COMMON + (
    ". EDM-ARS downloads the file from the NCES website and converts the "
    "numeric codes in it to their text labels (for example 1 to 'Male'), "
    "the labelled form EDM-ARS expects. The conversion is checked: the "
    "result must match a known SHA-256 fingerprint, or nothing is installed."
)


@dataclass(frozen=True)
class DatasetInfo:
    """What the CLI knows about one dataset."""

    name: str
    label: str
    #: Path relative to the raw-data folder, exactly as the adapter expects.
    filename: str
    #: "download" (from ``url``), "derived" (built locally) or "manual".
    source: Literal["download", "derived", "manual"]
    description: str
    url: str | None = None
    #: Base name of the CSV inside the zip at ``url``.
    member: str | None = None
    #: Pinned SHA-256 of the zip at ``url``. Required for ``label_table``.
    zip_sha256: str | None = None
    #: Bundled code-to-label table (``edmars/data``) that turns the numeric
    #: ``member`` into the labelled file; None = the member is used as is.
    label_table: str | None = None
    download_size: str = ""
    #: Free space needed for the download and the extracted file together.
    disk_needed_bytes: int = 0
    #: A valid file is at least this big; smaller files get a warning.
    min_bytes: int = 0
    required_columns: tuple[str, ...] = ()
    #: "labels" = text labels expected; "codes" = numeric codes expected.
    value_format: Literal["labels", "codes", "any"] = "any"
    #: Categorical columns whose first values tell labels from codes.
    format_probe: tuple[str, ...] = ()
    experimental: bool = False
    recommended: bool = False
    terms: str | None = None
    homepage: str | None = None
    #: Datasets that must be installed first ("derived" only).
    needs: tuple[str, ...] = field(default_factory=tuple)


#: Columns of the harmonized DiD panel (scripts/harmonize_els_hsls.py).
PANEL_COLUMNS: tuple[str, ...] = (
    "cohort", "low_ses", "female", "race5", "pared3", "expect_ba",
    "ses_std", "rank_base", "rank_follow",
)

CATALOG: dict[str, DatasetInfo] = {
    "hsls09_public": DatasetInfo(
        name="hsls09_public",
        label="HSLS:09 (High School Longitudinal Study of 2009)",
        filename="hsls_17_student_pets_sr_v1_0.csv",
        source="download",
        description=(
            "23,503 students followed from 9th grade (2009) into college "
            "and work; the public-use file from NCES, converted to the "
            "labelled CSV EDM-ARS expects."
        ),
        url="https://nces.ed.gov/EDAT/Data/Zip/HSLS_2017_PETS_SR_v1_0_CSV_Datasets.zip",
        member="hsls_17_student_pets_sr_v1_0.csv",
        # 296,995,863 bytes; the numeric-code CSV inside is 888,168,765
        # bytes, SHA-256 987b6097... (recorded in the label table).
        zip_sha256="770b2e64d509d8ed82f2ed1cf2a6983ebed969058f982e4cd22e12f4738e8639",
        label_table="hsls09_public.labels.json.gz",
        download_size="about 300 MB to download, about 2.0 GB once converted",
        # The zip (0.3 GB) and the labelled file (2.0 GB) side by side; the
        # numeric CSV is converted as it is unpacked and never written.
        disk_needed_bytes=2_600_000_000,
        min_bytes=1_000_000_000,
        required_columns=("X1SEX", "X1RACE", "X3TGPAACAD", "X4EVRATNDCLG"),
        value_format="labels",
        format_probe=("X1SEX", "X1RACE"),
        recommended=True,
        terms=NCES_TERMS_CONVERTED,
        homepage="https://nces.ed.gov/surveys/hsls09/",
    ),
    "els_2002": DatasetInfo(
        name="els_2002",
        label="ELS:2002 (Education Longitudinal Study of 2002)",
        filename="els_2002/els_02_12_byf3pststu_v1_0.csv",
        source="download",
        description=(
            "16,197 students followed from 10th grade (2002); numeric-code "
            "public-use CSV. Needed for the cross-cohort gap-change design."
        ),
        url=(
            "https://nces.ed.gov/EDAT/Data/Zip/"
            "ELS_2002-12_PETS_v1_0_Student_CSV_Datasets.zip"
        ),
        member="els_02_12_byf3pststu_v1_0.csv",
        download_size="about 18 MB to download",
        disk_needed_bytes=600_000_000,
        required_columns=(
            "BYSES1QU", "BYSES1", "BYSEX", "BYRACE", "BYPARED",
            "BYSTEXP", "BYTXMSTD", "F1TXMSTD",
        ),
        value_format="codes",
        format_probe=("BYSEX", "BYRACE"),
        # The archive's contents have not been checked against a fresh
        # download yet, so the CLI labels it until someone has.
        experimental=True,
        terms=NCES_TERMS,
        homepage="https://nces.ed.gov/surveys/els2002/",
    ),
    "did_els_hsls_panel": DatasetInfo(
        name="did_els_hsls_panel",
        label="ELS x HSLS cross-cohort panel",
        filename="did_els_hsls_panel/panel.csv",
        source="derived",
        description=(
            "Built on your computer from the ELS:2002 and HSLS:09 files by "
            "scripts/harmonize_els_hsls.py; used by gap-change studies."
        ),
        required_columns=PANEL_COLUMNS,
        value_format="any",
        needs=("hsls09_public", "els_2002"),
    ),
    "assistments_0910": DatasetInfo(
        name="assistments_0910",
        label="ASSISTments 2009-10 skill builder",
        filename="assistments_0910/skill_builder_0910.csv",
        source="manual",
        description=(
            "Problem-attempt log data. Automatic download is coming later; "
            "for now download it yourself and import it."
        ),
        required_columns=("user_id", "problem_id", "original", "correct", "skill_id"),
        value_format="any",
        homepage="https://sites.google.com/site/assistmentsdata/",
    ),
}

_NUMERIC = re.compile(r"^\s*-?\d+(?:\.\d+)?\s*$")
_SIDECAR_SUFFIX = ".edmars.json"
_SECRET_ENV_HINTS = ("API_KEY", "TOKEN", "SECRET", "PASSWORD")


class DatasetError(fetch.UserFacingError):
    """A dataset could not be downloaded, imported or built."""


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _info(name: str) -> DatasetInfo:
    try:
        return CATALOG[name]
    except KeyError:
        known = ", ".join(sorted(CATALOG))
        raise DatasetError(f"Unknown dataset {name!r}. Known datasets: {known}.") from None


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )


def _settings_get(settings: dict[str, Any], dotted: str, default: Any = None) -> Any:
    from edmars import settings as settings_mod

    return settings_mod.get(settings, dotted, default)


def _settings_set(settings: dict[str, Any], dotted: str, value: Any) -> None:
    from edmars import settings as settings_mod

    settings_mod.set_(settings, dotted, value)


def _save(settings: dict[str, Any]) -> None:
    from edmars import settings as settings_mod

    settings_mod.save(settings)


def get_command(name: str) -> str:
    """The command that puts dataset ``name`` on this computer.

    ``edmars data install`` for what EDM-ARS can download or build; for a
    dataset it cannot download yet (ASSISTments), ``install`` only says
    "coming later", so the way in is ``edmars data import``.
    """
    info = CATALOG.get(name)
    if info is not None and info.source == "manual":
        return f"edmars data import {name} <path to the .csv file>"
    return f"edmars data install {name}"


def raw_data_dir(settings: dict[str, Any] | None = None) -> Path:
    """The one folder the pipeline reads raw data from (``paths.raw_data``).

    ``<user data dir>/data/raw``. The ``data/raw`` tail is deliberate: it is
    the layout ``scripts/harmonize_els_hsls.py`` expects under its root.
    """
    from edmars import paths

    return Path(paths.data_dir()) / "data" / "raw"


def expected_path(name: str, settings: dict[str, Any] | None = None) -> Path:
    """Where the pipeline will look for dataset ``name``."""
    return raw_data_dir(settings) / _info(name).filename


def terms_text(name: str) -> str | None:
    """Terms summary to show before downloading ``name`` (None if none)."""
    return _info(name).terms


def terms_accepted(name: str, settings: dict[str, Any]) -> bool:
    """Whether the user accepted the terms for ``name`` (always True if none)."""
    if _info(name).terms is None:
        return True
    return bool(_settings_get(settings, f"datasets.{name}.terms_accepted_at"))


def accept_terms(name: str, settings: dict[str, Any]) -> None:
    """Record that the user accepted ``name``'s terms (timestamp, UTC)."""
    _info(name)
    _settings_set(settings, f"datasets.{name}.terms_accepted_at", _now())


def _sidecar_path(path: Path) -> Path:
    return path.with_name(path.name + _SIDECAR_SUFFIX)


def _read_sidecar(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(_sidecar_path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _write_sidecar(path: Path, record: dict[str, Any]) -> None:
    try:
        _sidecar_path(path).write_text(json.dumps(record, indent=2), encoding="utf-8")
    except OSError:
        pass


def _record(name: str, settings: dict[str, Any] | None, path: Path) -> dict[str, Any]:
    """The verification record for ``name`` (settings first, then sidecar)."""
    rec: dict[str, Any] = {}
    if settings is not None:
        value = _settings_get(settings, f"datasets.{name}", None)
        if isinstance(value, dict):
            rec = dict(value)
    if not rec.get("sha256"):
        side = _read_sidecar(path)
        for key in ("sha256", "size", "verified_at", "source"):
            if side.get(key) is not None and rec.get(key) is None:
                rec[key] = side[key]
    return rec


def _store_record(
    name: str,
    settings: dict[str, Any] | None,
    path: Path,
    sha256: str,
    source: str,
    *,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": str(path),
        "sha256": sha256,
        "size": path.stat().st_size,
        "verified_at": _now(),
        "source": source,
    }
    record.update(extra or {})
    _write_sidecar(path, record)
    if settings is not None:
        _adopt_record(name, settings, path, record)
    return record


def _adopt_record(
    name: str, settings: dict[str, Any], path: Path, record: dict[str, Any]
) -> None:
    """Copy a verification record into settings and save them."""
    _settings_set(settings, f"datasets.{name}.path", str(path))
    for key in ("sha256", "size", "verified_at", "source", "replaced_sha256"):
        if record.get(key) is not None:
            _settings_set(settings, f"datasets.{name}.{key}", record[key])
    _save(settings)


def _free_bytes(folder: Path) -> int | None:
    probe = Path(folder)
    while not probe.exists():
        if probe.parent == probe:
            return None
        probe = probe.parent
    try:
        return shutil.disk_usage(probe).free
    except OSError:
        return None


def _require_space(folder: Path, needed: int, what: str) -> None:
    free = _free_bytes(folder)
    if free is not None and needed > 0 and free < needed:
        raise DatasetError(
            f"Not enough free disk space for {what}: it needs about "
            f"{human_bytes(needed)} and {folder} has {human_bytes(free)} free. "
            "Free some space (or choose another disk) and try again."
        )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


#: Enough for the header and five rows of the widest file (HSLS: ~9,600
#: columns, well under 1 MB for six lines). Reading a bounded prefix keeps
#: a wrongly chosen multi-GB binary file from being pulled into memory as
#: one enormous "line".
HEAD_BYTES = 16 * 1024 * 1024


def _read_head(path: Path, n_rows: int = 5) -> tuple[list[str], list[list[str]]]:
    """Header plus the first ``n_rows`` rows, read with the csv module only."""
    with open(path, "rb") as handle:
        head = handle.read(HEAD_BYTES)
    text = head.decode("utf-8-sig", errors="replace")
    if len(head) == HEAD_BYTES and "\n" in text:
        text = text[: text.rindex("\n") + 1]  # drop the partial last row
    reader = csv.reader(io.StringIO(text, newline=""))
    header = next(reader, [])
    rows: list[list[str]] = []
    for row in reader:
        rows.append(row)
        if len(rows) >= n_rows:
            break
    return [h.strip() for h in header], rows


def validate_file(name: str, path: str | Path) -> Check:
    """Check that ``path`` looks like dataset ``name``.

    Reads only the header and the first five rows. ``fail`` means the file
    cannot be used (missing, wrong columns, codes where labels are needed,
    or the reverse); ``warn`` means usable but suspicious (e.g. much
    smaller than the real file).
    """
    info = _info(name)
    title = f"{info.label} file"
    path = Path(path)
    if not path.exists():
        return Check(title, "fail", f"No file at {path}.",
                     fix=get_command(name))
    if not path.is_file():
        return Check(title, "fail", f"{path} is a folder, not a CSV file.",
                     fix=f"edmars data import {name} <path to the .csv file>")
    try:
        header, rows = _read_head(path)
    except OSError as exc:
        return Check(title, "fail", f"Could not read {path}: {exc}.")
    except csv.Error as exc:
        return Check(title, "fail", f"{path} is not a readable CSV file ({exc}).")
    if len(header) < 2:
        return Check(title, "fail",
                     f"{path.name} does not look like a CSV file (no column header row).")
    missing = [c for c in info.required_columns if c not in header]
    if missing:
        shown = ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "")
        return Check(
            title, "fail",
            f"{path.name} is missing columns this dataset must have: {shown}. "
            "It is probably a different file or a different release.",
            fix=get_command(name),
        )
    if not rows:
        return Check(title, "fail", f"{path.name} has a header but no data rows.")

    probe_values: list[str] = []
    for column in info.format_probe:
        idx = header.index(column) if column in header else -1
        if idx < 0:
            continue
        for row in rows:
            if idx < len(row) and row[idx].strip():
                probe_values.append(row[idx].strip())
    numeric = [v for v in probe_values if _NUMERIC.match(v)]
    text = [v for v in probe_values if not _NUMERIC.match(v)]
    if info.value_format == "labels" and probe_values and not text:
        how = (
            " This is the numeric-code CSV as it comes in the NCES zip; EDM-ARS "
            f"makes the labelled one from that zip itself: run `{get_command(name)}`, "
            "or import a zip you downloaded yourself with "
            f"`edmars data import {name} <path to the .zip>`."
            if info.label_table else ""
        )
        return Check(
            title, "fail",
            f"{path.name} stores numeric codes (for example {', '.join(numeric[:3])}) "
            "where EDM-ARS expects text labels such as 'Male'/'Female'." + how,
            fix=get_command(name),
        )
    if info.value_format == "codes" and text:
        return Check(
            title, "fail",
            f"{path.name} stores text labels (for example {', '.join(text[:3])}) "
            "where EDM-ARS expects the numeric codes of the NCES export.",
            fix=get_command(name),
        )

    size = path.stat().st_size
    detail = f"{len(header):,} columns; the first rows look right ({human_bytes(size)})."
    if info.min_bytes and size < info.min_bytes:
        return Check(
            title, "warn",
            f"The columns look right, but the file is only {human_bytes(size)}; the "
            f"real file is at least {human_bytes(info.min_bytes)}. It may be cut short.",
            fix=get_command(name),
        )
    return Check(title, "ok", detail)


# ---------------------------------------------------------------------------
# Status and verification
# ---------------------------------------------------------------------------


def status(name: str, settings: dict[str, Any]) -> Check:
    """Cheap install status of ``name`` (no hashing, reads nothing but stat)."""
    info = _info(name)
    title = info.label
    path = expected_path(name, settings)
    if not path.is_file():
        if info.source == "manual":
            return Check(
                title, "info",
                "Not installed. Automatic download is coming later; you can "
                "download it yourself and import it.",
                fix=f"edmars data import {name} <path to the .csv file>",
            )
        if info.source == "derived":
            ready = all(expected_path(n, settings).is_file() for n in info.needs)
            detail = (
                "Not built yet. Both source datasets are installed, so it can be "
                "built now (takes a few minutes, no download)."
                if ready else
                "Not built yet. It needs "
                + " and ".join(CATALOG[n].label.split(" (")[0] for n in info.needs)
                + " installed first."
            )
            return Check(title, "info", detail, fix=get_command(name))
        return Check(
            title, "warn" if info.recommended else "info",
            f"Not installed ({info.download_size}).",
            fix=get_command(name),
        )

    rec = _record(name, settings, path)
    size = path.stat().st_size
    if info.min_bytes and size < info.min_bytes:
        # The import already warned; a hash of a cut-short file proves only
        # that it is the same cut-short file, so it is never "verified".
        return Check(
            title, "warn",
            f"Installed at {path}, but the file is only {human_bytes(size)}; the "
            f"real file is at least {human_bytes(info.min_bytes)}. It looks "
            "incomplete, and a study on it would stop early.",
            fix=get_command(name),
        )
    if not rec.get("sha256"):
        return Check(
            title, "warn",
            f"Installed at {path}, but not verified yet.",
            fix="edmars data verify",
        )
    if rec.get("size") is not None and int(rec["size"]) != size:
        return Check(
            title, "warn",
            f"The file at {path} changed size since it was verified "
            f"({human_bytes(int(rec['size']))} then, {human_bytes(size)} now).",
            fix="edmars data verify",
        )
    when = str(rec.get("verified_at") or "")[:10]
    extra = " (EXPERIMENTAL)" if info.experimental else ""
    return Check(title, "ok", f"Installed and verified{(' on ' + when) if when else ''}{extra}.")


def verify(
    name: str, settings: dict[str, Any], progress: ProgressFn | None = None
) -> Check:
    """Re-read the whole file and compare its SHA-256 with the record.

    The first verification of a file that has no record (and no pinned
    hash) records it: trust on first use.
    """
    info = _info(name)
    path = expected_path(name, settings)
    check = validate_file(name, path)
    if check.status == "fail":
        return check
    digest = fetch.sha256_file(path, progress=progress)
    pinned = EXPECTED_SHA256.get(name)
    if pinned and digest != pinned:
        return Check(
            info.label, "fail",
            f"{path.name} does not match the expected SHA-256 for this release. "
            "It may be damaged or a different release.",
            fix=get_command(name),
        )
    rec = _record(name, settings, path)
    if rec.get("sha256") and rec["sha256"] != digest and not pinned:
        return Check(
            info.label, "fail",
            f"{path.name} changed since it was first verified on "
            f"{str(rec.get('verified_at') or '?')[:10]} (SHA-256 differs). If you "
            "replaced it on purpose, import it again; otherwise reinstall it.",
            fix=get_command(name),
        )
    _store_record(name, settings, path, digest, str(rec.get("source") or "verify"))
    note = " (first verification recorded)" if not rec.get("sha256") and not pinned else ""
    return Check(info.label, check.status, f"{check.detail} SHA-256 {digest[:12]}...{note}")


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def _find_member(archive: zipfile.ZipFile, member: str) -> zipfile.ZipInfo:
    wanted = member.lower()
    matches = [
        item for item in archive.infolist()
        if PurePosixPath(item.filename.replace("\\", "/")).name.lower() == wanted
        and not item.is_dir()
    ]
    if not matches:
        csvs = [i.filename for i in archive.infolist() if i.filename.lower().endswith(".csv")]
        shown = ", ".join(csvs[:10]) or "none"
        raise DatasetError(
            f"The downloaded archive does not contain {member}. CSV files in it: "
            f"{shown}. NCES may have changed the release; please report this."
        )
    return max(matches, key=lambda item: item.file_size)


def _extract_member(
    zip_path: Path, member: str, dest: Path, progress: ProgressFn | None
) -> None:
    """Stream one zip member to ``dest`` (never loads it into memory)."""
    part = dest.with_name(dest.name + ".part")
    try:
        with zipfile.ZipFile(zip_path) as archive:
            item = _find_member(archive, member)
            _require_space(dest.parent, item.file_size + 50_000_000, "unpacking the file")
            done = 0
            emit_progress(progress, 0, item.file_size, "extract")
            with archive.open(item) as src, open(part, "wb") as out:
                while True:
                    chunk = src.read(fetch.CHUNK_BYTES)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    emit_progress(progress, done, item.file_size, "extract")
    except NotImplementedError as exc:
        _unlink_quiet(part)
        raise DatasetError(
            f"The archive at {zip_path} uses a compression method Python cannot "
            f"unpack ({exc}). It was kept: unzip it with your system's tools, then "
            f"run `edmars data import <name> <path to {member}>`."
        ) from exc
    except zipfile.BadZipFile as exc:
        _unlink_quiet(part)
        _unlink_quiet(zip_path)
        raise DatasetError(
            f"The downloaded archive is damaged ({exc}); it was deleted. Run the "
            "same command again to download it afresh."
        ) from exc
    except BaseException:
        _unlink_quiet(part)
        raise
    os.replace(part, dest)


def _unlink_quiet(path: Path) -> None:
    try:
        Path(path).unlink()
    except OSError:
        pass


class UnknownReleaseError(DatasetError):
    """The zip is not the release the label table was built for."""


#: Labels are longer than codes: the labelled HSLS:09 CSV is 2.25 times the
#: size of the numeric one. Used for the free-space check before converting.
_LABEL_GROWTH = 2.5


def _unknown_release(info: DatasetInfo, zip_name: str, note: str = "") -> str:
    return (
        f"{zip_name} is not the {info.label.split(' (')[0]} release EDM-ARS knows: "
        "its SHA-256 differs. EDM-ARS turns the numeric codes in that zip into "
        "text labels with a table made for one exact release, so it did not "
        f"convert this one rather than risk a wrong file.{note} NCES may have "
        f"published a new release; please report this at {_ISSUES_URL}. If you "
        "have the labelled CSV (text values such as 'Male'), put it in place "
        f"with: edmars data import {info.name} <path to the .csv file>"
    )


def convert_zip(
    name: str, zip_path: str | Path, dest: str | Path, progress: ProgressFn | None = None
) -> str:
    """Convert the numeric CSV in ``zip_path`` into the labelled ``dest``.

    Checks the zip's SHA-256 first (phase ``verify``), then streams the
    member through ``edmars.relabel`` into ``dest.part`` (phase
    ``convert``) and moves it into place only when the numeric input and
    the labelled output both match their expected SHA-256s. Returns the
    output's SHA-256. The zip itself is left alone.

    Raises :class:`UnknownReleaseError` for a zip that is not the pinned
    release, and :class:`DatasetError` for anything else.
    """
    info = _info(name)
    if not info.label_table or not info.zip_sha256 or not info.member:
        raise DatasetError(f"{info.label} is not converted from a zip.")
    zip_path, dest = Path(zip_path), Path(dest)
    if not zip_path.is_file() or not zipfile.is_zipfile(zip_path):
        raise DatasetError(f"{zip_path} is not a zip file.")
    if fetch.sha256_file(zip_path, progress=progress, phase="verify") != info.zip_sha256:
        raise UnknownReleaseError(_unknown_release(info, zip_path.name))
    table = relabel.load_table(info.label_table)

    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")

    def report(done: int, total: int | None) -> None:
        emit_progress(progress, done, total, "convert")

    try:
        with zipfile.ZipFile(zip_path) as archive:
            item = _find_member(archive, info.member)
            _require_space(dest.parent, int(item.file_size * _LABEL_GROWTH) + 50_000_000,
                           "converting the file")
            report(0, item.file_size)
            with archive.open(item) as src, open(part, "wb") as out:
                result = relabel.convert_stream(src, out, table, total=item.file_size,
                                                progress=report)
    except relabel.ConversionError as exc:
        _unlink_quiet(part)
        raise DatasetError(
            f"The file in {zip_path.name} could not be converted ({exc}). Nothing "
            f"was installed; please report this at {_ISSUES_URL}."
        ) from exc
    except (zipfile.BadZipFile, NotImplementedError) as exc:
        _unlink_quiet(part)
        raise DatasetError(
            f"{zip_path.name} could not be unpacked ({exc}). Nothing was installed."
        ) from exc
    except BaseException:
        _unlink_quiet(part)
        raise
    pinned = EXPECTED_SHA256.get(name)
    wrong_input = bool(table.numeric_csv_sha256 and result.sha256_in != table.numeric_csv_sha256)
    if wrong_input or (pinned and result.sha256_out != pinned):
        _unlink_quiet(part)
        raise DatasetError(
            "The converted file does not match the expected SHA-256 "
            f"({'input' if wrong_input else 'output'} differs), so it was not "
            f"installed. Please report this at {_ISSUES_URL}."
        )
    os.replace(part, dest)
    return result.sha256_out


def download(
    name: str,
    dest_dir: str | Path,
    progress: ProgressFn | None = None,
    *,
    settings: dict[str, Any] | None = None,
    session: Any | None = None,
    force: bool = False,
) -> Path:
    """Download dataset ``name`` into ``dest_dir`` and return the CSV's path.

    ``dest_dir`` is the raw-data folder (normally ``raw_data_dir(settings)``);
    the file lands at ``dest_dir / CATALOG[name].filename``. The zip is
    streamed to disk with resume support, only the one CSV member is
    extracted (or, with a ``label_table``, converted: :func:`convert_zip`),
    the result is validated and hashed, and the zip is deleted.

    When ``settings`` is given the terms must have been accepted
    (:func:`accept_terms`) and the verification record is saved into it.
    ``progress(done, total[, phase])`` is called with phase ``download``,
    ``extract`` (or ``convert``) and ``verify``.
    """
    info = _info(name)
    if info.source != "download" or not info.url or not info.member:
        how = (
            f"edmars data install {name}" if info.source == "derived"
            else f"edmars data import {name} <path to the .csv file>"
        )
        raise DatasetError(f"{info.label} cannot be downloaded automatically. Use: {how}")
    if settings is not None and not terms_accepted(name, settings):
        raise DatasetError(
            f"The terms for {info.label} have not been accepted yet; the "
            "download was not started."
        )
    dest_dir = Path(dest_dir)
    dest = dest_dir / info.filename
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.is_file() and not force and validate_file(name, dest).status == "ok":
        # Already installed: an explicit install request re-hashes it (a few
        # seconds even for 2 GB) so a damaged copy is repaired, not kept.
        rec = _record(name, settings, dest)
        digest = fetch.sha256_file(dest, progress=progress)
        pinned = EXPECTED_SHA256.get(name)
        damaged = bool(pinned and digest != pinned) or bool(
            rec.get("sha256") and rec["sha256"] != digest and not pinned
        )
        if not damaged:
            _store_record(name, settings, dest, digest, str(rec.get("source") or info.url))
            return dest
        # The copy on disk no longer matches what was verified: asking to
        # install it again is asking for a repair, so fetch a fresh copy.

    zip_path = dest_dir / ".downloads" / PurePosixPath(info.url).name
    partial = zip_path.with_name(zip_path.name + ".part")
    already = partial.stat().st_size if partial.exists() else 0
    already += zip_path.stat().st_size if zip_path.exists() else 0
    _require_space(dest_dir, max(0, info.disk_needed_bytes - already), f"{info.label}")

    if not zip_path.is_file():
        fetch.download_file(info.url, zip_path, progress=progress, session=session)
    if not zipfile.is_zipfile(zip_path):
        _unlink_quiet(zip_path)
        inside = "the .zip" if info.label_table else "csv inside it"
        instead = (
            f", or download the zip in a browser from {info.url} and use "
            f"`edmars data import {name} <{inside}>`"
        )
        raise DatasetError(
            "The NCES server sent something that is not a zip file (perhaps an "
            "error page or a network filter's warning page). Nothing was kept; "
            f"try again later{instead}."
        )
    pinned = EXPECTED_SHA256.get(name)
    if info.label_table:
        try:
            digest = convert_zip(name, zip_path, dest, progress)
        except UnknownReleaseError:
            # Downloading again gets the same unknown zip; keeping it only
            # fills the disk. A zip that matched is kept on other errors:
            # it is the right file, and a fixed EDM-ARS can use it.
            _remove_download(zip_path)
            raise UnknownReleaseError(_unknown_release(
                info, zip_path.name, " The downloaded zip was deleted.")) from None
    else:
        _extract_member(zip_path, info.member, dest, progress)
        check = validate_file(name, dest)
        if check.status == "fail":
            rejected = dest.with_name(dest.name + ".rejected")
            os.replace(dest, rejected)
            raise DatasetError(f"The downloaded file did not pass validation: {check.detail} "
                               f"(kept for inspection at {rejected}).")
        digest = fetch.sha256_file(dest, progress=progress)
        if pinned and digest != pinned:
            rejected = dest.with_name(dest.name + ".rejected")
            os.replace(dest, rejected)
            raise DatasetError(
                f"The downloaded file does not match the expected SHA-256 (kept for "
                f"inspection at {rejected})."
            )
    rec = _record(name, settings, dest)
    extra: dict[str, Any] = {}
    if rec.get("sha256") and rec["sha256"] != digest and not pinned:
        # Trust-on-first-use: a fresh download that differs from the first
        # verified copy is a new NCES release (or damage in the old copy).
        # The user asked for this download, so keep it, and keep the old
        # hash on record so the change is visible.
        extra["replaced_sha256"] = rec["sha256"]
    _store_record(name, settings, dest, digest, info.url, extra=extra)
    _remove_download(zip_path)
    return dest


def _remove_download(zip_path: Path) -> None:
    """Delete a downloaded zip, and its ``.downloads`` folder once empty."""
    _unlink_quiet(zip_path)
    try:
        zip_path.parent.rmdir()
    except OSError:
        pass


def install(
    name: str,
    settings: dict[str, Any],
    progress: ProgressFn | None = None,
    *,
    session: Any | None = None,
    force: bool = False,
) -> Path:
    """Download or build ``name`` into ``raw_data_dir(settings)``."""
    info = _info(name)
    if info.source == "derived":
        return build_did_panel(settings)
    if info.source == "manual":
        raise DatasetError(
            f"{info.label}: automatic download is coming later. Download the file "
            f"yourself and run `edmars data import {name} <path to the .csv file>`."
        )
    return download(name, raw_data_dir(settings), progress, settings=settings,
                    session=session, force=force)


# ---------------------------------------------------------------------------
# Import
# ---------------------------------------------------------------------------


def _is_synced(path: Path) -> bool:
    try:
        from edmars import paths

        return paths.sync_provider(path) is not None
    except Exception:  # noqa: BLE001 - sync detection is advisory only
        return False


def import_file(name: str, path: str | Path, settings: dict[str, Any]) -> Path:
    """Validate ``path`` and place it where the pipeline expects ``name``.

    Uses a hard link when the source is on the same disk and not inside a
    cloud-sync folder (no second copy of a 2 GB file); otherwise copies.
    For a dataset with a ``label_table``, ``path`` may also be the zip as
    downloaded from NCES: it is checked and converted (:func:`convert_zip`)
    and left where it is. The verification record is saved into ``settings``.
    """
    info = _info(name)
    src = Path(path).expanduser()
    if info.label_table and src.is_file() and zipfile.is_zipfile(src):
        dest = expected_path(name, settings)
        digest = convert_zip(name, src, dest)
        _store_record(name, settings, dest, digest, f"import:{src.name}")
        return dest
    check = validate_file(name, src)
    if check.status == "fail":
        raise DatasetError(check.detail)
    src = src.resolve()
    dest = expected_path(name, settings)
    dest.parent.mkdir(parents=True, exist_ok=True)

    same = False
    if dest.exists():
        try:
            same = os.path.samefile(src, dest)
        except OSError:
            same = False
    if not same:
        tmp = dest.with_name(dest.name + ".importing")
        _unlink_quiet(tmp)
        linked = False
        if not _is_synced(src):
            try:
                os.link(src, tmp)
                linked = True
            except OSError:
                linked = False
        if not linked:
            _require_space(dest.parent, src.stat().st_size + 50_000_000, "the copy")
            try:
                shutil.copyfile(src, tmp)
            except BaseException:
                _unlink_quiet(tmp)
                raise
        os.replace(tmp, dest)

    digest = fetch.sha256_file(dest)
    pinned = EXPECTED_SHA256.get(name)
    if pinned and digest != pinned:
        raise DatasetError(
            f"{src.name} has the right columns, but its SHA-256 differs from the "
            "one EDM-ARS expects for this release, so it may be a different "
            f"release or export. It was placed at {dest} but not marked verified."
        )
    _store_record(name, settings, dest, digest, f"import:{src.name}")
    return dest


# ---------------------------------------------------------------------------
# Derived panel
# ---------------------------------------------------------------------------

#: Runs scripts/harmonize_els_hsls.py's own main() with its ROOT pointed at
#: the user's data folder. The script hard-codes ``ROOT = <repo>`` and reads
#: ``ROOT/data/raw/...``; the user data folder has the same ``data/raw``
#: layout, so overriding the two module globals is all it takes -- the
#: harmonization logic itself runs unchanged.
_HARMONIZE_DRIVER = """\
import importlib.util, pathlib, sys
script, root = sys.argv[1], pathlib.Path(sys.argv[2])
spec = importlib.util.spec_from_file_location('harmonize_els_hsls', script)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.ROOT = root
mod.OUT = root / 'data' / 'raw' / 'did_els_hsls_panel'
mod.main()
"""


def _child_env() -> dict[str, str]:
    env = {
        k: v for k, v in os.environ.items()
        if not any(h in k.upper() for h in _SECRET_ENV_HINTS)
        and k.upper() not in ("PYTHONPATH", "PYTHONHOME")
    }
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def build_did_panel(settings: dict[str, Any], *, timeout_s: int = 3600) -> Path:
    """Build the ELS x HSLS panel from the two installed source files.

    Runs ``scripts/harmonize_els_hsls.py`` in a child Python (via
    ``edmars.proc``) against the user's raw-data folder. Takes a few
    minutes (it reads eight columns of the 2 GB HSLS file). A previous
    panel is kept until the new one has been written and validated.
    """
    from edmars import paths, proc

    info = _info("did_els_hsls_panel")
    raw = raw_data_dir(settings)
    for needed in info.needs:
        if not (raw / CATALOG[needed].filename).is_file():
            raise DatasetError(
                f"The panel needs {CATALOG[needed].label} first: "
                f"run `edmars data install {needed}`."
            )
    script = Path(paths.app_root()) / "scripts" / "harmonize_els_hsls.py"
    if not script.is_file():
        raise DatasetError(f"The panel builder is missing from the app: {script}.")
    root = raw.parent.parent
    if root / "data" / "raw" != raw:  # pragma: no cover - raw_data_dir guarantees it
        raise DatasetError(f"Unexpected raw-data folder layout: {raw}.")

    out = raw / info.filename
    out.parent.mkdir(parents=True, exist_ok=True)
    previous = out.with_name(out.name + ".previous")
    had_previous = out.is_file()
    if had_previous:
        os.replace(out, previous)

    def _restore() -> None:
        _unlink_quiet(out)
        if had_previous and previous.exists():
            os.replace(previous, out)

    try:
        result = proc.run(
            [sys.executable, "-c", _HARMONIZE_DRIVER, str(script), str(root)],
            timeout=timeout_s,
            cwd=str(paths.app_root()),
            env=_child_env(),
        )
    except Exception as exc:  # noqa: BLE001 - timeout or spawn failure
        _restore()
        if type(exc).__name__ == "TimeoutExpired":
            raise DatasetError(
                f"Building the panel took longer than {timeout_s // 60} minutes and was stopped."
            ) from exc
        raise DatasetError(f"Could not start the panel builder: {exc}") from exc
    if result.returncode != 0 or not out.is_file():
        _restore()
        tail = "\n".join((result.stderr or result.stdout or "").strip().splitlines()[-6:])
        raise DatasetError(
            f"The panel builder failed (exit code {result.returncode}).\n{tail}"
        )
    check = validate_file("did_els_hsls_panel", out)
    if check.status == "fail":
        _restore()
        raise DatasetError(f"The built panel did not pass validation: {check.detail}")
    _unlink_quiet(previous)
    digest = fetch.sha256_file(out)
    _store_record("did_els_hsls_panel", settings, out, digest, "built:harmonize_els_hsls.py")
    return out


def catalog_checks(settings: dict[str, Any]) -> list[Check]:
    """``status()`` for every catalog entry (for ``edmars data list``/doctor)."""
    return [status(name, settings) for name in CATALOG]
