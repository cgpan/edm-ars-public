"""The numeric-code to labelled CSV conversion (edmars.relabel), byte for byte."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import pytest

from edmars import relabel


def _table(tmp_path: Path, **overrides: Any) -> relabel.LabelTable:
    raw: dict[str, Any] = {
        "format": 1,
        "header_renames": {"PSU": "psu"},
        "quote_all": ["ID", "SEX"],
        "labels": {
            "SEX": {"1": "Male", "2": "Female", "-9": "Missing"},
            "NOTE": {"1": 'Said "yes"', "2": "Yes, twice", "3": "plain"},
            "SCORE": {"-0.0002": "-2e-04"},
        },
    }
    raw.update(overrides)
    path = tmp_path / "t.labels.json.gz"
    path.write_bytes(gzip.compress(json.dumps(raw).encode("utf-8")))
    return relabel.load_table(path)


def _convert(data: bytes, table: relabel.LabelTable, **kwargs: Any) -> tuple[bytes, Any]:
    out = io.BytesIO()
    result = relabel.convert_stream(io.BytesIO(data), out, table, **kwargs)
    return out.getvalue(), result


def test_codes_become_labels_with_the_labelled_files_quoting(tmp_path: Path) -> None:
    numeric = (
        b'"ID","PSU",SEX,NOTE,SCORE,OTHER\r\n'
        b'"1",-5,1,1,-0.0002,7\n'
        b'"2",-5,2,2,0.5,"x"\n'
        b'"3",-5,-9,3,1.25,8\n'
    )
    labelled, result = _convert(numeric, _table(tmp_path))
    assert labelled == (
        # header: bytes and CRLF kept, PSU renamed with its quotes
        b'"ID","psu",SEX,NOTE,SCORE,OTHER\r\n'
        # quote_all column quoted; a label with a quote is quoted and escaped;
        # a number re-formatted as the labelled file has it; rows end CRLF
        b'"1",-5,"Male","Said ""yes""",-2e-04,7\r\n'
        # a label with a comma is quoted; unmapped values are left alone
        b'"2",-5,"Female","Yes, twice",0.5,"x"\r\n'
        # a plain label outside quote_all stays unquoted
        b'"3",-5,"Missing",plain,1.25,8\r\n'
    )
    assert result.rows == 3
    assert result.bytes_in == len(numeric) and result.bytes_out == len(labelled)
    assert result.sha256_in == hashlib.sha256(numeric).hexdigest()
    assert result.sha256_out == hashlib.sha256(labelled).hexdigest()


def test_unquoted_header_names_are_renamed_unquoted(tmp_path: Path) -> None:
    labelled, _ = _convert(b"ID,PSU\n1,2\n", _table(tmp_path))
    assert labelled == b'ID,psu\n"1",2\r\n'


def test_crlf_input_and_a_missing_final_newline_give_crlf_rows(tmp_path: Path) -> None:
    labelled, result = _convert(b'ID,SEX\r\n1,1\r\n2,2', _table(tmp_path))
    assert labelled == b'ID,SEX\r\n"1","Male"\r\n"2","Female"\r\n'
    assert result.rows == 2


def test_rows_longer_than_the_read_size_are_joined_correctly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(relabel, "_READ_BYTES", 5)  # every row spans several reads
    labelled, result = _convert(b"ID,SEX\n1001,1\n1002,2\n", _table(tmp_path))
    assert labelled == b'ID,SEX\n"1001","Male"\r\n"1002","Female"\r\n'
    assert result.rows == 2


def test_progress_reports_bytes_read_against_the_total(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(relabel, "_READ_BYTES", 8)
    data = b"ID,SEX\n1,1\n2,2\n3,1\n"
    seen: list[tuple[int, Any]] = []
    _convert(data, _table(tmp_path), total=len(data), progress=lambda d, t: seen.append((d, t)))
    assert seen[-1] == (len(data), len(data))
    assert [d for d, _ in seen] == sorted(d for d, _ in seen)


def test_a_row_with_the_wrong_number_of_fields_stops_the_conversion(tmp_path: Path) -> None:
    with pytest.raises(relabel.ConversionError, match="Row 2 has 3 fields"):
        _convert(b"ID,SEX\n1,1\n2,2,3\n", _table(tmp_path))


def test_an_empty_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(relabel.ConversionError, match="empty"):
        _convert(b"", _table(tmp_path))


def test_a_table_of_another_format_is_refused(tmp_path: Path) -> None:
    with pytest.raises(relabel.ConversionError, match="format-1"):
        _table(tmp_path, format=2)


def test_bare_table_names_are_looked_up_in_the_package_data() -> None:
    assert relabel.table_path("x.labels.json.gz") == relabel.DATA_DIR / "x.labels.json.gz"
    assert relabel.DATA_DIR.parent == Path(relabel.__file__).resolve().parent
