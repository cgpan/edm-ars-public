"""Turn a numeric-code NCES CSV into the labelled CSV EDM-ARS expects.

The HSLS:09 zip NCES publishes holds the student file with numeric codes
(``X1SEX`` is ``1``/``2``), but the pipeline, the variable registry and
the prompts were built on the labelled file (``Male``/``Female``). This
module rebuilds the labelled file from the numeric one with a bundled
code-to-label table, ``edmars/data/<dataset>.labels.json.gz``.

The table holds column names, codes and labels only, no student rows. It
was built by comparing the two files value by value, and applied to the
numeric CSV from the NCES zip it reproduces the labelled file byte for
byte. That holds for that exact release only, so the callers in
``edmars.datasets`` check the zip's SHA-256 before converting and the
result's SHA-256 after: a different release is refused, never guessed at.

The conversion streams: it holds one row (about 50 KB) at a time, never
the file. For each row it

* maps every coded value to its label (for example ``1`` to ``Male``),
* puts the values of the ``quote_all`` columns in double quotes, as the
  labelled file does, and quotes any label that contains a comma or a
  quote,
* ends the row with CRLF (the numeric file uses LF; the labelled one CRLF).

The header line keeps its bytes and line ending apart from the renames in
``header_renames`` (``PSU`` is ``psu`` in the labelled file).

The numeric file has no quoted commas, so a row splits on every comma; a
row with the wrong number of fields stops the conversion. A source
checkout can run it with the pipeline's own requirements (none of the
CLI's packages are needed)::

    python -m edmars.relabel HSLS_2017_PETS_SR_v1_0_CSV_Datasets.zip \
        data/raw/hsls_17_student_pets_sr_v1_0.csv
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Callable, Iterator, Optional

#: Folder of the bundled tables.
DATA_DIR = Path(__file__).resolve().parent / "data"

#: The table format this module reads.
TABLE_FORMAT = 1

_READ_BYTES = 1 << 20


class ConversionError(ValueError):
    """The input is not the file the table was built for."""


@dataclass(frozen=True)
class LabelTable:
    """A code-to-label table, ready for :func:`convert_stream`."""

    #: column -> {code bytes -> label bytes}
    labels: dict[str, dict[bytes, bytes]]
    #: Columns whose every value the labelled file puts in double quotes.
    quote_all: frozenset[str]
    #: Header names that differ: numeric-file name -> labelled-file name.
    header_renames: dict[str, str]
    #: SHA-256 of the numeric CSV the table was built from, if recorded.
    numeric_csv_sha256: Optional[str] = None
    #: SHA-256 of the labelled CSV it reproduces, if recorded.
    labelled_csv_sha256: Optional[str] = None


@dataclass(frozen=True)
class ConversionResult:
    """What :func:`convert_stream` read and wrote."""

    rows: int
    bytes_in: int
    bytes_out: int
    sha256_in: str
    sha256_out: str


def table_path(name: str) -> Path:
    """Where a bundled table lives: a bare file name is looked up in DATA_DIR."""
    path = Path(name)
    return path if path.is_absolute() else DATA_DIR / path


def load_table(name: str | Path) -> LabelTable:
    """Read a gzipped JSON table (see ``edmars/data``)."""
    path = table_path(str(name))
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict) or raw.get("format") != TABLE_FORMAT:
        raise ConversionError(f"{path.name} is not a format-{TABLE_FORMAT} label table.")
    labels = {
        str(column): {str(code).encode("utf-8"): str(label).encode("utf-8")
                      for code, label in dict(pairs).items()}
        for column, pairs in dict(raw.get("labels") or {}).items()
    }
    return LabelTable(
        labels=labels,
        quote_all=frozenset(str(c) for c in raw.get("quote_all") or ()),
        header_renames={str(k): str(v) for k, v in dict(raw.get("header_renames") or {}).items()},
        numeric_csv_sha256=raw.get("numeric_csv_sha256"),
        labelled_csv_sha256=raw.get("labelled_csv_sha256"),
    )


def _quote(value: bytes) -> bytes:
    return b'"' + value.replace(b'"', b'""') + b'"'


def _unquote(raw: bytes) -> tuple[bytes, bool]:
    if raw.startswith(b'"'):
        return raw[1:-1].replace(b'""', b'"'), True
    return raw, False


def _split_eol(line: bytes) -> tuple[bytes, bytes]:
    body = line.rstrip(b"\r\n")
    return body, line[len(body):]


def convert_stream(
    src: IO[bytes],
    out: IO[bytes],
    table: LabelTable,
    *,
    total: Optional[int] = None,
    progress: Optional[Callable[[int, Optional[int]], None]] = None,
) -> ConversionResult:
    """Read a numeric CSV from ``src`` and write the labelled CSV to ``out``.

    ``progress(bytes_read, total)`` is called about once per MiB read.
    Both SHA-256s are computed on the way, so the caller can check the
    input and the output without reading either file again.
    """
    hash_in = hashlib.sha256()
    hash_out = hashlib.sha256()
    counts = {"in": 0, "out": 0}

    def lines() -> Iterator[bytes]:
        rest = b""
        while True:
            block = src.read(_READ_BYTES)
            if not block:
                break
            hash_in.update(block)
            counts["in"] += len(block)
            if progress is not None:
                progress(counts["in"], total)
            parts = (rest + block).split(b"\n")
            rest = parts.pop()
            for part in parts:
                yield part + b"\n"
        if rest:
            yield rest

    def write(data: bytes) -> None:
        out.write(data)
        hash_out.update(data)
        counts["out"] += len(data)

    source = lines()
    header = next(source, b"")
    if not header.strip():
        raise ConversionError("The file is empty: there is no header line.")
    head_body, head_eol = _split_eol(header)
    try:
        names = next(csv.reader([head_body.decode("utf-8")]))
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ConversionError(f"The header line is not readable CSV ({exc}).") from exc
    raw_names = head_body.split(b",")
    if len(raw_names) != len(names):
        raise ConversionError("The header has quoted commas; this is not the expected file.")
    new_names = []
    for raw in raw_names:
        value, quoted = _unquote(raw)
        renamed = table.header_renames.get(value.decode("utf-8"))
        if renamed is None:
            new_names.append(raw)
        else:
            new = renamed.encode("utf-8")
            new_names.append(_quote(new) if quoted else new)
    write(b",".join(new_names) + head_eol)

    width = len(names)
    active = [
        (i, table.labels.get(column, {}), column in table.quote_all)
        for i, column in enumerate(names)
        if column in table.labels or column in table.quote_all
    ]
    rows = 0
    for line in source:
        body = line.rstrip(b"\r\n")
        fields = body.split(b",")
        if len(fields) != width:
            raise ConversionError(
                f"Row {rows + 1} has {len(fields):,} fields where the header has "
                f"{width:,}; this is not the file the label table was built for."
            )
        # Inlined (no helper calls): this loop runs about 77 million times
        # for HSLS:09, and call overhead alone would add several seconds.
        for i, codes, quote_all in active:
            raw = fields[i]
            quoted = raw.startswith(b'"')
            value = raw[1:-1].replace(b'""', b'"') if quoted else raw
            label = codes.get(value)
            if label is None:
                if quote_all and not quoted:
                    fields[i] = b'"' + value.replace(b'"', b'""') + b'"'
            elif quote_all or b"," in label or b'"' in label:
                fields[i] = b'"' + label.replace(b'"', b'""') + b'"'
            else:
                fields[i] = label
        write(b",".join(fields) + b"\r\n")
        rows += 1
    return ConversionResult(
        rows=rows,
        bytes_in=counts["in"],
        bytes_out=counts["out"],
        sha256_in=hash_in.hexdigest(),
        sha256_out=hash_out.hexdigest(),
    )


def main(argv: Optional[list[str]] = None) -> int:
    """``python -m edmars.relabel ZIP OUT_CSV`` for source checkouts."""
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2 or args[0] in ("-h", "--help"):
        print("usage: python -m edmars.relabel HSLS_2017_PETS_SR_v1_0_CSV_Datasets.zip "
              "data/raw/hsls_17_student_pets_sr_v1_0.csv")
        return 0 if args[:1] in (["-h"], ["--help"]) else 2
    from edmars import datasets

    print(f"Checking {args[0]} and converting the CSV in it (about a minute)...", flush=True)
    try:
        digest = datasets.convert_zip("hsls09_public", Path(args[0]), Path(args[1]))
    except datasets.DatasetError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(f"Wrote {args[1]} (SHA-256 {digest}, checked).")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
