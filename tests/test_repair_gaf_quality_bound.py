"""Bound synthetic base-quality allocation in the GAF repair script.

These tests fail while ``scripts/repair_mojo_gaf_for_methylcall.py`` builds
``bq:Z:`` by repeating one character ``qlen`` times with no upper bound.
They pass when every record whose query length is above
``MAX_ACCEPTED_QUERY_BASES`` is skipped or rejected before that string is
allocated.

``MAX_ACCEPTED_QUERY_BASES`` is the cap the repair step must enforce. It is
above every read length shipped in this repository's fixtures and is the
maximum query length the repair step may materialize.
"""
from __future__ import annotations

import io
import sys
import tracemalloc
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from repair_mojo_gaf_for_methylcall import (  # noqa: E402
    _repair_line,
    main,
    repair_stream,
)

# 1 MiB. Existing fixtures are 151 bp; a record past this cap is not a read
# the repair step may turn into a quality string.
MAX_ACCEPTED_QUERY_BASES = 1_048_576
ONE_PAST_CAP = MAX_ACCEPTED_QUERY_BASES + 1
# Large enough that a missing cap allocates megabytes from a short line, and
# small enough that the failing test can measure that allocation.
OVERSIZED_QUERY_BASES = 3_000_000
# Parsing and a short repaired line stay under this. Materializing a quality
# string at the cap (~1 MiB) or at OVERSIZED_QUERY_BASES does not.
PEAK_BUDGET_BYTES = 256 * 1024


def _gaf_line(qlen: int, *, extra_tags: str = "") -> str:
    """12-column GAF row. No bq:Z tag unless ``extra_tags`` adds one."""
    line = (
        f"r\t{qlen}\t0\t{qlen}\t+\t>1\t{qlen}\t0\t{qlen}\t{qlen}\t{qlen}\t20"
    )
    if extra_tags:
        line += "\t" + extra_tags
    return line + "\n"


def _bq_payload_len(line: str) -> int | None:
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 13:
        return None
    for tag in parts[12:]:
        if tag.startswith("bq:Z:"):
            return len(tag) - 5
    return None


def _invoke(fn):
    """Run ``fn`` and return ``(result, exception, tracemalloc peak)``.

    ``MemoryError`` propagates: an allocator failure is not a refusal that
    happened before the quality string was built.
    """
    tracemalloc.start()
    result = None
    caught = None
    try:
        try:
            result = fn()
        except MemoryError:
            raise
        except (Exception, SystemExit) as exc:
            caught = exc
    finally:
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
    return result, caught, peak


def _assert_not_materialized(result, caught, peak, written: str = "") -> None:
    assert peak < PEAK_BUDGET_BYTES, peak
    texts = []
    if isinstance(result, str):
        texts.append(result)
    if written:
        texts.append(written)
    for text in texts:
        for line in text.splitlines():
            payload = _bq_payload_len(line + "\n")
            if payload is not None:
                assert payload <= MAX_ACCEPTED_QUERY_BASES
    # A refusal is optional. A success must still have stayed inside the budget.
    assert caught is None or peak < PEAK_BUDGET_BYTES


def test_short_line_at_cap_still_receives_synthetic_quality():
    """In-range records keep the repair behavior MethylCall relies on."""
    line = _gaf_line(MAX_ACCEPTED_QUERY_BASES)
    out = _repair_line(line)
    assert _bq_payload_len(out) == MAX_ACCEPTED_QUERY_BASES


def test_repair_line_does_not_allocate_one_past_cap():
    line = _gaf_line(ONE_PAST_CAP)
    assert len(line) < 200
    result, caught, peak = _invoke(lambda: _repair_line(line))
    _assert_not_materialized(result, caught, peak)


def test_repair_line_does_not_allocate_oversized_quality_from_short_line():
    line = _gaf_line(OVERSIZED_QUERY_BASES, extra_tags="cs:Z::1\tri:i:1")
    assert len(line) < 200
    assert "bq:Z:" not in line
    result, caught, peak = _invoke(lambda: _repair_line(line))
    _assert_not_materialized(result, caught, peak)


def test_repair_line_keeps_existing_quality_without_using_query_length():
    line = _gaf_line(OVERSIZED_QUERY_BASES, extra_tags="bq:Z:IIII")
    result, caught, peak = _invoke(lambda: _repair_line(line))
    assert caught is None
    assert peak < PEAK_BUDGET_BYTES
    assert result is not None
    assert _bq_payload_len(result) == 4


def test_repair_stream_bounds_every_record():
    text = (
        _gaf_line(151)
        + _gaf_line(OVERSIZED_QUERY_BASES)
        + _gaf_line(10)
    )
    fin = io.StringIO(text)
    fout = io.StringIO()

    def run():
        return repair_stream(fin, fout)

    _result, caught, peak = _invoke(run)
    written = fout.getvalue()
    _assert_not_materialized(written, caught, peak, written)
    assert len(written) < PEAK_BUDGET_BYTES


def _run_main(argv: list[str]):
    old = sys.argv
    sys.argv = ["repair_mojo_gaf_for_methylcall.py", *argv]
    try:
        return main()
    finally:
        sys.argv = old


def test_cli_output_does_not_materialize_oversized_quality(tmp_path: Path):
    src = tmp_path / "in.gaf"
    src.write_text(_gaf_line(OVERSIZED_QUERY_BASES))
    out = tmp_path / "out.gaf"

    def run():
        return _run_main([str(src), "-o", str(out)])

    result, caught, peak = _invoke(run)
    written = out.read_text() if out.exists() else ""
    _assert_not_materialized(None, caught, peak, written)
    assert len(written) < PEAK_BUDGET_BYTES
    if caught is None:
        assert result == 0
    for path in tmp_path.iterdir():
        assert path.stat().st_size < PEAK_BUDGET_BYTES


def test_cli_replace_does_not_swap_in_oversized_quality(tmp_path: Path):
    src = tmp_path / "in.gaf"
    original = _gaf_line(OVERSIZED_QUERY_BASES)
    src.write_text(original)

    def run():
        return _run_main([str(src), "--replace"])

    _result, caught, peak = _invoke(run)
    _assert_not_materialized(None, caught, peak)
    if caught is not None:
        # Refusing the file must not rename or replace the caller's alignment.
        assert src.read_text() == original
        assert not Path(str(src) + ".pre_methylcall_fix.bak").exists()
    else:
        for line in src.read_text().splitlines():
            payload = _bq_payload_len(line + "\n")
            if payload is not None:
                assert payload <= MAX_ACCEPTED_QUERY_BASES
    for path in tmp_path.iterdir():
        assert path.stat().st_size < PEAK_BUDGET_BYTES


def test_cli_repairs_in_range_record(tmp_path: Path):
    src = tmp_path / "in.gaf"
    src.write_text(_gaf_line(151, extra_tags="cs:Z::32"))
    out = tmp_path / "out.gaf"
    code = _run_main([str(src), "-o", str(out)])
    assert code == 0
    assert _bq_payload_len(out.read_text()) == 151
