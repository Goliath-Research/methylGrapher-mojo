"""Segment-pack offset/length must be rejected without reading past sequences.bin.

The production Mojo reader (`DensePack.get` / `get_by_index`) and the GPU
gapless kernel both accept a record when the signed sum of the uint64 offset
and length wraps. These tests require that record to be rejected, and the
process to exit cleanly, before any byte outside sequences.bin is consumed.

They fail while that check is missing or still uses a wrapping signed add.
A fix that checks the record and then loads the offset or length again before
the read is also a failure: the GPU test rejects a kernel that reloads those
fields between the check and the sequence load.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DRIVER = ROOT / "tests" / "segment_offset_overflow_driver.mojo"
SEQ = b"ACGTACGT"

# Positive signed values whose sum does not fit in a signed 64-bit word.
_I64_MAX = (1 << 63) - 1
OVERFLOW_RECORDS = {
    "sum_of_two_2_62": (1 << 62, 1 << 62),
    "offset_inside_length_wraps": (4, _I64_MAX),
    "offset_near_signed_max": (_I64_MAX - 1, 4),
}


def _mojo() -> str:
    return os.environ.get("MOJO") or shutil.which("mojo") or "mojo"


def _write_pack(directory: Path, records: list[tuple[int, int]], seq: bytes = SEQ) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "sequences.bin").write_bytes(seq)
    blob = b"".join(struct.pack("<QQ", offset, length) for offset, length in records)
    (directory / "offsets.bin").write_bytes(blob)
    (directory / "ids.txt").write_text("".join(f"{i}\n" for i in range(1, len(records) + 1)))
    (directory / "meta.json").write_text(
        json.dumps({"format": "dense-v1", "n_segments": len(records)}) + "\n"
    )


def _run_driver(pack: Path, *args: str, timeout: float = 15) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_mojo(), "-I", "src", str(DRIVER), str(pack), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env={**os.environ, "PYTHONPATH": str(ROOT)},
    )


def _require_clean(proc: subprocess.CompletedProcess[str], pack: Path) -> str:
    assert proc.returncode == 0, (
        f"driver crashed or exited {proc.returncode} for {pack}\n"
        f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    assert "PATH=mmap" in proc.stdout, proc.stdout + proc.stderr
    assert "PATH=python" not in proc.stdout
    return proc.stdout


def _field(stdout: str, name: str) -> str:
    match = re.search(rf"^{name}=\[(.*)\]$", stdout, re.MULTILINE)
    assert match, f"missing {name} in:\n{stdout}"
    return match.group(1)


def test_in_range_segment_is_returned(tmp_path: Path) -> None:
    pack = tmp_path / "valid"
    _write_pack(pack, [(0, len(SEQ))])
    stdout = _require_clean(_run_driver(pack, "get", "1"), pack)
    assert _field(stdout, "RESULT") == SEQ.decode()
    assert "RESULT_LEN=8" in stdout


def test_offset_past_end_without_overflow_is_rejected(tmp_path: Path) -> None:
    pack = tmp_path / "past"
    _write_pack(pack, [(100, 4)])
    stdout = _require_clean(_run_driver(pack, "get", "1"), pack)
    assert _field(stdout, "RESULT") == ""
    assert "RESULT_LEN=0" in stdout


def test_length_that_fits_exactly_at_end_is_returned(tmp_path: Path) -> None:
    pack = tmp_path / "tail"
    _write_pack(pack, [(4, 4)])
    stdout = _require_clean(_run_driver(pack, "get", "1"), pack)
    assert _field(stdout, "RESULT") == "ACGT"


@pytest.mark.parametrize("name", list(OVERFLOW_RECORDS))
def test_wrapping_offset_and_length_are_rejected(tmp_path: Path, name: str) -> None:
    pack = tmp_path / name
    _write_pack(pack, [OVERFLOW_RECORDS[name]])
    stdout = _require_clean(_run_driver(pack, "get", "1"), pack)
    assert _field(stdout, "RESULT") == ""
    assert "RESULT_LEN=0" in stdout


def test_sibling_record_still_readable_when_another_wraps(tmp_path: Path) -> None:
    pack = tmp_path / "mixed"
    _write_pack(pack, [OVERFLOW_RECORDS["sum_of_two_2_62"], (0, len(SEQ))])
    stdout = _require_clean(_run_driver(pack, "get", "2"), pack)
    assert _field(stdout, "RESULT") == SEQ.decode()


def test_host_gapless_matches_a_real_segment(tmp_path: Path) -> None:
    pack = tmp_path / "valid"
    _write_pack(pack, [(0, len(SEQ))])
    stdout = _require_clean(
        _run_driver(pack, "gapless", "1", "0", SEQ.decode()), pack
    )
    assert "HITS=1" in stdout
    assert _field(stdout, "HITPATH") == ">1"


@pytest.mark.parametrize("orient", ["0", "1"])
def test_host_gapless_rejects_wrapping_segment(tmp_path: Path, orient: str) -> None:
    pack = tmp_path / f"gapless-{orient}"
    _write_pack(pack, [OVERFLOW_RECORDS["offset_inside_length_wraps"]])
    stdout = _require_clean(
        _run_driver(pack, "gapless", "1", orient, SEQ.decode()), pack
    )
    assert "HITS=0" in stdout
    assert _field(stdout, "HITPATH") == ""
    assert _field(stdout, "HITCS") == ""


_WRAPPER = textwrap.dedent(
    """\
    from std.python import Python
    from giraffe_pack import DensePack
    from giraffe_min_index import MojoMinIndex
    from giraffe_gpu_map_kernels import gpu_native_stream_loop

    def main() raises:
        var pack = DensePack("/tmp/no-such-pack")
        var idx = MojoMinIndex("/tmp/no-such.min")
        var builtins = Python.import_module("builtins")
        var fh = builtins.open("/tmp/no-such-fq", "r")
        var n = gpu_native_stream_loop(
            pack,
            idx,
            fh,
            fh,
            False,
            fh,
            String(""),
            1,
            False,
            String("devicecontext-cuda"),
            String("nvidia"),
        )
        print(n)
    """
)


def _compile_gapless_ptx(directory: Path) -> str:
    source = directory / "force_gapless.mojo"
    source.write_text(_WRAPPER)
    asm = directory / "force_gapless.s"
    proc = subprocess.run(
        [
            _mojo(),
            "build",
            "--target-accelerator",
            "sm_80",
            "--emit",
            "asm",
            "-I",
            "src",
            "-o",
            str(asm),
            str(source),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    ptx_files = [
        path
        for path in directory.glob("*.ptx")
        if "kernels_gapless" in path.name
    ]
    assert len(ptx_files) == 1, f"expected one gapless kernel PTX, found {ptx_files}\n{proc.stderr}"
    return ptx_files[0].read_text()


def _regs_defined_by(lines: list[str]) -> dict[str, tuple[str, int, re.Match[str]]]:
    patterns = (
        ("ld64", re.compile(r"ld\.global\.b64\s+(%r[a-z0-9]+),\s+\[(%r[a-z0-9]+)([+-]*\d+)?\]")),
        ("param", re.compile(r"ld\.param\.b64\s+(%r[a-z0-9]+),")),
        ("add", re.compile(r"add\.s64\s+(%r[a-z0-9]+),\s+(%r[a-z0-9]+),\s+(%r[a-z0-9]+)")),
        ("sub", re.compile(r"sub\.s64\s+(%r[a-z0-9]+),\s+(%r[a-z0-9]+),\s+(%r[a-z0-9]+)")),
    )
    defs: dict[str, tuple[str, int, re.Match[str]]] = {}
    for index, line in enumerate(lines):
        for kind, pattern in patterns:
            match = pattern.search(line)
            if match:
                defs[match.group(1)] = (kind, index, match)
                break
    return defs


def _setp(line: str) -> re.Match[str] | None:
    return re.search(
        r"setp\.([a-z0-9.]+)\s+%p\d+,\s+(%r[a-z0-9]+),\s+(%r[a-z0-9]+|-?\d+)",
        line,
    )


def _ordered_gt(pred: str, left: str, right: str, want_left: str, want_right: str) -> bool:
    """True when the compare says want_left > want_right (either operand order)."""
    if pred in ("gt.s64", "ge.s64", "gt.u64", "ge.u64", "hi.u64", "hs.u64"):
        return left == want_left and right == want_right
    if pred in ("lt.s64", "le.s64", "lt.u64", "le.u64", "lo.u64", "ls.u64"):
        return left == want_right and right == want_left
    return False


def _displacement(text: str | None) -> int:
    if not text:
        return 0
    return int(text.replace("+-", "-"))


def gapless_kernel_accepts_wrapping_segment(ptx: str) -> bool:
    """True when the compiled gapless kernel still trusts a wrapping signed sum.

    Also true when a sequence load uses an offset or length re-read from the
    pack after the range check, instead of the values that were checked.
    """
    lines = [
        line.strip().rstrip(";")
        for line in ptx.splitlines()
        if line.strip() and not line.strip().startswith("//")
    ]
    defs = _regs_defined_by(lines)
    ld64 = re.compile(
        r"ld\.global\.b64\s+(%r[a-z0-9]+),\s+\[(%r[a-z0-9]+)([+-]*\d+)?\]"
    )
    fields: list[tuple[str, str, int, int]] = []
    for index, line in enumerate(lines):
        match = ld64.search(line)
        if match:
            fields.append(
                (match.group(1), match.group(2), _displacement(match.group(3)), index)
            )
    pair: tuple[tuple[str, str, int, int], tuple[str, str, int, int]] | None = None
    for left, right in zip(fields, fields[1:]):
        if left[1] == right[1] and abs(left[2] - right[2]) == 8:
            pair = (left, right)
            break
    if pair is None:
        return True
    (field_a, _base_a, _disp_a, index_a), (field_b, _base_b, _disp_b, index_b) = pair
    byte_loads = [i for i, line in enumerate(lines) if line.startswith("ld.global.b8")]
    if not byte_loads or byte_loads[0] <= index_b:
        return True
    window = lines[index_a : byte_loads[0]]

    def compared_against_param(reg: str) -> bool:
        for line in window:
            match = _setp(line)
            if not match:
                continue
            left, right = match.group(2), match.group(3)
            for info in defs.values():
                if info[0] != "param":
                    continue
                if _ordered_gt(match.group(1), left, right, reg, info[2].group(1)):
                    return True
        return False

    def length_fits_remainder(length: str, offset: str) -> bool:
        remainders: list[str] = []
        for reg, (kind, _index, match) in defs.items():
            if kind != "sub":
                continue
            left, right = match.group(2), match.group(3)
            if right == offset and defs.get(left, ("",))[0] == "param":
                remainders.append(reg)
        for line in window:
            match = _setp(line)
            if not match:
                continue
            for remainder in remainders:
                if _ordered_gt(
                    match.group(1), match.group(2), match.group(3), length, remainder
                ):
                    return True
        return False

    per_record_safe = (
        compared_against_param(field_a) and length_fits_remainder(field_b, field_a)
    ) or (
        compared_against_param(field_b) and length_fits_remainder(field_a, field_b)
    )

    sums: list[str] = []
    for reg, (kind, _index, match) in defs.items():
        if kind != "add":
            continue
        if {match.group(2), match.group(3)} == {field_a, field_b}:
            sums.append(reg)
    overflow_checked = False
    sum_guard = False
    for total in sums:
        for line in window:
            match = _setp(line)
            if not match:
                continue
            pred, left, right = match.group(1), match.group(2), match.group(3)
            if pred in ("lt.s64", "le.s64", "lt.u64", "le.u64") and (
                (left == total and right in (field_a, field_b))
                or (right == total and left in (field_a, field_b))
            ):
                overflow_checked = True
            other = right
            if (
                pred == "gt.s64"
                and left == total
                and defs.get(other, ("",))[0] == "param"
            ):
                sum_guard = True
    safe_sum = bool(sums) and sum_guard and overflow_checked

    addresses = []
    for reg in (field_a, field_b):
        addresses.append(defs[reg][2].group(0).split(",", 1)[1].strip())
    reloaded = False
    for line in lines[index_b + 1 : byte_loads[-1]]:
        if not line.startswith("ld.global.b64"):
            continue
        for address in addresses:
            if address in line:
                reloaded = True
    return not (per_record_safe or safe_sum) or reloaded


def test_gpu_gapless_kernel_rejects_wrapping_segment_before_load(tmp_path: Path) -> None:
    ptx = _compile_gapless_ptx(tmp_path)
    assert "ld.global.b8" in ptx
    assert not gapless_kernel_accepts_wrapping_segment(ptx), (
        "gapless kernel still ranges the segment with a wrapping signed "
        "offset+length, or reloads those fields after the check"
    )
