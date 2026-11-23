# Tests for missing base-quality synthesis in native MethylCall and GAF emit.
#
# These tests fail while original-sequence tags without bq:Z: are expanded
# one character at a time. They pass once that string is built in one
# allocation and original-sequence tags longer than MAX_ACCEPTED_OS_BASES
# are rejected before the quality string is built.
#
# Run from the repository root:
#   pixi run mojo -I src tests/test_missing_quality_dos.mojo

from std.python import Python, PythonObject
from std.time import perf_counter_ns

from giraffe_gaf_emit import format_gaf_line
from giraffe_hit import AlignmentHit
from mcall import _python_alignment_to_parsed


# Longer than an Illumina read, shorter than the few-hundred-thousand-base
# records that make per-character appends dominate the caller.
comptime MAX_ACCEPTED_OS_BASES = 100000

# 4x length. A per-character copy is ~16x time; one allocation is ~4x.
comptime RATIO_LIMIT = 8.0


def expect(cond: Bool, msg: String) raises:
    if not cond:
        raise Error(msg)


def repo_root() raises -> String:
    var os_mod = Python.import_module("os")
    var root = String(os_mod.getcwd())
    var flag = String(os_mod.path.isdir(root + "/engine"))
    if flag != "True":
        raise Error("run this file from the repository root")
    var sys_mod = Python.import_module("sys")
    sys_mod.path.insert(0, root)
    return root


def write_gaf(path: String, n: Int, bq: String, include_bq: Bool) raises:
    """One GAF record: mapq 60, match length n, os:Z of length n, optional bq."""
    var seq = "A" * n
    var line = (
        "read1\t"
        + String(n)
        + "\t0\t"
        + String(n)
        + "\t+\t>1\t"
        + String(n)
        + "\t0\t"
        + String(n)
        + "\t"
        + String(n)
        + "\t"
        + String(n)
        + "\t60\tcs:Z::"
        + String(n)
        + "\tos:Z:"
        + seq
        + "\trc:Z:CT"
    )
    if include_bq:
        line = line + "\tbq:Z:" + bq
    line = line + "\n"
    var builtins = Python.import_module("builtins")
    var fh = builtins.open(path, "w")
    fh.write(line)
    fh.close()


def first_alignment(work_dir: String) raises -> PythonObject:
    """Same batch reader and filter arguments native MethylCall uses."""
    var mcall_mod = Python.import_module("engine.mcall")
    var builtins = Python.import_module("builtins")
    var batch_iter = mcall_mod.iter_alignment_batches(
        work_dir,
        work_dir + "/missing.node.replacement.json",
        minimum_identity=20,
        minimum_mapq=0,
        discard_multimapped=True,
        genotyping_cytosine=False,
        batch_size=8,
    )
    var batch = builtins.next(batch_iter)
    # Finish the generator so its report write is not left pending.
    try:
        while True:
            _ = builtins.next(batch_iter)
    except:
        pass
    var n_frags = Int(py=builtins.len(batch))
    expect(n_frags == 1, "reader dropped or split the alignment")
    var frag = batch[0]
    var n_aln = Int(py=builtins.len(frag))
    expect(n_aln == 1, "reader dropped the alignment row")
    return frag[0]


def row_has_prefix(row: PythonObject, prefix: String) raises -> Bool:
    var builtins = Python.import_module("builtins")
    var n = Int(py=builtins.len(row))
    var i = 12
    while i < n:
        var tag = String(row[i]).strip()
        if tag.startswith(prefix):
            return True
        i += 1
    return False


def load_row(n: Int, bq: String, include_bq: Bool) raises -> PythonObject:
    var tempfile = Python.import_module("tempfile")
    var work = String(tempfile.mkdtemp(prefix="mg-bq-"))
    write_gaf(work + "/alignment.gaf", n, bq, include_bq)
    return first_alignment(work)


def elapsed_ms(t0: UInt, t1: UInt) -> Float64:
    return Float64(t1 - t0) / 1000000.0


def parse_timed(row: PythonObject) raises -> Tuple[Float64, Bool, Int, String]:
    var t0 = perf_counter_ns()
    var parsed = _python_alignment_to_parsed(row)
    var t1 = perf_counter_ns()
    return (
        elapsed_ms(t0, t1),
        parsed.tags_ok,
        parsed.phred_score.byte_length(),
        parsed.phred_score,
    )


def test_reader_yields_missing_quality_and_keeps_supplied() raises:
    var missing = load_row(32, "", False)
    expect(row_has_prefix(missing, "os:Z:"), "os tag missing from yielded row")
    expect(not row_has_prefix(missing, "bq:Z:"), "reader invented a quality tag")
    var dt: Float64
    var ok: Bool
    var nph: Int
    var phred: String
    dt, ok, nph, phred = parse_timed(missing)
    expect(ok, "short missing quality must still be callable")
    expect(nph == 32, "short synthetic quality has the wrong length")
    expect(phred == "I" * 32, "short synthetic quality is not Phred+33 I")
    _ = dt

    var supplied = load_row(32, "ABCD" * 8, True)
    expect(row_has_prefix(supplied, "bq:Z:"), "supplied quality was stripped")
    dt, ok, nph, phred = parse_timed(supplied)
    _ = dt
    _ = nph
    expect(ok, "supplied quality record was rejected")
    expect(phred == "ABCD" * 8, "supplied quality was replaced")


def test_missing_quality_scales_linearly() raises:
    # Warm the parser and the Python bridge before measuring.
    var warm = load_row(128, "", False)
    _ = parse_timed(warm)

    var small = load_row(16000, "", False)
    var large = load_row(64000, "", False)
    expect(not row_has_prefix(small, "bq:Z:"), "16kb record gained a quality tag")
    expect(not row_has_prefix(large, "bq:Z:"), "64kb record gained a quality tag")

    var dt_s: Float64
    var dt_l: Float64
    var ok: Bool
    var nph: Int
    var phred: String
    dt_s, ok, nph, phred = parse_timed(small)
    expect(ok, "16000-base missing quality was rejected")
    expect(nph == 16000, "16000-base quality length")
    expect(phred == "I" * 16000, "16000-base quality contents")

    dt_l, ok, nph, phred = parse_timed(large)
    expect(ok, "64000-base missing quality was rejected")
    expect(nph == 64000, "64000-base quality length")
    expect(phred == "I" * 64000, "64000-base quality contents")

    var ratio = dt_l / dt_s
    print("missing-quality ms", dt_s, dt_l, "ratio", ratio)
    expect(
        ratio < RATIO_LIMIT,
        "missing quality grew superlinearly: 64000/16000 time ratio "
        + String(ratio),
    )


def test_overlong_original_sequence_is_rejected() raises:
    comptime n = 1000000
    expect(n > MAX_ACCEPTED_OS_BASES, "fixture must sit above the accepted cap")
    var row = load_row(n, "", False)
    expect(row_has_prefix(row, "os:Z:"), "reader did not yield the long os tag")
    expect(not row_has_prefix(row, "bq:Z:"), "reader invented a quality tag")

    var t0 = perf_counter_ns()
    var rejected = False
    var tags_ok = False
    var phred_len = -1
    try:
        var parsed = _python_alignment_to_parsed(row)
        tags_ok = parsed.tags_ok
        phred_len = parsed.phred_score.byte_length()
    except:
        rejected = True
    var dt = elapsed_ms(t0, perf_counter_ns())
    print("overlong parse ms", dt, "rejected", rejected, "phred_len", phred_len)
    expect(
        dt < 3000.0,
        "overlong original sequence was not refused before the quadratic fill: "
        + String(dt)
        + " ms",
    )
    expect(
        rejected or (not tags_ok and phred_len != n),
        "original sequence above "
        + String(MAX_ACCEPTED_OS_BASES)
        + " bases was accepted",
    )


def emit_timed(n: Int) raises -> Tuple[Float64, String]:
    var hit = AlignmentHit(
        "read1",
        ">1",
        n,
        60,
        "cs:Z::" + String(n),
        "ri:i:1\tos:Z:" + ("A" * n) + "\trc:Z:CT",
    )
    var t0 = perf_counter_ns()
    var line = format_gaf_line(hit)
    var t1 = perf_counter_ns()
    return elapsed_ms(t0, t1), line


def bq_value_length(line: String) raises -> Int:
    var key = "\tbq:Z:"
    var at = line.find(key)
    expect(at >= 0, "emitted GAF has no bq:Z tag")
    var start = at + key.byte_length()
    var rest = line[byte = start : line.byte_length()]
    var tab = rest.find("\t")
    if tab < 0:
        return rest.byte_length()
    return tab


def test_gaf_emit_quality_scales_linearly() raises:
    var warm_ms: Float64
    var warm_line: String
    warm_ms, warm_line = emit_timed(128)
    _ = warm_ms
    _ = warm_line

    var dt_s: Float64
    var dt_l: Float64
    var line_s: String
    var line_l: String
    dt_s, line_s = emit_timed(16000)
    dt_l, line_l = emit_timed(64000)
    expect(bq_value_length(line_s) == 16000, "16kb emitted quality length")
    expect(bq_value_length(line_l) == 64000, "64kb emitted quality length")
    var ratio = dt_l / dt_s
    print("gaf-emit ms", dt_s, dt_l, "ratio", ratio)
    expect(
        ratio < RATIO_LIMIT,
        "GAF quality emit grew superlinearly: 64000/16000 time ratio "
        + String(ratio),
    )


def test_gaf_emit_long_read_is_bounded() raises:
    var t0 = perf_counter_ns()
    var refused = False
    var bq_len = -1
    try:
        var dt_unused: Float64
        var line: String
        dt_unused, line = emit_timed(1000000)
        _ = dt_unused
        bq_len = bq_value_length(line)
    except:
        refused = True
    var dt = elapsed_ms(t0, perf_counter_ns())
    print("long emit ms", dt, "refused", refused, "bq_len", bq_len)
    expect(
        dt < 3000.0,
        "GAF emit built a million-base quality string too slowly: "
        + String(dt)
        + " ms",
    )
    # One allocation may emit the tag. Refusal is also acceptable.
    # A per-character append does not finish inside the bound above.
    if not refused:
        expect(bq_len == 1000000, "emitted quality length does not match qlen")


def main() raises:
    _ = repo_root()
    var failed = 0

    try:
        test_reader_yields_missing_quality_and_keeps_supplied()
        print("PASS test_reader_yields_missing_quality_and_keeps_supplied")
    except e:
        print("FAIL test_reader_yields_missing_quality_and_keeps_supplied:", e)
        failed += 1

    try:
        test_missing_quality_scales_linearly()
        print("PASS test_missing_quality_scales_linearly")
    except e:
        print("FAIL test_missing_quality_scales_linearly:", e)
        failed += 1

    try:
        test_overlong_original_sequence_is_rejected()
        print("PASS test_overlong_original_sequence_is_rejected")
    except e:
        print("FAIL test_overlong_original_sequence_is_rejected:", e)
        failed += 1

    try:
        test_gaf_emit_quality_scales_linearly()
        print("PASS test_gaf_emit_quality_scales_linearly")
    except e:
        print("FAIL test_gaf_emit_quality_scales_linearly:", e)
        failed += 1

    try:
        test_gaf_emit_long_read_is_bounded()
        print("PASS test_gaf_emit_long_read_is_bounded")
    except e:
        print("FAIL test_gaf_emit_long_read_is_bounded:", e)
        failed += 1

    if failed != 0:
        raise Error(String(failed) + " test(s) failed")
    print("ALL TESTS PASSED")
