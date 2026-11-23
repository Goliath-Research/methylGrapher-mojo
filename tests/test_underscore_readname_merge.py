"""Spec for underscore read-name handling on the default cpu_vg Align path.

These tests fail while the vulnerability is present. A read name is caller
input. The converter appends ``_{conversion}_{shard}_{sequence}`` and the
merge step splits the vg GAF query name on every underscore, taking field 1
as the conversion and field 3 as the original sequence. A name that already
contains underscores therefore replaces the submitted bases, and a second
field shorter than three characters raises inside the merge worker. That
worker never posts its sentinel, so the writer blocks in ``Queue.get``.

The vg stand-in implements only the name and tag transforms in vg 1.70.0
(tag v1.70.0, libvgio aea4aa7931c3406b9bb3e8d3db6bc3e389eac9e6):

- ``get_next_alignment_from_fastq`` keeps the header from the character after
  ``@`` up to the first whitespace (``src/alignment.cpp``).
- Giraffe copies that alignment, and the only ``set_name`` in
  ``MinimizerMapper`` writes the mate name onto ``fragment_next`` /
  ``fragment_prev`` (``src/minimizer_mapper.cpp`` ``pair_all``).
- ``alignment_to_gaf`` copies ``aln.name()`` into GAF column 1 and, by
  default, emits ``bq``, ``cs``, and ``fn``/``fp``
  (``deps/libvgio/src/alignment_io.cpp``, ``alignment_emitter.cpp``).
- A mapped record is twelve populated columns. Unmapped ``*`` columns are
  not emitted here because the shard router drops them before the merge.

The stand-in does not choose a graph locus. It reports a perfect alignment
of the converted sequence to segment ``1`` so MethylCall scores the ``os``
tag against a known reference window.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
REF = "ACGTACGTACGTACGTACGT"
FORGED = "T" * len(REF)
QUAL = "I" * len(REF)
# Cytosines in CG context on REF (0-based). Submitted bases are C, so a
# faithful C2T call is methylated.
CG_POSITIONS = (1, 5, 9, 13, 17)

VG_STUB = textwrap.dedent(
    r'''
    #!/usr/bin/env python3
    """Name-preserving GAF emitter matching vg 1.70.0 giraffe for mapped reads."""
    import sys

    _WS = set(" \t\n\r\v\f")


    def qname_from_header(header: str) -> str:
        # alignment.cpp get_next_alignment_from_fastq: drop one leading byte
        # and cut at the first whitespace. FASTQ '@' is that leading byte.
        line = header.rstrip("\n")
        div = next((i for i, ch in enumerate(line) if ch in _WS), None)
        if div is None:
            return line[1:]
        return line[1:div]


    def read_fastq(path: str):
        recs = []
        with open(path, "r", encoding="utf-8") as fh:
            while True:
                header = fh.readline()
                if not header:
                    break
                seq = fh.readline().rstrip("\n")
                fh.readline()
                qual = fh.readline().rstrip("\n")
                recs.append((qname_from_header(header), seq, qual))
        return recs


    def emit(name: str, seq: str, qual: str, link: str) -> None:
        length = len(seq)
        # std::map order used by gafkluge operator<<: AS, bq, cs, fn/fp.
        sys.stdout.write(
            f"{name}\t{length}\t0\t{length}\t+\t>1\t{length}\t0\t{length}\t"
            f"{length}\t{length}\t60\t"
            f"AS:i:{length}\tbq:Z:{qual}\tcs:Z::{length}\t{link}\n"
        )


    fastqs = []
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "-f" and i + 1 < len(args):
            fastqs.append(args[i + 1])
            i += 2
        else:
            i += 1

    # GafAlignmentEmitter writes this header before any alignment.
    sys.stdout.write("@HD\tVN:Z:1.0\n")
    groups = [read_fastq(path) for path in fastqs]
    if len(groups) >= 2:
        for left, right in zip(groups[0], groups[1]):
            # pair_all: read 1 fragment_next = read 2 name; read 2
            # fragment_prev = read 1 name.
            emit(left[0], left[1], left[2], f"fn:Z:{right[0]}")
            emit(right[0], right[1], right[2], f"fp:Z:{left[0]}")
    elif groups:
        for name, seq, qual in groups[0]:
            emit(name, seq, qual, "fi:i:0")
    sys.exit(0)
    '''
).lstrip()


def _write_fastq(path: Path, name: str, seq: str) -> None:
    path.write_text(f"{name}\n{seq}\n+\n{QUAL}\n", encoding="utf-8")


def _install_vg(directory: Path) -> Path:
    vg = directory / "vg"
    vg.write_text(VG_STUB, encoding="utf-8")
    vg.chmod(0o755)
    return vg


def _touch_indexes(index_prefix: Path) -> None:
    for conversion in ("C2T", "G2A"):
        base = f"{index_prefix}.wl.{conversion}"
        for suffix in (".dist", ".giraffe.gbz", ".min"):
            Path(base + suffix).write_bytes(b"x")


def _run_cpu_vg_align(work: Path, index_prefix: Path, r1_name: str, r1_seq: str) -> None:
    """Converter plus shard router, the same calls Align makes for cpu_vg."""
    from engine import alignments, utility

    _touch_indexes(index_prefix)
    fq1 = work / "r1.fastq"
    fq2 = work / "r2.fastq"
    _write_fastq(fq1, r1_name, r1_seq)
    _write_fastq(fq2, "@mate", REF)
    utility.fastq_converter(
        str(fq1),
        str(fq2),
        str(work),
        compress=False,
        thread=1,
        directional=True,
        split_num=alignments.tmp_alignment_file_count,
    )
    alignments.alignment(
        work_dir=str(work),
        index_prefix=str(index_prefix),
        output_format="gaf",
        thread=1,
        directional=True,
        compress=False,
        vg_path=str(_install_vg(work)),
        align_engine="cpu_vg",
    )


def _cg_plus_calls(work: Path):
    from engine.mcall import alignment_parse, alignment_to_methylation

    calls = []
    groups = alignment_parse(
        str(work),
        {},
        minimum_identity=0,
        minimum_mapq=0,
        discard_multimapped=True,
    )
    for best in groups:
        if not best:
            continue
        frag, _gcall = alignment_to_methylation(
            best,
            {"1": REF},
            cg_only=True,
            perform_gcall=False,
        )
        for segment_id, pos, strand, category, methylated in frag:
            if segment_id == "1" and strand == "+" and category == "CG":
                calls.append((pos, methylated))
    return calls


def test_underscore_name_cannot_replace_submitted_bases(tmp_path: Path):
    """A planted sequence in the read name must not be what MethylCall scores."""
    work = tmp_path / "work"
    work.mkdir()
    index_prefix = tmp_path / "idx"
    _run_cpu_vg_align(work, index_prefix, f"@sample_C2T_0_{FORGED}", REF)

    from engine.alignments import alignment_merge_main

    alignment_merge_main(str(work), worker_num=1)
    merged = (work / "alignment.gaf").read_text(encoding="utf-8")

    # Field 3 of the attacker name is the forged sequence. It must not survive
    # as the original-sequence tag the caller scores.
    assert f"os:Z:{FORGED}" not in merged

    calls = _cg_plus_calls(work)
    # Rejecting the read yields no CG calls. Keeping it must score the
    # submitted cytosines, which match the reference, as methylated.
    if calls:
        assert {pos for pos, _meth in calls} == set(CG_POSITIONS)
        assert all(meth == 1 for _pos, meth in calls)


@pytest.mark.parametrize("read_name", ["@a_b_0", "@a_bb_0", "@a__0"])
def test_short_second_field_does_not_block_merge(tmp_path: Path, read_name: str):
    """Second fields of length 0, 1, and 2 must not kill the merge worker.

    The shard router keeps a line whose third underscore field is an integer.
    These names put a short token in field 1 and ``0`` in field 2, which is
    the shape that reaches ``tmp_gaf_processing``.
    """
    work = tmp_path / "work"
    work.mkdir()
    index_prefix = tmp_path / "idx"
    _run_cpu_vg_align(work, index_prefix, read_name, REF)

    driver = tmp_path / "merge_driver.py"
    driver.write_text(
        "import sys\n"
        "from engine.alignments import alignment_merge_main\n"
        "alignment_merge_main(sys.argv[1], worker_num=1)\n",
        encoding="utf-8",
    )
    proc = subprocess.Popen(
        [sys.executable, str(driver), str(work)],
        cwd=str(REPO),
        env={**os.environ, "PYTHONPATH": str(REPO)},
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = proc.communicate(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(timeout=10)
        pytest.fail(
            f"alignment_merge_main did not finish for read name {read_name}; "
            "the writer is blocked on a merge worker that died"
        )
    assert proc.returncode == 0, (
        f"merge failed for {read_name}:\n{stderr.decode('utf-8', 'replace')}"
    )
    assert stdout is not None
