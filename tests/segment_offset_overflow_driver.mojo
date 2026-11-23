# Drive the production dense-pack reader and host gapless extender.
#
# Run from the repo root:
#   mojo -I src tests/segment_offset_overflow_driver.mojo <pack_dir> get <node_id>
#   mojo -I src tests/segment_offset_overflow_driver.mojo <pack_dir> gapless <node_id> <orient> <query>
#
# Prints PATH=mmap or PATH=python, then either RESULT_LEN/RESULT or HITS/HITPATH/HITCS.
# A record that does not fit in sequences.bin must come back empty (no hit),
# including when offset and length are each in range of a signed word but
# their sum is not.

from std.collections import List
from std.sys import argv

from giraffe_gapless import gapless_extend_with_pack
from giraffe_pack import DensePack


def main() raises:
    var args = argv()
    if len(args) < 4:
        raise Error(
            "usage: segment_offset_overflow_driver.mojo <pack> get <node> | gapless <node> <orient> <query>"
        )
    var pack = DensePack(args[1])
    if pack._use_py:
        print("PATH=python")
    else:
        print("PATH=mmap")

    var mode = args[2]
    var node = Int(args[3])
    if mode == "get":
        var seq = pack.get(node)
        print("RESULT_LEN=" + String(seq.byte_length()))
        print("RESULT=[" + seq + "]")
        return
    if mode != "gapless":
        raise Error("unknown mode " + mode)
    if len(args) < 6:
        raise Error("gapless mode needs orient and query")
    var seeds = List[String]()
    var orient = args[4]
    var q = args[5]
    seeds.append(String(node) + ":" + orient + ":0")
    var hits = gapless_extend_with_pack(pack, "q", q, seeds)
    print("HITS=" + String(len(hits)))
    if len(hits) == 0:
        print("HITPATH=[]")
        print("HITCS=[]")
        return
    print("HITPATH=[" + hits[0].path + "]")
    print("HITCS=[" + hits[0].cs_tag + "]")
