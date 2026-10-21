"""Estimate per-kernel memory traffic from an XLA HLO dump.

Usage: python hlo_traffic.py results/de_objective_hlo.txt

For each kernel (fusion or custom-call) in the ENTRY computation, prints the
output shape, bytes written, and bytes read (sum of its inputs). This assumes
each input is read once in full and each output written once: a rough model.
"""
import re
import sys

BYTES = {"f32": 4, "s32": 4, "u32": 4, "s8": 1, "u8": 1, "pred": 1, "f16": 2, "bf16": 2}
SHAPE = re.compile(r"\b(f32|s32|u32|s8|u8|pred|f16|bf16)\[([\d,]*)\]")
LINE = re.compile(r"^\s*(?:ROOT )?%(\S+) = (.*?) ([\w-]+)\((.*?)\)(?:,|$)")


def nbytes(type_str):
    total = 0
    for dtype, dims in SHAPE.findall(type_str):
        n = 1
        for d in filter(None, dims.split(",")):
            n *= int(d)
        total += n * BYTES[dtype]
    return total


def main(path):
    text = open(path).read()
    entry = text[text.index("ENTRY"):]
    entry = entry[:entry.index("\n}")]
    sizes = {}
    rows = []
    for line in entry.splitlines()[1:]:
        m = LINE.match(line)
        if not m:
            continue
        name, out_type, op, args = m.groups()
        out_bytes = nbytes(out_type)
        sizes[name] = out_bytes
        if op in ("fusion", "custom-call"):
            inputs = re.findall(r"%([\w.\-]+)", args)
            read = sum(sizes.get(i, 0) for i in inputs)
            shape = " + ".join(f"{d}[{x}]" for d, x in SHAPE.findall(out_type))
            rows.append((name, shape, read, out_bytes))
    print(f"{'kernel':<32}{'output shape':<34}{'read MB':>10}{'write MB':>10}")
    for name, shape, r, w in rows:
        print(f"{name:<32}{shape:<34}{r/1e6:>10.1f}{w/1e6:>10.1f}")
    tr, tw = sum(r[2] for r in rows), sum(r[3] for r in rows)
    print(f"{'TOTAL':<66}{tr/1e6:>10.1f}{tw/1e6:>10.1f}")
    print(f"\nkernels: {len(rows)}, total traffic: {(tr + tw)/1e6:.0f} MB")


if __name__ == "__main__":
    main(sys.argv[1])
