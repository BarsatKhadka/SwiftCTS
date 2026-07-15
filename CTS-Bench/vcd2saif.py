#!/usr/bin/env python3
"""
VCD to SAIF converter (pure Python, no external VCD library).
Usage: python3 vcd2saif.py -o output.saif input.vcd
"""

import sys
import argparse
from collections import defaultdict


def tokenize(f):
    """Yield whitespace-delimited tokens from a VCD file."""
    for line in f:
        if isinstance(line, bytes):
            line = line.decode("ascii", errors="replace")
        for tok in line.split():
            yield tok


def convert(vcd_path: str, saif_path: str):
    toggle_count = defaultdict(int)
    time_in_0    = defaultdict(int)
    time_in_1    = defaultdict(int)
    time_in_x    = defaultdict(int)
    time_in_z    = defaultdict(int)
    last_value   = {}
    last_time    = {}
    total_time   = 0
    timescale    = "1 ns"
    scope_stack  = []
    sig_fullname = {}  # id_code -> full hierarchical name

    with open(vcd_path, "rb") as f:
        tokens = tokenize(f)
        try:
            while True:
                tok = next(tokens)

                # Header section
                if tok == "$timescale":
                    parts = []
                    while True:
                        t = next(tokens)
                        if t == "$end":
                            break
                        parts.append(t)
                    timescale = " ".join(parts)

                elif tok == "$scope":
                    next(tokens)          # scope type (module/begin/fork/task/function)
                    ident = next(tokens)  # scope name
                    next(tokens)          # $end
                    scope_stack.append(ident)

                elif tok == "$upscope":
                    next(tokens)  # $end
                    if scope_stack:
                        scope_stack.pop()

                elif tok == "$var":
                    _type  = next(tokens)   # var type
                    _size  = next(tokens)   # bit width
                    id_code = next(tokens)  # identifier code
                    reference = next(tokens)  # signal name
                    # consume optional bit-select and $end
                    while True:
                        t = next(tokens)
                        if t == "$end":
                            break
                    name = ".".join(scope_stack + [reference])
                    sig_fullname[id_code] = name
                    last_value[id_code] = "x"
                    last_time[id_code]  = 0

                elif tok == "$enddefinitions":
                    next(tokens)  # $end

                # Simulation section
                elif tok.startswith("#"):
                    total_time = int(tok[1:])

                elif tok[0] in ("0", "1", "x", "z", "X", "Z") and len(tok) > 1:
                    # scalar value change: "0id_code" or "1id_code"
                    new_val = tok[0].lower()
                    id_code = tok[1:]
                    if id_code not in sig_fullname:
                        continue
                    old_val = last_value.get(id_code, "x")
                    t       = last_time.get(id_code, 0)
                    dt      = total_time - t

                    if   old_val == "0": time_in_0[id_code] += dt
                    elif old_val == "1": time_in_1[id_code] += dt
                    elif old_val == "x": time_in_x[id_code] += dt
                    elif old_val == "z": time_in_z[id_code] += dt

                    if (old_val in ("0","1")) and (new_val in ("0","1")) and old_val != new_val:
                        toggle_count[id_code] += 1

                    last_value[id_code] = new_val
                    last_time[id_code]  = total_time

                elif tok == "b" or tok.startswith("b"):
                    # vector value change: "bXXXX id_code"
                    if tok == "b":
                        _vec = next(tokens)
                    # else vec is tok[1:]
                    _id = next(tokens)  # skip vectors for SAIF (not tracking bit-level)

        except StopIteration:
            pass

    # flush final state
    for id_code, val in last_value.items():
        dt = total_time - last_time.get(id_code, 0)
        if   val == "0": time_in_0[id_code] += dt
        elif val == "1": time_in_1[id_code] += dt
        elif val == "x": time_in_x[id_code] += dt
        elif val == "z": time_in_z[id_code] += dt

    # build hierarchical SAIF tree
    tree = {}

    def get_node(parts):
        node = tree
        for part in parts:
            node = node.setdefault(part, {})
        return node

    for id_code, fullname in sig_fullname.items():
        parts = fullname.split(".")
        node = get_node(parts[:-1])
        sigs = node.setdefault("__sigs__", {})
        sigs[parts[-1]] = id_code

    def write_instance(out, node, indent=0):
        pad = "  " * indent
        for key, child in node.items():
            if key == "__sigs__":
                continue
            out.write(f"{pad}(INSTANCE {key}\n")
            sigs = child.get("__sigs__", {})
            if sigs:
                out.write(f"{pad}  (NET\n")
                for sig, id_code in sigs.items():
                    tc = toggle_count.get(id_code, 0)
                    t0 = time_in_0.get(id_code, 0)
                    t1 = time_in_1.get(id_code, 0)
                    tx = time_in_x.get(id_code, 0)
                    tz = time_in_z.get(id_code, 0)
                    out.write(f"{pad}    ({sig} (T0 {t0}) (T1 {t1}) (TX {tx}) (TZ {tz}) (TC {tc}))\n")
                out.write(f"{pad}  )\n")
            write_instance(out, child, indent + 1)
            out.write(f"{pad})\n")

    with open(saif_path, "w") as out:
        out.write(f'(SAIF (VERSION "2.0") (DIRECTION "backward")\n')
        out.write(f'(TIMESCALE {timescale})\n')
        out.write(f'(DURATION {total_time})\n')
        write_instance(out, tree, indent=0)
        out.write(')\n')

    print(f"[vcd2saif] {vcd_path} -> {saif_path}  "
          f"({len(sig_fullname)} signals, duration={total_time})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("input")
    args = parser.parse_args()
    convert(args.input, args.output)
