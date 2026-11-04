"""The filesystem as the model holds it, read from its state, and what the circuit is doing.

`read` decodes the model's state in a cache (nothing is kept on the side): the file count, each
file's version and path from the DeltaNet version and path tables (layer 4), where rows are
dual-rail, so a bit is whichever of its two rails is set; and every link from the seek layer's
link rows (layer 6), each a pair in its key's signs and +1 or -1 in its value.

`Monitor` follows the token stream and lights the circuit's layers: the call parser while Qwen
writes a call, the lookup and the tables at a call's end, the path printer, the link bits and the
copy head as the response comes out.
"""
from qfs import circuit as F
from qfs.disk import COPY, COUNTL, LOOKUP, SEEK, TABLE

# the FS layers as the panel shows them: (name, what it holds or does)
LAYERS = [
    ("P", "DeltaNet", "lag registers: the tokens before"),
    ("1", "DeltaNet", "the call's registers, offsets"),
    ("2", "DeltaNet", "the path being read, file count"),
    ("3", "attention", "lookup: a path -> its file"),
    ("4", "DeltaNet", "version table, path table"),
    ("5", "DeltaNet", "held acks; MLP: forced tokens"),
    ("6", "attention", "seek: links, words"),
    ("7", "attention", "copy: the files' text"),
]


def bits01(row, n):
    """n dual-rail bits of a state row (rail 2b on means 1)."""
    return [int(row[2 * b] > row[2 * b + 1]) for b in range(n)]


def value(bs):
    return sum(b << k for k, b in enumerate(bs))


def read(cache):
    """{"count": n, "files": {I: (path tokens, V)}, "links": {(S, D)}, "rows": copy-layer rows,
    "lookup": lookup rows, "vbits": each slot's version bits} from a cache's state."""
    def state(li):            # a layer's state, or None before the first token
        r = getattr(cache.layers[li], "recurrent_states", None)
        return r[0][0] if r is not None and len(r) and r[0] is not None else None
    rec = {li: x for li in (TABLE, COUNTL) if (x := state(li)) is not None}
    if TABLE not in rec:
        return {"count": 0, "files": {}, "links": set(), "rows": 0, "lookup": 0,
                "vbits": [[0] * F.VB for _ in range(F.NFILE)]}
    table = rec[TABLE][:, :F.NFILE, :2 * F.IDB].float().cpu()
    count = value(bits01(rec[COUNTL][F.NSLOT + 1, 0, :2 * F.CB].float().cpu(), F.CB))
    files = {}
    for i in range(count):
        v = value(bits01(table[0, i], F.VB))
        name = []
        for j in range(F.NSLOT):
            t = value(bits01(table[1 + j, i], F.IDB))
            if t == (1 << F.IDB) - 1:
                break
            name.append(t)
        files[i] = (tuple(name), v)
    links = seek_links(cache, count)

    def length(li):
        c = cache.layers[li]
        return c.keys.shape[-2] if getattr(c, "keys", None) is not None else 0
    vbits = [bits01(table[0, i], F.VB) if i < count else [0] * F.VB for i in range(F.NFILE)]
    return {"count": count, "files": files, "links": links, "rows": length(COPY),
            "lookup": length(LOOKUP), "vbits": vbits}


def seek_links(cache, count):
    """{(S, D)}: the pairs whose link rows in the seek layer add up to +1."""
    c = cache.layers[SEEK]
    if getattr(c, "keys", None) is None or not c.keys.shape[-2]:
        return set()
    k, v = c.keys[0, 0].float().cpu(), c.values[0, 0, :, 1].float().cpu()
    at = F.ROT + 2 + len(F.SFEATS)
    rows = (v.abs() > 0.5).nonzero().flatten().tolist()
    tally = {}
    for r in rows:
        s_ = value([int(x > 0) for x in k[r, at:at + F.IB].tolist()])
        d_ = value([int(x > 0) for x in k[r, at + F.IB:at + 2 * F.IB].tolist()])
        tally[(s_, d_)] = tally.get((s_, d_), 0) + round(v[r].item())
    return {p for p, n in tally.items() if n > 0 and max(p) < count}


class Monitor:
    """Lights for the FS layers (and Qwen's own, as one row), from the token stream the bridge
    annotates: each token lights the layers that act on it, and the lights fade."""

    FADE = 0.8

    def __init__(self):
        self.lights = [0.0] * (len(LAYERS) + 1)    # the last one: Qwen's layers
        self.forced = 0                            # tokens the circuit printed

    def feed(self, t, phase, func=None):
        """phase: "qwen" (Qwen's text), "call" (Qwen writing a call), "response" (the circuit
        printing the tool response); func: the call's tool."""
        lit = []
        if phase in ("qwen", "call"):
            lit.append(len(LAYERS))
        if phase == "call":
            lit += [0, 1, 2]                       # the call parser: lags, registers, path slots
            if t == F.TCEND:
                lit += [3, 4, 5]                   # the call ends: lookup, tables, held acks
        elif phase == "response":
            self.forced += 1
            lit.append(5)                          # every response token is a forced token
            if F.MARK0 <= t < F.MARK0 + F.NFILE:
                lit.append(4)                      # a listed path: the path table
            if (func in ("search_files", "list_links", "list_backlinks")
                    and (F.MARK0 <= t < F.MARK0 + F.NFILE or F.SKIP0 <= t < F.SKIP0 + F.NFILE)):
                lit += [4, 6]                      # a file checked: the seek layer
            elif F.ACKW0 <= t < F.ACKR0:
                lit += [3, 4]                      # a write: lookup and the version table
            elif F.ACKR0 <= t < F.NONE:
                lit += [3, 4, 7]                   # a read: its file found, the copy begins
                                                   # (or its next segment)
            elif func == "read_file" and t < F.V_QWEN:
                lit.append(7)                      # copied text
            if t in (F.LINKED, F.REMOVED, F.LINKEDN, F.REMOVEDN):
                lit.append(6)
        for k in lit:
            self.lights[k] = 1.0

    def tick(self):
        self.lights = [x * self.FADE for x in self.lights]
