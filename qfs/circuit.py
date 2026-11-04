"""qwenFS: a filesystem compiled into Qwen3.5 layers.

Qwen writes ordinary Qwen3.5 tool calls; the circuit reads them as they are written and, after
</tool_call>, prints the tool response itself, in a user turn as Qwen's chat template has it,
then opens a new assistant turn:

    write_file(name, text)      -> saved, then the path         ([w I V] ack, then a marker)
    append_file(name, text)     -> saved, then the path         (the same, as a new segment)
    read_file(name)             -> the stored text, copied token for token, or "not found"
    list_files()                -> one path per line, or "empty"
    link_files(name, target)    -> linked, the two paths
    unlink_files(name, target)  -> removed, the two paths
    list_links(file)            -> the files a file links to, or "empty"
    list_backlinks(path)        -> the files that link to a file, or "empty"
    search_files(query)         -> the files whose text has the word, or "empty"

Responses look like

    </tool_call>
    <|im_start|>user
    <tool_response>
    [w I V]saved
    [m I]notes/todo.txt
    </tool_response>
    <|im_start|>assistant
    <think>

    </think>

Acks ([w I V], [r I V], [none]), list markers ([m I]) and skips ([s I]) are FS tokens: they
carry a file's index and version into the circuit's state. To Qwen's layers an ack or a skip
reads as a newline and a marker as "-". A search goes through the files in order, printing a
marker and the path of each file that has the word and a skip for each that does not.

How a call is read. The circuit keys on single tokens that Qwen's tool-call format always uses:
"=name", "=text", "=target", "=file", "=path", "=query", ">", newline, "</", <tool_call>,
</tool_call>, <tool_response>, </tool_response>. A call's kind comes from its parameters (a text
makes a write, a target a link, a file a link listing, a path a backlink listing, a query a
search, none a file listing); only unlink_files and append_file are told apart by their
function name tokens, "unlink" and "append", which have no other spelling.

Segments. Every write or append is a version V of its file (a segment), its text keyed by
(I, V, offset). A file is its segments from its base (its last write_file) to its latest
version: an append adds a segment and keeps the base. A read starts at the base, and at the
end of each segment but the last prints the next segment's read ack instead of
</tool_response>: the ack restarts the offset counter, so a file holds up to 63 segments of up
to ~590 tokens each.

Where things live.
  file text      the copy layer's attention log, keyed by (index, version, offset)
  names          a lookup attention head: a path -> its index (the path's end is the not-found
                 fallback, scoring between an exact match and any near miss)
  versions, bases, paths, file count     DeltaNet rows (one row per file)
  links          a table of link bits: row (source, target) of DeltaNet state, 4 sources x 32
                 targets per head, written by link and emptied by unlink. Rows store the target's
                 column in eight heads and the source's in eight more, so a multi-hot query over
                 a file's 32 rows reads all its links, or all its backlinks, as bits.
  offsets        a decaying DeltaNet counter, alpha = 2^(-1/16) per token, read in the head
                 norm's eps regime and decoded by thresholds (exact to 600 tokens)
  words          a search head on the copy layer: each word of a written text is a row, keyed
                 at the token after it by (index, base, the word's form; qfs/words.py), and a
                 search asks for one file's current base and the query's form per token

Layer P   DeltaNet  lag registers: flags of the previous one and two tokens, the forms and word
                    flags of the previous three
          MLP       the newlines that start a path (name, target, file, path, query) or a text
Layer 1   DeltaNet  counters (path slot, data offset) and call registers (naming, pending, has
                    text, in text, unlink, target / file / path seen, listing, list index, tail)
          MLP       offset thermometer, path slots, GO (the newline that ends a path), keys
Layer 2   DeltaNet  path slots (the path being read, token by token), op register, file count,
                    the fourth token back's word flag, past the text's "</", a search call, an
                    append
          MLP       offset bits, file count one-hot, link-listing flag, a word's end, empty
                    query slots, the file a search checks
Layer 3   attention lookup: the path at GO against the paths at earlier acks
          MLP       table query, the query's length
Layer 4   DeltaNet  version table, path table (index -> path, for listings and echoes), base
                    table (read at a search's checked file and at GO), the read file's version
          MLP       the write and read acks at GO, empty path slots, a read's next segment
Layer 5   DeltaNet  hold: the acks, the call's source and a link's target, the written text's
                    base, from their GOs
          MLP       the forced tokens: the response frame, echoes, the path printer, link-bit
                    write one-hots and listing queries
Layer 6   DeltaNet  link bits
          MLP       the listed file's links as clean bits; the search features: a word's key at
                    the token after it, a check's query
Layer 7   attention copy: a read's (index, version, offset) against the stored text; search:
                    a checked file's current version and the query against the words
          MLP       exit: the output code into the id dims, a read's next segment, link
                    listings (the next set bit), search results (a marker or a skip per file),
                    and every other FS dim cleared

Exactness. Every token carries a ballast, so each FS norm is a fixed scale (256). DeltaNet
registers pad their set rails to 2, 8, 32 or 128, so reads come back exactly 0/1 or +-1. Bits
that enter DeltaNet values are dual-rail; a key that is off sits far below 0 (l2norm would
rescale any small value to full size), and so does any unit one flag short of firing.
"""
import math

import torch
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

# Qwen3.5 tokens the circuit keys on
NL, GT, LTS, EQNAME, EQTEXT, OK, NL2 = 198, 29, 510, 37943, 46696, 547, 271
TCEND, TRESP, TRESPEND = 248059, 248066, 248067
IMSTART, USER, ASSISTANT, THINK, THINKEND, NOT, FOUND_ = 248045, 846, 74455, 248068, 248069, 1856, 1669
TCSTART, EMPTYW, DASH = 248058, 3092, 12
EQTARGET, EQFILE, EQPATH, EQQUERY = 93435, 57242, 79114, 76478
V_QWEN = 248320
IB, VB = 5, 6
ACKW0, ACKR0, NONE = 122 << 11, 123 << 11, 124 << 11   # id = base + (I << VB) + V
MARK0 = 125 << 11                  # list markers: MARK0 + I
SKIP0 = 126 << 11                  # search skips: SKIP0 + I
UNLINK_TOKENS = (64778, 45896)     # "unlink", " unlink"
APPEND = 4928                      # "append"
SAVED, LINKED, REMOVED, TO = 31617, 41710, 44232, 951
IDB, OB = 18, 11
NMAX = 600
NSLOT = 8
NFILE = 1 << IB
VOCAB = SKIP0 + NFILE
S = 3                              # a searched word's tokens, at most (qfs.words.S)
EMPTYC = (1 << IDB) - 1            # an empty slot's code: all +1

DK, HD, ROT = 128, 256, 64
K = 128.0
Y = -3.125
OFF, H = 2048, 3584
NBAL, BAL = 14, 4096.0             # rms = sqrt(14 * 4096^2 / 3584) = 256 exactly
SCALE = 256.0
EPS = 1.0 / 64
LAYERS = ["linear_attention", "linear_attention", "linear_attention", "full_attention",
          "linear_attention", "linear_attention", "linear_attention", "full_attention"]
LOOKUP, PAIR, COPY, TABLE, COUNTL, DATAL = 3, 6, 7, 4, 2, 1     # layer indices


def ackw(i, v):
    return ACKW0 + (i << VB) + v


def ackr(i, v):
    return ACKR0 + (i << VB) + v


class Dims:
    def __init__(self, start=0):
        self.n = start

    def take(self, k=1):
        r = list(range(self.n, self.n + k))
        self.n += k
        return r if k > 1 else r[0]


D = Dims(OFF)
B_ = D.take(NBAL)
ONE, ONE2 = D.take(), D.take()
IDC = D.take(IDB)
FLAG_TOKENS = {NL: "NL", GT: "GT", LTS: "LTS", EQNAME: "EQNAME", EQTEXT: "EQTEXT", OK: "OK",
               TCEND: "TCEND", TRESP: "TRESP", TRESPEND: "TRESPEND", IMSTART: "IMSTART",
               USER: "USER", ASSISTANT: "ASSISTANT", THINK: "THINK", THINKEND: "THINKEND",
               NL2: "NL2", NOT: "NOT", FOUND_: "FOUND", TCSTART: "TCSTART", EMPTYW: "EMPTYW",
               EQTARGET: "EQTARGET", EQFILE: "EQFILE", EQPATH: "EQPATH", EQQUERY: "EQQUERY",
               SAVED: "SAVED",
               LINKED: "LINKED", REMOVED: "REMOVED", TO: "TO",
               APPEND: "APPEND", **{t: "UNLINK" for t in UNLINK_TOKENS}}
FL = {name: D.take() for name in dict.fromkeys(FLAG_TOKENS.values())}
ISACKW, ISACKR, ISNONE, ISMARK, ISSKIP = (D.take() for _ in range(5))
ACKI, ACKV = D.take(IB), D.take(VB)
LAG1 = {n: D.take() for n in ("NL", "GT", "TCEND", "TRESP", "TRESPEND", "OK", "USER",
                              "ASSISTANT", "THINK", "THINKEND", "NOT", "FOUND", "NONE",
                              "EMPTYW", "ACKW", "SAVED", "LINKED", "REMOVED", "TO", "LTS")}
LAG2 = {n: D.take() for n in ("EQNAME", "EQTEXT", "EQTARGET", "EQFILE", "EQPATH", "EQQUERY")}
NSTART, TSTART, GSTART, FSTART, BSTART, QSTART = (D.take() for _ in range(6))
OCNT, CNT, NAMING, PENDING, WROTE, TAIL, INCALL, LISTMODE = (D.take() for _ in range(8))
PTGT, SAWTGT, SAWFILE, UNL, SAWPATH, INTEXT, TOSEEN = (D.take() for _ in range(7))
LI = D.take(IB)
U = D.take(NMAX)
SLOTRAW = D.take(NSLOT)
LS0 = D.take()                     # with SLOTRAW: the list slot one-hot (n = 0..NSLOT)
LIOH = D.take(NFILE)
GO, ISCREATE = D.take(), D.take()
CINC = D.take(IB)
TK = D.take(NFILE)
NAME = [D.take(IDB) for _ in range(NSLOT)]
RI, RV = D.take(IB), D.take(VB)
MW, MR = D.take(), D.take()
COUNT = D.take(IB)
QF, QB = D.take(NFILE), D.take(NFILE)   # link-bit queries (made late; free early dims)
# search. A token's word form (right-aligned, slot 0 the last token) and flags (a word token,
# one with a leading space, one whose form is a single token): from qfs.words, in the embeddings
WFORM = [D.take(IDB) for _ in range(S)]
WD, SP, X1 = D.take(), D.take(), D.take()
# the previous tokens' (layer P): the last token's whole form, the two before it their slot 0;
# each with its flags; the fourth token back's word flag (layer 2)
L1W = [D.take(IDB) for _ in range(S)]
L2W, L3W = D.take(IDB), D.take(IDB)
LF = [(D.take(), D.take(), D.take()) for _ in range(3)]          # (WD, SP, X1) of lags 1-3
L4WD, PASTLTS, SAWQUERY = D.take(), D.take(), D.take()
WEND = D.take(S)                   # this token ends a word of 1, 2 or 3 tokens (in a text)
QE = D.take(S + 1)                 # query slot j is empty
CK, CHK = D.take(NFILE), D.take()  # the file a search checks at this token (one-hot); a check
QLEN, QV = D.take(S), D.take()     # the query's length (one-hot), a query that can match
BCK = D.take(VB)                   # the checked file's base, 0/1
XI, XV = D.take(IB), D.take(VB)    # search features: index, base, form (+-1)
XW = [D.take(IDB) for _ in range(S)]
XT, SFOUND = D.take(), D.take()    # a word's row; the checked file has the word
# segments: an append call; the base at GO; the read file (one-hot) and its version; a read's
# segment is not the last; the written text's base (+-1, made at GO and held)
APP = D.take()
BGO = D.take(VB)
RIOH, TRD, MORE = D.take(NFILE), D.take(VB), D.take()
TRC, RVC = D.take(VB), D.take(VB)  # the read file's version and the read's segment, clean 0/1
HBS, HB = D.take(VB), D.take(VB)
assert D.n <= H, D.n
# features made from layer 2's MLP on, which clears the thermometer U, live in U's dims (the
# file count one-hot is written by that same MLP: the clear and the write add up)
Late = Dims(U[0])
CO = Late.take(NFILE)              # file count one-hot
KOFF, QOFF = Late.take(OB), Late.take(OB)
LL = Late.take()                   # a link listing call (list_links or list_backlinks)
GOS, GOT = Late.take(), Late.take()  # GO of a name/file path, GO of a target path
HSRC, HDST = Late.take(IB), Late.take(IB)
HSF, HDF = Late.take(), Late.take()
# link-bit table: write one-hots at a link's target GO (source, target), unlink-gated values,
# the listing queries, the reads and their clean bits
SOHW, DOHW = Late.take(NFILE), Late.take(NFILE)
SV, DV = Late.take(NFILE), Late.take(NFILE)
PF, PB = Late.take(NFILE), Late.take(NFILE)
BIT = Late.take(NFILE)
FOUND = Late.take()
FIDX = Late.take(IB)
TQ = Late.take(NFILE)
TBIT = Late.take(VB)
WACK, RACK = Late.take(IDB), Late.take(IDB)
HW, HR = Late.take(IDB), Late.take(IDB)
OUTC = Late.take(IDB)
NT = [Late.take(IDB) for _ in range(NSLOT)]     # name table read: the listed file's path
NTE = Late.take(NSLOT)                          # path slot is empty
assert Late.n <= U[-1] + 1, Late.n


def bits(x, n):
    return [(x >> i) & 1 for i in range(n)]


def pm(x, n):
    return [2 * b - 1 for b in bits(x, n)]


def code(t):
    return pm(t, IDB)


class MLP:
    """Units silu(gate . x) * (up . x), collected and then written into a Qwen3.5 MLP."""

    def __init__(self):
        self.units = []

    def unit(self, gate, up, down):
        self.units.append((gate, up, down))

    def relu(self, lin, down):
        """relu(lin) for integer-valued lin, written to `down` (dict dim -> weight)."""
        self.unit({d: K * w for d, w in lin.items()}, {ONE: 1 / K}, down)

    def all_of(self, flags, down, nots=()):
        """1 when every 0/1 flag is set and every `nots` flag is clear; one flag short gives
        relu(-1), never relu(0): with approximate flags a 0 could come out a hair either side,
        and a DeltaNet key's l2norm would blow that up into a full write."""
        lin = {f: 2.0 for f in flags}
        for f in nots:
            lin[f] = lin.get(f, 0.0) - 2.0
        lin[ONE] = -(2.0 * len(flags) - 1.0)
        self.relu(lin, down)

    def gated(self, x, out, flags, nots=()):
        """out += x (a +-1 dim) when every flag is set and every `nots` flag clear, else 0."""
        f = {d: 2.0 for d in flags}
        for d in nots:
            f[d] = f.get(d, 0.0) - 2.0
        c = -2.0 * len(flags)
        self.relu({x: 1.0, **f, ONE: c}, {out: 1.0})
        self.relu({x: -1.0, **f, ONE: c}, {out: -1.0})

    def gated_any(self, x, out, ors, flags=()):
        """out += x (a +-1 dim) when one of `ors` (at most one is ever set) and every flag is
        set, else 0."""
        f = {d: 2.0 for d in (*ors, *flags)}
        c = -2.0 * (1 + len(flags))
        self.relu({x: 1.0, **f, ONE: c}, {out: 1.0})
        self.relu({x: -1.0, **f, ONE: c}, {out: -1.0})

    def match(self, code_dims, value, flags=(), pm_dims=True, down=None, nots=()):
        """relu(1) when the bits in code_dims spell `value`, every flag is set and every
        `nots` flag is clear."""
        n = len(code_dims)
        vb = bits(value, n)
        lin = {d: (1.0 if b else -1.0) for d, b in zip(code_dims, vb)}
        const = -(n - 1) if pm_dims else (n - sum(vb)) - (n - 1)
        for f in flags:
            lin[f] = lin.get(f, 0.0) + 2.0
        for f in nots:
            lin[f] = lin.get(f, 0.0) - 2.0
        lin[ONE] = const - 2.0 * len(flags)
        self.relu(lin, down)

    def write(self, mlp):
        with torch.no_grad():
            for i, (g, u, d) in enumerate(self.units):
                for k, w in g.items():
                    mlp.gate_proj.weight[i, k] = w
                for k, w in u.items():
                    mlp.up_proj.weight[i, k] = w
                for k, w in d.items():
                    mlp.down_proj.weight[k, i] = w


def codes(dims, t):
    return {d: float(c) for d, c in zip(dims, code(t))}


class Delta:
    """Weights of one Gated DeltaNet layer, head by head."""

    def __init__(self, net):
        self.n = net
        self.kd = net.key_dim

    def q(self, h, row, lin, tap=-1):
        self._chan(h * DK + row, lin, tap)

    def k(self, h, row, lin, tap=-1):
        self._chan(self.kd + h * DK + row, lin, tap)

    def v(self, h, col, lin, tap=-1):
        self._chan(2 * self.kd + h * DK + col, lin, tap)

    def _chan(self, c, lin, tap):
        self.n.in_proj_qkv.weight[c, :] = 0
        for d, w in lin.items():
            self.n.in_proj_qkv.weight[c, d] = w
        self.n.conv1d.weight[c, 0, :] = 0
        self.n.conv1d.weight[c, 0, tap] = 1

    def z(self, h, lin):
        for c in range(DK):
            for d, w in lin.items():
                self.n.in_proj_z.weight[h * DK + c, d] = w

    def beta(self, h, lin):
        for d, w in lin.items():
            self.n.in_proj_b.weight[h, d] = w

    def decay(self, h, lin):
        for d, w in lin.items():
            self.n.in_proj_a.weight[h, d] = w

    def out(self, h, col, dim, w):
        self.n.out_proj.weight[dim, h * DK + col] = w

    def register(self, h, key, fields, tap=-1, n_set=None):
        """A one-row register written where the key form is on and read at every token.
        key: dict dim -> weight of a form that is ~1 where it writes and <= 0 elsewhere.
        fields: [(kind, lin, out)]: "pm" (+-1, dual-rail, read back +-1), "bit01" (0/1,
        dual-rail, read back 0/1), "01" (one-hot group: exactly one is 1 at each write).

        Reads are exact: constant rails pad the set rails to n in {2, 8, 32, 128}, so the head
        norm scales a set rail by sqrt(128 / n), a power of two, and the gate is silu(32) = 32;
        the read gain is a power of two and +-1 comes back as exactly +-1."""
        self.q(h, 0, {ONE: 20.0})
        self.k(h, 0, {d: 20.0 * w for d, w in key.items()})
        self.beta(h, {ONE: 30.0})
        self.decay(h, {ONE: -100.0})
        self.z(h, {ONE: 32.0})
        rails = []
        for kind, lin, out in fields:
            c = lin.get(ONE, 0.0)
            if kind == "pm":
                rails.append(({**{d: 10.0 * w for d, w in lin.items()}, ONE: 10.0 + 10.0 * c}, out, 1.0))
                rails.append(({**{d: -10.0 * w for d, w in lin.items()}, ONE: 10.0 - 10.0 * c}, out, -1.0))
            elif kind == "bit01":
                rails.append(({d: 20.0 * w for d, w in lin.items()}, out, 1.0))
                rails.append(({**{d: -20.0 * w for d, w in lin.items()}, ONE: 20.0 - 20.0 * c}, None, 0.0))
            else:
                rails.append(({d: 20.0 * w for d, w in lin.items()}, out, 1.0))
        count = sum(1 for kind, _, _ in fields if kind != "01")
        count += any(kind == "01" for kind, _, _ in fields)
        n_set = next(n for n in (2, 8, 32, 128) if n >= count)
        rails += [({ONE: 20.0}, None, 0.0)] * (n_set - count)
        assert len(rails) <= DK, len(rails)
        gain = math.sqrt(DK / n_set) * 32.0
        for col, (lin, out, sign) in enumerate(rails):
            self.v(h, col, lin, tap)
            if out is not None and sign:
                self.out(h, col, out, sign / gain)


def common(net):
    net.A_log.zero_()
    net.dt_bias.zero_()
    net.norm.weight.fill_(1.0)


def counter(d, h, anchor, out):
    """alpha^n since the last anchor token, read in the norm's eps regime."""
    d.q(h, 0, {ONE: 1e-5})
    d.q(h, 1, {ONE: 20.0})
    d.k(h, 0, {ONE: 20.0})
    d.v(h, 0, {a: 20.0 for a in anchor})
    d.beta(h, {ONE: -30.0, **{a: 60.0 for a in anchor}})
    d.decay(h, {ONE: Y, **{a: 200.0 - Y for a in anchor}})
    d.z(h, {ONE: 1.0})
    d.out(h, 0, out, 100.0)


def layer_p(net):
    d = Delta(net)
    always = {ONE: 1.0}
    src = {**FL, "NONE": ISNONE, "ACKW": ISACKW}
    d.register(0, always, [("bit01", {src[n]: 1.0}, LAG1[n]) for n in LAG1], tap=-2)
    d.register(1, always, [("bit01", {FL[n]: 1.0}, LAG2[n]) for n in LAG2], tap=-3)
    # word forms and flags: the last token's whole form, the two before it their last slot
    flags = lambda k: [("bit01", {f: 1.0}, o) for f, o in zip((WD, SP, X1), LF[k])]
    for r in range(S):
        fields = [("pm", {WFORM[r][b]: 1.0}, L1W[r][b]) for b in range(IDB)]
        d.register(2 + r, always, fields + (flags(0) if r == 0 else []), tap=-2)
    for k, (out, tap) in enumerate(((L2W, -3), (L3W, -4)), 1):
        fields = [("pm", {WFORM[0][b]: 1.0}, out[b]) for b in range(IDB)]
        d.register(1 + S + k, always, fields + flags(k), tap=tap)
    common(net)


def mlp_p():
    m = MLP()
    # a newline after "=name>" / "=text>": Qwen writes it as \n or, sometimes, as \n\n
    m.relu({FL["NL"]: 2.0, FL["NL2"]: 2.0, LAG1["GT"]: 2.0, LAG2["EQNAME"]: 2.0, ONE: -5.0}, {NSTART: 1.0})
    m.relu({FL["NL"]: 2.0, FL["NL2"]: 2.0, LAG1["GT"]: 2.0, LAG2["EQTEXT"]: 2.0, ONE: -5.0}, {TSTART: 1.0})
    m.relu({FL["NL"]: 2.0, FL["NL2"]: 2.0, LAG1["GT"]: 2.0, LAG2["EQTARGET"]: 2.0, ONE: -5.0}, {GSTART: 1.0})
    m.relu({FL["NL"]: 2.0, FL["NL2"]: 2.0, LAG1["GT"]: 2.0, LAG2["EQFILE"]: 2.0, ONE: -5.0}, {FSTART: 1.0})
    m.relu({FL["NL"]: 2.0, FL["NL2"]: 2.0, LAG1["GT"]: 2.0, LAG2["EQPATH"]: 2.0, ONE: -5.0}, {BSTART: 1.0})
    m.relu({FL["NL"]: 2.0, FL["NL2"]: 2.0, LAG1["GT"]: 2.0, LAG2["EQQUERY"]: 2.0, ONE: -5.0}, {QSTART: 1.0})
    return m


def layer1(net):
    d = Delta(net)
    counter(d, 0, [NSTART, GSTART, FSTART, BSTART, QSTART, ISMARK], OCNT)
    counter(d, 1, [TSTART, ISACKR, ISNONE], CNT)
    # naming: +1 from a name's start to the next "</"
    starts = {NSTART: 1.0, GSTART: 1.0, FSTART: 1.0, BSTART: 1.0, QSTART: 1.0}
    d.register(2, {**starts, FL["LTS"]: 1.0}, [("pm", {**starts, FL["LTS"]: -1.0}, NAMING)])
    # pending: a call in progress, from a name's start until </tool_response>
    d.register(3, {NSTART: 1.0, FL["TRESPEND"]: 1.0}, [("bit01", {NSTART: 1.0}, PENDING)])
    # wrote: the call has a text parameter
    d.register(4, {NSTART: 1.0, TSTART: 1.0}, [("bit01", {TSTART: 1.0}, WROTE)])
    # tail: from </tool_response> to the token after </think> (the new assistant turn's header)
    d.register(5, {FL["TRESPEND"]: 1.0, LAG1["THINKEND"]: 1.0}, [("bit01", {FL["TRESPEND"]: 1.0}, TAIL)])
    # incall: from <tool_call> to </tool_response>; listmode: from the first list marker
    d.register(6, {FL["TCSTART"]: 1.0, FL["TRESPEND"]: 1.0}, [("bit01", {FL["TCSTART"]: 1.0}, INCALL)])
    d.register(7, {ISMARK: 1.0, FL["TRESPEND"]: 1.0}, [("bit01", {ISMARK: 1.0}, LISTMODE)])
    # li: the index of the file being listed (a marker's, or a search's skip)
    d.register(8, {ISMARK: 1.0, ISSKIP: 1.0}, [("pm", {ACKI[b]: 1.0}, LI[b]) for b in range(IB)])
    # the path being read is a link's target; the call has a target / a file parameter
    d.register(9, starts, [("bit01", {GSTART: 1.0}, PTGT)])
    d.register(10, {GSTART: 1.0, FL["TCSTART"]: 1.0}, [("bit01", {GSTART: 1.0}, SAWTGT)])
    d.register(11, {FSTART: 1.0, FL["TCSTART"]: 1.0}, [("bit01", {FSTART: 1.0}, SAWFILE)])
    # the call is unlink_files (its function name token)
    d.register(12, {FL["UNLINK"]: 1.0, FL["TCSTART"]: 1.0}, [("bit01", {FL["UNLINK"]: 1.0}, UNL)])
    d.register(13, {BSTART: 1.0, FL["TCSTART"]: 1.0}, [("bit01", {BSTART: 1.0}, SAWPATH)])
    # in text: from the newline that opens a write's text to just after </tool_call>
    d.register(14, {TSTART: 1.0, LAG1["TCEND"]: 1.0}, [("bit01", {TSTART: 1.0}, INTEXT)])
    # a link echo is past its "to": from that token to the next <tool_response>
    d.register(15, {FL["TO"]: 1.0, FL["TRESP"]: 1.0}, [("bit01", {FL["TO"]: 1.0}, TOSEEN)])
    common(net)


def mlp1(xs):
    """xs[n]: a counter as the MLPs see it, n = 0..NMAX (from calibrate)."""
    m = MLP()

    def thermo(src, j, down):
        # [n >= j]; gates stay below 128 so a pair's halves round alike in bf16
        tau = math.sqrt(xs[j - 1] * xs[j])
        kk = 126.0 / tau
        m.unit({ONE: 126.0, ONE2: 1.0, src: -kk}, {ONE: 1.0}, down)
        m.unit({ONE: 126.0, src: -kk}, {ONE: 1.0}, {k: -w for k, w in down.items()})

    for j in range(1, NMAX + 1):
        thermo(CNT, j, {U[j - 1]: 1.0})
    for j in range(1, NSLOT + 2):
        down = {}
        if j <= NSLOT:
            down[SLOTRAW[j - 1]] = 1.0
        if j >= 2:
            down[SLOTRAW[j - 2]] = -1.0
        else:
            down[LS0] = -1.0
        thermo(OCNT, j, down)
    m.relu({ONE: 1.0}, {LS0: 1.0})

    for i in range(NFILE):
        m.match(LI, i, down={LIOH[i]: 1.0})
    # GO: the newline that ends a name (naming, and not the newline that started it)
    m.relu({FL["NL"]: 2.0, FL["NL2"]: 2.0, NAMING: 2.0, NSTART: -2.0, ONE: -3.0}, {GO: 1.0})
    m.match(ACKV, 1, flags=[ISACKW], down={ISCREATE: 1.0})
    for i in range(NFILE):
        inc = {CINC[b]: float(c) for b, c in enumerate(pm(min(i + 1, NFILE - 1), IB))}
        m.match(ACKI + ACKV, i | (1 << IB), flags=[ISACKW], down=inc)
        m.match(ACKI, i, flags=[ISACKW], down={TK[i]: 1.0})
    return m


def layer2(net):
    d = Delta(net)
    nl, nl2 = code(NL), code(NL2)
    for j in range(NSLOT):
        # the name's j-th token, or EMPTY (all +1) at the name's start (a \n or \n\n token;
        # slots are written only there and at name tokens, which are never newlines)
        fields = [("pm", {IDC[b]: 1.0, FL["NL"]: 1.0 - nl[b], FL["NL2"]: 1.0 - nl2[b]}, NAME[j][b])
                  for b in range(IDB)]
        d.register(j, {}, fields)
        # on (~200) at a name's start or at slot j of a name token; far below 0 elsewhere
        d.k(j, 0, {NSTART: 800.0, GSTART: 800.0, FSTART: 800.0, BSTART: 800.0, QSTART: 800.0,
                   SLOTRAW[j]: 400.0, NAMING: 400.0, FL["NL"]: -400.0, FL["NL2"]: -400.0,
                   ONE: -600.0})
    fields = [("pm", {ACKI[b]: 1.0}, RI[b]) for b in range(IB)]
    fields += [("pm", {ACKV[b]: 1.0}, RV[b]) for b in range(VB)]
    fields += [("01", {ISACKW: 1.0}, MW), ("01", {ISACKR: 1.0}, MR),
               ("01", {LAG1["TCEND"]: 1.0, LAG1["TRESPEND"]: 1.0, ISNONE: 1.0}, None)]
    d.register(NSLOT, {ISACKW: 1.0, ISACKR: 1.0, ISNONE: 1.0, LAG1["TCEND"]: 1.0,
                       LAG1["TRESPEND"]: 1.0}, fields)
    d.register(NSLOT + 1, {ISCREATE: 1.0},
               [("bit01", {CINC[b]: 0.5, ONE: 0.5}, COUNT[b]) for b in range(IB)])
    # search: the fourth token back is a word token; past the "</" that ends a text; a search
    d.register(NSLOT + 2, {ONE: 1.0}, [("bit01", {LF[2][0]: 1.0}, L4WD)], tap=-2)
    d.register(NSLOT + 3, {LAG1["LTS"]: 1.0, TSTART: 1.0}, [("bit01", {LAG1["LTS"]: 1.0}, PASTLTS)])
    d.register(NSLOT + 4, {QSTART: 1.0, FL["TCSTART"]: 1.0}, [("bit01", {QSTART: 1.0}, SAWQUERY)])
    # the call is append_file (its function name token)
    d.register(NSLOT + 5, {FL["APPEND"]: 1.0, FL["TCSTART"]: 1.0}, [("bit01", {FL["APPEND"]: 1.0}, APP)])
    common(net)


def mlp2():
    m = MLP()
    for j in range(1, NMAX + 1):
        dk = {k: 2.0 * (bits(j, OB)[i] - bits(j - 1, OB)[i]) for i, k in enumerate(KOFF)}
        dq = {k: 2.0 * (bits(j + 1, OB)[i] - bits(j, OB)[i]) for i, k in enumerate(QOFF)}
        down = {**dk, **dq}
        m.relu({U[j - 1]: 4.0, ONE: -1.5}, down)
        m.relu({U[j - 1]: 4.0, ONE: -2.5}, {k: -w for k, w in down.items()})
    bias = {k: -1.0 for k in KOFF}
    bias.update({k: 2.0 * bits(1, OB)[i] - 1.0 for i, k in enumerate(QOFF)})
    m.relu({ONE: 1.0}, bias)
    for u in U:                                   # free U's dims for the late features
        m.unit({ONE: K}, {u: -1 / K}, {u: 1.0})
    for c in range(NFILE):
        m.match(COUNT, c, pm_dims=False, down={CO[c]: 1.0})
    m.relu({SAWFILE: 2.0, ONE: -1.0}, {LL: 1.0})
    m.relu({SAWPATH: 2.0, ONE: -1.0}, {LL: 1.0})
    # a word of m tokens ends here: this token is in a text (up to its closing "</") and is no
    # word token or starts a new word; the m before it are word tokens, all but the first
    # without a leading space, and the first starts one (a leading space, or no word token
    # before it); a first token other than the last must have a one-token form
    (wd1, sp1, _), (wd2, sp2, x12), (wd3, sp3, x13) = LF
    here = {INTEXT: 2.0, TSTART: -2.0, PASTLTS: -2.0, WD: -2.0, SP: 2.0, ONE: 2.0 - 3.0}
    for k, (on, off) in enumerate((
            ([wd1, sp1], []), ([wd1], [sp1, wd2]),
            ([wd1, wd2, sp2, x12], [sp1]), ([wd1, wd2, x12], [sp1, sp2, wd3]),
            ([wd1, wd2, wd3, sp3, x13], [sp1, sp2]), ([wd1, wd2, wd3, x13], [sp1, sp2, sp3, L4WD]))):
        lin = dict(here)
        for f in on:
            lin[f] = lin.get(f, 0.0) + 2.0
        for f in off:
            lin[f] = lin.get(f, 0.0) - 2.0
        lin[ONE] -= 2.0 * len(on)
        m.relu(lin, {WEND[k // 2]: 1.0})
    for j in range(S + 1):
        m.match(NAME[j], EMPTYC, down={QE[j]: 1.0})
    for i in range(NFILE):
        m.match(RI, i, down={RIOH[i]: 1.0})
    # a search checks file 0 at the newline that opens its response, then file i + 1 at the
    # newline that ends file i's path and at file i's skip
    m.all_of([FL["NL"], LAG1["TRESP"], INCALL, SAWQUERY], {CK[0]: 1.0, CHK: 1.0})
    for i in range(NFILE - 1):
        m.all_of([FL["NL"], LISTMODE, INCALL, SAWQUERY, LIOH[i]], {CK[i + 1]: 1.0, CHK: 1.0})
        m.all_of([ISSKIP, LIOH[i]], {CK[i + 1]: 1.0, CHK: 1.0})
    return m


def layer3(att):
    """Lookup: the name at GO against the names at every earlier ack."""
    q, k = att.q_proj.weight, att.k_proj.weight
    names = [d for s in NAME for d in s]
    n, W = len(names), 4.0
    q[ROT, ONE] = W
    k[ROT, ISACKW] = 2.0
    k[ROT, ISACKR] = 2.0
    k[ROT, GO] = 2.0 - 1.0 / W
    k[ROT, ONE] = -1.0
    for i, dd in enumerate(names):
        q[ROT + 1 + i, dd] = 1.0
        k[ROT + 1 + i, dd] = 1.0
    q[ROT + 1 + n, ONE] = -(W + n)
    k[ROT + 1 + n, ONE] = 1.0
    att.v_proj.weight[0, ISACKW] = 1.0
    att.v_proj.weight[0, ISACKR] = 1.0
    for b in range(IB):
        att.v_proj.weight[1 + b, ACKI[b]] = 1.0
    att.o_proj.weight[FOUND, 0] = 1.0
    for b in range(IB):
        att.o_proj.weight[FIDX[b], 1 + b] = 1.0
    q[HD:2 * HD, ONE] = 20.0
    att.q_norm.weight.fill_(50.0 - 1)
    att.k_norm.weight.fill_(50.0 - 1)


def mlp3():
    m = MLP()
    for i in range(NFILE):
        m.match(FIDX, i, flags=[GO, FOUND], down={TQ[i]: 1.0})
    for n in range(1, S + 1):                       # slot n - 1 used, slot n empty
        m.relu({QE[n]: 2.0, QE[n - 1]: -2.0, ONE: -1.0}, {QLEN[n - 1]: 1.0, QV: 1.0})
    return m


def layer4(net):
    d = Delta(net)
    d.register(0, {}, [("bit01", {ACKV[b]: 0.5, ONE: 0.5}, TBIT[b]) for b in range(VB)])
    for r in range(NFILE):
        d.q(0, r, {TQ[r]: 20.0})
        d.k(0, r, {TK[r]: 20.0})
    # name table: row I = the path (written at each write ack), read at the listed file's row
    for j in range(NSLOT):
        h = 1 + j
        d.register(h, {}, [("pm", {NAME[j][b]: 1.0}, NT[j][b]) for b in range(IDB)])
        for r in range(NFILE):
            d.q(h, r, {LIOH[r]: 20.0})
            d.k(h, r, {TK[r]: 20.0})
    # the base table: a write ack's version, unless the call appends to a file (an append that
    # creates one sets it); read at the file a search checks, and at GO (two heads, one table)
    for h, query, out in ((1 + NSLOT, CK, BCK), (2 + NSLOT, TQ, BGO)):
        d.register(h, {}, [("bit01", {ACKV[b]: 0.5, ONE: 0.5}, out[b]) for b in range(VB)])
        for r in range(NFILE):
            d.q(h, r, {query[r]: 20.0})
            d.k(h, r, {TK[r]: 20.0})
        d.beta(h, {ONE: 30.0, APP: -60.0, ISCREATE: 60.0})
    # the version table again, read at the file being read
    h = 3 + NSLOT
    d.register(h, {}, [("bit01", {ACKV[b]: 0.5, ONE: 0.5}, TRD[b]) for b in range(VB)])
    for r in range(NFILE):
        d.q(h, r, {RIOH[r]: 20.0})
        d.k(h, r, {TK[r]: 20.0})
    common(net)


def mlp4():
    """Both acks at GO: the write ack <w I V+1> (or <w COUNT 1> for a new name) and the read
    ack <r I V> (or <none>). Ack ids are (base, index, version) bit fields."""
    m = MLP()
    sub = lambda dims: (dims[:VB], dims[VB:VB + IB], dims[VB + IB:])
    wlo, wmid, whi = sub(WACK)
    rlo, rmid, rhi = sub(RACK)
    top = IDB - IB - VB
    for v in range(1 << VB):
        nxt = {d: float(c) for d, c in zip(wlo, pm(min(v + 1, (1 << VB) - 1), VB))}
        m.match(TBIT, v, flags=[GO, FOUND], pm_dims=False, down=nxt)
        # the written text's base: the new version, unless the call appends to the file
        m.match(TBIT, v, flags=[GO, FOUND], nots=[APP], pm_dims=False,
                down={h: nxt[w] for h, w in zip(HBS, wlo)})
        # a read starts at the file's base
        m.match(BGO, v, flags=[GO, FOUND], pm_dims=False,
                down={d: float(c) for d, c in zip(rlo, pm(v, VB))})
    new = {d: float(c) for d, c in zip(wlo, pm(1, VB))}
    m.relu({GO: 2.0, FOUND: -2.0, ONE: -1.0}, {**new, **{h: new[w] for h, w in zip(HBS, wlo)}})
    for b in range(VB):
        m.all_of([BGO[b], GO, FOUND, APP], {HBS[b]: 2.0})
    m.all_of([GO, FOUND, APP], {h: -1.0 for h in HBS})
    # the read file's version and the read's segment as clean bits (a register read can be off by
    # a little in a long prefill; the comparison weighs the high bits by up to 32)
    for b in range(VB):
        for x, out in (({TRD[b]: 2.0, ONE: -0.5}, TRC[b]), ({RV[b]: 1.0, ONE: 0.5}, RVC[b])):
            m.relu(x, {out: 1.0})
            m.relu({**x, ONE: x[ONE] - 1.0}, {out: -1.0})
    for b in range(IB):
        m.gated(FIDX[b], wmid[b], [GO, FOUND])
        m.gated(FIDX[b], rmid[b], [GO, FOUND])
        # a new file's index: 2 COUNT_b - 1, at GO when not found
        m.relu({COUNT[b]: 2.0, GO: 2.0, FOUND: -2.0, ONE: -1.0 - 2.0}, {wmid[b]: 1.0})
        m.relu({COUNT[b]: -2.0, GO: 2.0, FOUND: -2.0, ONE: 1.0 - 2.0}, {wmid[b]: -1.0})
    m.relu({GO: 2.0, ONE: -1.0}, {d: float(c) for d, c in zip(whi, pm(ACKW0 >> (IB + VB), top))})
    m.relu({GO: 2.0, FOUND: 2.0, ONE: -3.0},
           {d: float(c) for d, c in zip(rhi, pm(ACKR0 >> (IB + VB), top))})
    m.relu({GO: 2.0, FOUND: -2.0, ONE: -1.0}, codes(RACK, NONE))
    m.relu({GO: 2.0, PTGT: -2.0, ONE: -1.0}, {GOS: 1.0})
    m.relu({GO: 2.0, PTGT: 2.0, ONE: -3.0}, {GOT: 1.0})
    empty = code((1 << IDB) - 1)
    for j in range(NSLOT):
        m.relu({**{d: float(c) for d, c in zip(NT[j], empty)}, ONE: -(IDB - 1.0)}, {NTE[j]: 1.0})
    return m


def layer5(net):
    d = Delta(net)
    d.register(0, {GO: 1.0}, [("pm", {WACK[b]: 1.0}, HW[b]) for b in range(IDB)])
    d.register(1, {GO: 1.0}, [("pm", {RACK[b]: 1.0}, HR[b]) for b in range(IDB)])
    # a call's source (name or file path) and a link's target, from their GOs
    fields = [("pm", {FIDX[b]: 1.0}, HSRC[b]) for b in range(IB)] + [("bit01", {FOUND: 1.0}, HSF)]
    d.register(2, {GOS: 1.0}, fields)
    d.register(3, {GOT: 1.0}, [("pm", {FIDX[b]: 1.0}, HDST[b]) for b in range(IB)] +
               [("bit01", {FOUND: 1.0}, HDF)])
    d.register(4, {GO: 1.0}, [("pm", {HBS[b]: 1.0}, HB[b]) for b in range(VB)])
    common(net)


def mlp5():
    """The forced tokens, each picked by the current token, the previous one and the call's
    state. After </tool_call> the response goes in a user turn, as in Qwen's chat template
    (without <|im_end|>, so a chat loop that stops there runs on), then a new assistant turn:

        \n<|im_start|>user\n<tool_response>\n  ok | [ack]TEXT | [none]not found  \n</tool_response>
        \n<|im_start|>assistant\n<think>\n\n</think>\n\n

    The write ack goes where the text starts; a read's ack starts the copy."""
    m = MLP()
    f, l1 = FL, LAG1
    for b in range(IDB):
        m.gated(HW[b], OUTC[b], [f["NL"], l1["TRESP"], PENDING, WROTE])
        m.gated(HR[b], OUTC[b], [f["NL"], l1["TRESP"], PENDING], nots=[WROTE, SAWTGT])
    rules = [   # (flags, nots, token)
        ([f["TCEND"], INCALL], [], NL),
        ([f["NL"], l1["TCEND"], INCALL], [], IMSTART),
        ([f["IMSTART"], l1["NL"], INCALL], [], USER),
        ([f["USER"], INCALL], [], NL),
        ([f["NL"], l1["USER"], INCALL], [], TRESP),
        ([f["TRESP"], INCALL], [], NL),
        # list_files (a call without a name): the first marker, or "empty"
        ([f["NL"], l1["TRESP"], INCALL], [PENDING, CO[0], LL, SAWQUERY], MARK0),
        ([f["NL"], l1["TRESP"], INCALL, CO[0]], [PENDING, LL, SAWQUERY], EMPTYW),
        ([f["EMPTYW"], INCALL], [], NL),
        ([f["NL"], l1["EMPTYW"], INCALL], [], TRESPEND),
        # write: [w I V]saved, then the path (a marker, printed as in a listing), then the end
        ([ISACKW, INCALL, WROTE], [], SAVED),
        ([f["SAVED"], INCALL, WROTE], [], NL),
        ([f["NL"], LISTMODE, INCALL, PENDING, WROTE], [l1["SAVED"]], TRESPEND),
        ([ISNONE, INCALL], [], NOT),
        ([f["NOT"], l1["NONE"], INCALL], [], FOUND_),
        ([f["FOUND"], l1["NOT"], INCALL], [], NL),
        ([f["NL"], l1["FOUND"], INCALL], [], TRESPEND),
        # link_files / unlink_files: ok, or not found (either file missing)
        ([f["NL"], l1["TRESP"], INCALL, SAWTGT, HSF, HDF], [UNL], LINKED),
        ([f["NL"], l1["TRESP"], INCALL, SAWTGT, HSF, HDF, UNL], [], REMOVED),
        ([f["NL"], l1["TRESP"], INCALL, SAWTGT], [HSF], NONE),
        ([f["NL"], l1["TRESP"], INCALL, SAWTGT, HSF], [HDF], NONE),
        # link / unlink: linked|removed, the source's path, "to", the target's path, the end
        ([f["LINKED"], INCALL, SAWTGT], [], NL),
        ([f["REMOVED"], INCALL, SAWTGT], [], NL),
        ([f["NL"], LISTMODE, INCALL, SAWTGT], [TOSEEN], TO),
        ([f["TO"], INCALL, SAWTGT], [], NL),
        ([f["NL"], LISTMODE, INCALL, SAWTGT, TOSEEN], [l1["TO"]], TRESPEND),
        # list_links / list_backlinks: a missing file (the links themselves: the exit MLP)
        ([f["NL"], l1["TRESP"], INCALL, LL], [HSF], NONE),
        ([f["TRESPEND"]], [], NL),
        ([f["NL"], l1["TRESPEND"], TAIL], [], IMSTART),
        ([f["IMSTART"], l1["NL"], TAIL], [PENDING], ASSISTANT),
        ([f["ASSISTANT"], TAIL], [], NL),
        ([f["NL"], l1["ASSISTANT"], TAIL], [], THINK),
        ([f["THINK"], TAIL], [], NL2),
        ([f["NL2"], l1["THINK"], TAIL], [], THINKEND),
        ([f["THINKEND"], TAIL], [], NL2),
    ]
    for flags, nots, t in rules:
        m.all_of(flags, codes(OUTC, t), nots=nots)
    # a read's segment (its last ack's version) is before the file's latest version
    x = {ONE: 0.0}
    for b in range(VB):
        x[TRC[b]] = 2.0 ** b
        x[RVC[b]] = -(2.0 ** b)
    m.relu(x, {MORE: 1.0})
    m.relu({**x, ONE: -1.0}, {MORE: -1.0})
    # list: after a marker (slot n = 0) or a path's n-th token, its next token or a newline
    ls = [LS0] + SLOTRAW
    for n in range(NSLOT):
        for b in range(IDB):
            m.gated(NT[n][b], OUTC[b], [LISTMODE, ls[n]], nots=[NTE[n], f["NL"], f["TO"], ISSKIP])
        m.all_of([LISTMODE, ls[n], NTE[n]], codes(OUTC, NL), nots=[f["NL"], f["TO"], ISSKIP])
    m.all_of([LISTMODE, ls[NSLOT]], codes(OUTC, NL), nots=[f["NL"], f["TO"], ISSKIP])
    # echo markers: the written file; a link's source, then (after "to") its target
    mtop = {d: float(c) for d, c in zip(OUTC[IB:], pm(MARK0 >> IB, IDB - IB))}
    for flags, src in (([f["NL"], l1["SAVED"], INCALL, PENDING, WROTE], HW[VB:VB + IB]),
                       ([f["NL"], l1["LINKED"], INCALL, SAWTGT], HSRC),
                       ([f["NL"], l1["REMOVED"], INCALL, SAWTGT], HSRC),
                       ([f["NL"], l1["TO"], INCALL, SAWTGT], HDST)):
        for b in range(IB):
            m.gated(src[b], OUTC[b], flags)
        m.all_of(flags, mtop)
    # link bits: at a link's target GO (both found), the pair to write, its value (empty for
    # unlink); for a link listing, the rows to read (a file's links, or its backlinks)
    at = [GOT, HSF, FOUND]
    for i in range(NFILE):
        m.match(HSRC, i, flags=at, down={SOHW[i]: 1.0})
        m.match(FIDX, i, flags=at, down={DOHW[i]: 1.0})
        m.match(HSRC, i, flags=at, nots=[UNL], down={SV[i]: 1.0})
        m.match(FIDX, i, flags=at, nots=[UNL], down={DV[i]: 1.0})
        m.match(HSRC, i, flags=[SAWFILE], down={QF[i]: 1.0})
        m.match(HSRC, i, flags=[SAWPATH], down={QB[i]: 1.0})
    # after a path's newline: the next file's marker, or the end of the response
    for i in range(NFILE):
        more = {CO[c]: 1.0 for c in range(i + 2, NFILE)}
        done = {CO[c]: 1.0 for c in range(0, min(i + 2, NFILE))}
        # (a list_files call has no path: PENDING is clear; WROTE may be left from a write)
        base = {f["NL"]: 2.0, LISTMODE: 2.0, LIOH[i]: 2.0, LL: -2.0, PENDING: -2.0, SAWTGT: -2.0,
                SAWQUERY: -2.0, ONE: -7.0}
        if i + 1 < NFILE:
            m.relu({**base, **{c: 2.0 for c in more}}, codes(OUTC, MARK0 + i + 1))
        m.relu({**base, **{c: 2.0 for c in done}}, codes(OUTC, TRESPEND))
    return m


def layer_pair(net):
    """Link bits. Row (s, d) of head s // 4 (rows (s % 4) * 32 + d) is the pair: heads 0-7 hold
    the target's column (a file's links), heads 8-15 the source's (its backlinks). Written at
    a link's target GO (key on that one row, value one column or nothing for unlink); read with
    a multi-hot query over all of a listed file's rows, which sums them."""
    d = Delta(net)
    for g, (vals, reads) in enumerate(((DV, PF), (SV, PB))):
        for hh in range(NFILE // 4):
            h = 8 * g + hh
            for sl in range(4):
                s_ = 4 * hh + sl
                for t in range(NFILE):
                    r = sl * NFILE + t
                    d.k(h, r, {SOHW[s_]: 400.0, DOHW[t]: 400.0, ONE: -600.0})
                    d.q(h, r, {QF[s_]: 20.0, QB[t]: 20.0})
            for c in range(NFILE):
                d.v(h, c, {vals[c]: 20.0})
                d.out(h, c, reads[c], 1.0 / 32)
            d.beta(h, {ONE: 30.0})
            d.decay(h, {ONE: -100.0})
            d.z(h, {ONE: 32.0})
    common(net)


def mlp_pair():
    """Clean link bits of the listed file: a read is 0 or at least 2 (the head norm over at most
    32 set columns), so relu(x) - relu(x - 1) is 0 or 1, for the call's direction."""
    m = MLP()
    for j in range(NFILE):
        for x, flag in ((PF[j], SAWFILE), (PB[j], SAWPATH)):
            m.unit({x: K}, {flag: 1 / K}, {BIT[j]: 1.0})
            m.unit({x: K, ONE: -K}, {flag: 1 / K}, {BIT[j]: -1.0})
    search(m)
    return m


def search(m):
    """The search features (all 0 elsewhere). At a word's end, its row: the write's index and
    version (held since the path's GO) and the word's form: the last token's whole form for a
    one-token word, else each token's form slot by slot. At a check, the query: the checked
    file's index and version, and the query's tokens right-aligned like a form."""
    ends = WEND
    for b in range(IB):
        m.gated_any(HW[VB + b], XI[b], ends)
    for b in range(VB):
        m.gated_any(HB[b], XV[b], ends)
    empty = {d: float(c) for d, c in zip(range(IDB), code(EMPTYC))}
    for b in range(IDB):
        m.gated_any(L1W[0][b], XW[0][b], ends)
        m.gated_any(L1W[1][b], XW[1][b], [WEND[0]])
        m.gated_any(L2W[b], XW[1][b], [WEND[1], WEND[2]])
        m.gated_any(L1W[2][b], XW[2][b], [WEND[0]])
        m.gated_any(L3W[b], XW[2][b], [WEND[2]])
    m.relu({WEND[1]: 2.0, ONE: -1.0}, {XW[2][b]: float(c) for b, c in empty.items()})
    m.relu({WEND[0]: 2.0, WEND[1]: 2.0, WEND[2]: 2.0, ONE: -1.0}, {XT: 1.0})
    for k in range(NFILE):
        m.all_of([CK[k], QV], {XI[b]: float(c) for b, c in enumerate(pm(k, IB))})
    for b in range(VB):
        m.all_of([BCK[b], CHK, QV], {XV[b]: 2.0})
        m.all_of([CHK, QV], {XV[b]: -1.0})
    for n in range(1, S + 1):                      # a query of n tokens
        for r in range(S):
            if r < n:
                for b in range(IDB):
                    m.gated(NAME[n - 1 - r][b], XW[r][b], [CHK, QLEN[n - 1]])
            else:
                m.all_of([CHK, QLEN[n - 1]], {XW[r][b]: float(c) for b, c in empty.items()})


def layer6(att):
    """Copy: a read's (index, version, offset) against the stored tokens."""
    q, k = att.q_proj.weight, att.k_proj.weight
    q[ROT, ONE] = 1.0
    k[ROT, INTEXT] = 2.0                 # a write's text (not the newline that opens it)
    k[ROT, TSTART] = -2.0
    k[ROT, ONE] = -1.0
    # a read asks by its ack's index and version; the text is keyed by the write's, held
    # since the path's GO (its write ack: version bits low, index bits above)
    for i, (r, h) in enumerate(zip(RI + RV, HW[VB:VB + IB] + HW[:VB])):
        q[ROT + 1 + i, r] = 1.0
        k[ROT + 1 + i, h] = 1.0
    base = ROT + 1 + IB + VB
    for i in range(OB):
        q[base + i, QOFF[i]] = 1.0
        k[base + i, KOFF[i]] = 1.0
    for i in range(IDB):
        att.v_proj.weight[i, IDC[i]] = 1.0
        att.o_proj.weight[OUTC[i], i] = 1.0
    q[HD:2 * HD, MR] = 30.0
    q[HD:2 * HD, FL["TRESPEND"]] = -30.0
    q[HD:2 * HD, ONE] = -15.0
    att.q_norm.weight.fill_(GAIN7 - 1)
    att.k_norm.weight.fill_(GAIN7 - 1)
    search_head(att)


GAIN7, WS = 16.0, 4.0


def search_head(att):
    """Search: at a check, the query (index, version, form) against the words' rows, on the
    first query head of the second key/value head. The check's own row is the fallback: its
    features are the query's, so it scores a little below a word that matches and above any
    near miss. The value is 1 on a word's row: the output is ~1 if the file has the word."""
    n_kv = att.k_proj.out_features // HD
    n_q = att.q_proj.out_features // (2 * HD)
    qh = n_q // n_kv
    qb, kb = qh * 2 * HD, HD
    q, k = att.q_proj.weight, att.k_proj.weight
    feats = XI + XV + [d for w in XW for d in w]
    q[qb + ROT, ONE] = WS
    k[kb + ROT, XT] = 2.0
    k[kb + ROT, CHK] = 2.0 - 1.0 / WS
    k[kb + ROT, ONE] = -1.0
    for i, d in enumerate(feats):
        q[qb + ROT + 1 + i, d] = 1.0
        k[kb + ROT + 1 + i, d] = 1.0
    q[qb + HD:qb + 2 * HD, ONE] = 20.0
    att.v_proj.weight[kb, XT] = 1.0
    att.o_proj.weight[SFOUND, qh * HD] = 1.0


def mlp6():
    """Exit: the output code into the id dims, EPS-sized. A copied "</" (the text's end, as
    Qwen closed its parameter) comes out as </tool_response>.
    Every other FS dim, ballast included, is cleared."""
    m = MLP()
    for o, i in zip(OUTC, IDC):
        m.unit({ONE: K}, {o: EPS / K, i: -1 / K}, {i: 1.0})
    swap = {d: EPS * (a - b) for d, a, b in zip(IDC, code(TRESPEND), code(LTS))}
    s = {o: 4.0 * float(c) for o, c in zip(OUTC, code(LTS))}       # 72 on a match, <= 64 else
    m.relu({**s, MORE: -2.0, ONE: -70.0}, swap)
    m.relu({**s, MORE: -2.0, ONE: -71.0}, {d: -w for d, w in swap.items()})
    # a segment that is not the last: its "</" becomes the next segment's read ack (the read's
    # index, its version + 1); g = 1 there, <= -1 everywhere else
    g = {**s, MORE: 2.0, ONE: -73.0}
    top = IB + VB
    m.relu(g, {**{d: -EPS * c for d, c in zip(IDC, code(LTS))},
               **{d: EPS * c - EPS * c0 for d, c, c0 in
                  zip(IDC[top:], pm(ACKR0 >> top, IDB - top), code(LTS)[top:])}})
    for b in range(IB):
        m.relu({**{k: 2.0 * w for k, w in g.items()}, RI[b]: 1.0, ONE: 2.0 * g[ONE] - 2.0},
               {IDC[VB + b]: EPS})
        m.relu({**{k: 2.0 * w for k, w in g.items()}, RI[b]: -1.0, ONE: 2.0 * g[ONE] - 2.0},
               {IDC[VB + b]: -EPS})
    for v in range(1 << VB):
        lin = {r: (1.0 if bit else -1.0) for r, bit in zip(RV, bits(v, VB))}
        lin.update({k: 2.0 * w for k, w in g.items() if k != ONE})
        lin[ONE] = 2.0 * g[ONE] - 2.0 - (VB - 1)
        m.relu(lin, {d: EPS * c for d, c in zip(IDC[:VB], pm(min(v + 1, (1 << VB) - 1), VB))})
    # link listings: the first set bit, then after file i's path the next set bit after i, or
    # the end; with no set bit, "empty"
    out = lambda t: {d: EPS * c for d, c in zip(IDC, code(t))}
    first = [FL["NL"], LAG1["TRESP"], INCALL, LL, HSF]
    nxt = [FL["NL"], LISTMODE, LL]
    fl = lambda fs: {f: 2.0 for f in fs}
    for j in range(NFILE):
        m.relu({**fl(first), BIT[j]: 2.0, **{BIT[k]: -2.0 for k in range(j)},
                ONE: -2.0 * len(first) - 1.0}, out(MARK0 + j))
    m.relu({**fl(first), **{BIT[k]: -2.0 for k in range(NFILE)}, ONE: -2.0 * len(first) + 1.0},
           out(EMPTYW))
    for i in range(NFILE):
        for j in range(i + 1, NFILE):
            m.relu({**fl(nxt), LIOH[i]: 2.0, BIT[j]: 2.0, **{BIT[k]: -2.0 for k in range(i + 1, j)},
                    ONE: -2.0 * len(nxt) - 3.0}, out(MARK0 + j))
        m.relu({**fl(nxt), LIOH[i]: 2.0, **{BIT[k]: -2.0 for k in range(i + 1, NFILE)},
                ONE: -2.0 * len(nxt) - 1.0}, out(TRESPEND))
    # search results: at each check, the checked file's marker if it has the word, else its
    # skip; past the last file the end, or "empty" if no file had it
    def rule(on, off, t):
        lin = {d: 2.0 for d in on}
        for d in off:
            lin[d] = lin.get(d, 0.0) - 2.0
        lin[ONE] = lin.get(ONE, 0.0) - (2.0 * len(on) - 1.0)
        m.relu(lin, out(t))

    def at(cs, trig, on, off, t):           # one of the count cases cs, trig and on, not off
        flags = set(trig) | set(on)
        lin = {d: 2.0 for d in (*cs, *flags)}
        for d in off:
            lin[d] = lin.get(d, 0.0) - 2.0
        lin[ONE] = -(2.0 * (1 + len(flags)) - 1.0)
        m.relu(lin, out(t))

    first = [FL["NL"], LAG1["TRESP"], INCALL, SAWQUERY]
    rule(first, [QV], EMPTYW)
    rule(first + [QV, CO[0]], [], EMPTYW)
    rule(first + [QV, SFOUND], [CO[0]], MARK0)
    rule(first + [QV], [CO[0], SFOUND], SKIP0)
    for i in range(NFILE):
        more = [CO[c] for c in range(i + 2, NFILE)]
        done = [CO[c] for c in range(min(i + 2, NFILE))]
        for trig in ([FL["NL"], LISTMODE, INCALL, SAWQUERY, LIOH[i]], [ISSKIP, LIOH[i]]):
            if more:
                at(more, trig, [SFOUND], [], MARK0 + i + 1)
                at(more, trig, [], [SFOUND], SKIP0 + i + 1)
            at(done, trig, [LISTMODE], [], TRESPEND)
            if LISTMODE not in trig:
                at(done, trig, [], [LISTMODE], EMPTYW)
    for dd in range(B_[-1] + 1, D.n):
        if dd not in IDC:
            m.unit({ONE: K}, {dd: -1 / K}, {dd: 1.0})
    m.unit({ONE: K}, {ONE: -BAL / K}, {b: 1.0 for b in B_})
    return m


def embeddings(E, words=None):
    """words: (slots, flags) from qfs.words.table, the word forms search_files matches; without
    them no token is a word."""
    E[:, B_] = BAL
    E[:, ONE] = 1
    E[:, ONE2] = 1
    E[:, IDC] = torch.tensor([code(i) for i in range(VOCAB)], dtype=E.dtype)
    for t, name in FLAG_TOKENS.items():
        E[t, FL[name]] = 1
    E[NONE, ISNONE] = 1
    for i in range(NFILE):
        E[MARK0 + i, ISMARK] = 1
        E[MARK0 + i, ACKI] = torch.tensor(pm(i, IB), dtype=E.dtype)
    for i in range(NFILE):
        for v in range(1 << VB):
            for t, flag in ((ackw(i, v), ISACKW), (ackr(i, v), ISACKR)):
                E[t, flag] = 1
                E[t, ACKI] = torch.tensor(pm(i, IB), dtype=E.dtype)
                E[t, ACKV] = torch.tensor(pm(v, VB), dtype=E.dtype)
    for i in range(NFILE):
        E[SKIP0 + i, ISSKIP] = 1
        E[SKIP0 + i, ACKI] = torch.tensor(pm(i, IB), dtype=E.dtype)
    if words is not None:
        slots, flags = words
        bits = torch.arange(IDB)
        for r in range(S):
            ids = torch.as_tensor(slots[:VOCAB, r])
            E[:, WFORM[r]] = (((ids[:, None] >> bits) & 1) * 2 - 1).to(E.dtype)
        for c, d in enumerate((WD, SP, X1)):
            E[:, d] = torch.as_tensor(flags[:VOCAB, c]).to(E.dtype)


def norms(layers):
    for layer in layers:
        for n in (layer.input_layernorm, layer.post_attention_layernorm):
            n.weight.fill_(-1)
            n.weight[OFF:D.n] = SCALE - 1
            n.weight[B_] = -1


def code_gain(fn, gain):
    fn.weight[OFF:] = -1
    fn.weight[IDC] = gain - 1


def fill(layers, xs):
    """Write the FS layers into `layers` (a model's first len(LAYERS) decoder layers)."""
    norms(layers)
    layer_p(layers[0].linear_attn)
    layer1(layers[1].linear_attn)
    layer2(layers[2].linear_attn)
    layer3(layers[3].self_attn)
    layer4(layers[4].linear_attn)
    layer5(layers[5].linear_attn)
    layer_pair(layers[6].linear_attn)
    layer6(layers[7].self_attn)
    mlps = [mlp_p(), mlp1(xs) if xs is not None else MLP(), mlp2(), mlp3(), mlp4(), mlp5(),
            mlp_pair(), mlp6()]
    for layer, m in zip(layers, mlps):
        assert len(m.units) <= layer.mlp.gate_proj.out_features, len(m.units)
        m.write(layer.mlp)


def config():
    return Qwen3_5TextConfig(
        vocab_size=VOCAB, hidden_size=H, intermediate_size=2 * NMAX + 1600,
        num_hidden_layers=len(LAYERS), layer_types=LAYERS, num_attention_heads=2,
        num_key_value_heads=2, head_dim=HD, attn_output_gate=True,
        linear_num_key_heads=16, linear_num_value_heads=16, linear_key_head_dim=DK, linear_value_head_dim=DK,
        linear_conv_kernel_dim=4, tie_word_embeddings=True, rms_norm_eps=1e-6, hidden_act="silu")


def build(dev="mps", xs=None, words=None):
    """The standalone FS model (no Qwen); without xs the data thermometer is left empty, without
    words (qfs.words.table) no token is a word."""
    model = Qwen3_5ForCausalLM(config())
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
        embeddings(model.model.embed_tokens.weight, words)
        fill(model.model.layers, xs)
        code_gain(model.model.norm, 1.0)
    model.tie_weights()
    return model.to(dev, torch.bfloat16).eval()


@torch.no_grad()
def calibrate(dev="mps"):
    """The data counter as layer 1's MLP sees it, n = 0..NMAX tokens after an ack (the op
    counter is the same head design, so it reads the same values)."""
    model = build(dev)
    seen = {}

    def grab(mod, args, out):
        seen["x"] = out[0, :, CNT].float().cpu()
    h = model.model.layers[DATAL].post_attention_layernorm.register_forward_hook(grab)
    ids = torch.tensor([[ackr(0, 1)] + [1000 + i for i in range(NMAX)]], device=dev)
    try:
        model(input_ids=ids)
    finally:
        h.remove()
    xs = seen["x"].tolist()
    for a, b in zip(xs, xs[1:]):
        assert 0 < b < a, "counter values must fall strictly"
    return xs
