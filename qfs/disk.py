"""The filesystem on disk: only paths, token ids and links are stored; mounting and paging
rebuild the circuit's rows and tables with the model's own projections.

Disk layout (DIR):
  index.bin   one record per file: its path (NSLOT token ids, empty slots 0xFFFFFFFF), its index
              I and latest version V
  files/I.tok the file's text as 3-byte token ids: its segments, from its last write to V
  files/I.seg each segment's length in tokens (uint16); the file's base is V minus the
              segments plus one (no .seg: one segment, an image from before appends)
  links.bin   the links, as (source, target) byte pairs (a set: link and unlink applied)

Only the current segments are kept, so overwrites and deletes free their old text. The image is cut
from a session's token log by `compact`, using the write acks the model emitted (they carry I
and V) and the paths and texts Qwen wrote in its tool calls.

Mounting fills what a fresh session needs from the index: the lookup layer's rows (path -> I),
the version and base tables (I -> V, base), the name table (I -> path, for list_files), the
file count, the seek layer's link rows (one per link); and
from the texts, the search rows: one per distinct word of each file, keyed as at the word's end
(I, V, its form; the forms come from the model's own embeddings). Everything else starts empty:
the idle state. Paging a file in, when the model emits its read
ack, builds its copy-layer rows from its token ids: the features the copy head saw when the
text was written (data flag, I, V, offset, token id) through the layer's k_proj, k_norm, v_proj.
"""
import time
from pathlib import Path

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from qfs import circuit as F
from qfs import words

LOOKUP, SEEK, COPY, TABLE, COUNTL = F.LOOKUP, F.SEEK, F.COPY, F.TABLE, F.COUNTL
NO_TOKEN = 0xFFFFFFFF
REC = np.dtype([("name", "<u4", F.NSLOT), ("i", "u1"), ("v", "u1")])


def writes(toks):
    """Every write or append in a session's token log, in order: (I, V, path, text start, text
    end, append), the text being toks[start:end]."""
    out = []
    for p, t in enumerate(toks):
        if not (F.ACKW0 <= t < F.ACKR0):
            continue
        i, v = (t - F.ACKW0) >> F.VB, (t - F.ACKW0) & ((1 << F.VB) - 1)
        # the ack is in the response; the call before it has the text, and before that the path
        def after(eq, before):
            return max(k for k in range(before) if toks[k] == eq and toks[k + 1] == F.GT
                       and toks[k + 2] in (F.NL, F.NL2)) + 3
        ts = after(F.EQTEXT, p)
        te = next(k for k in range(ts, p) if toks[k] == F.NL and toks[k + 1] == F.LTS)
        ns = after(F.EQNAME, ts)
        call = toks[max(k for k in range(ns) if toks[k] == F.TCSTART):ns]
        out.append((i, v, tuple(toks[ns:toks.index(F.NL, ns)]), ts, te, F.APPEND in call))
    return out


def compact(toks, files=None):
    """{I: (path, V, segments)}: `files` (a mounted image's), then a session's token log: a
    write starts a file's segments over, an append adds one. A version the image already has
    is skipped, so the log can be applied again to an image saved from it."""
    out = dict(files or {})
    for i, v, name, ts, te, append in writes(toks):
        if i in out and v <= out[i][1]:
            continue
        segs = out[i][2] if append and i in out else []
        out[i] = (name, v, [*segs, toks[ts:te]])
    return out


def base(v, segs):
    """A file's base: the version of its first segment."""
    return v - len(segs) + 1


def compact_links(toks, files, links=()):
    """{(S, D)}: `links`, then every link_files / unlink_files call the model answered "linked" /
    "removed"."""
    index = {name: i for i, (name, _, _) in files.items()}
    links, p = set(links), 0
    while F.TCSTART in toks[p:]:
        a = toks.index(F.TCSTART, p)
        b = toks.index(F.TCEND, a)
        call = toks[a:b]
        p = b + 1
        if F.EQTARGET not in call:
            continue
        def path(eq):
            k = call.index(eq) + 3
            return tuple(call[k:call.index(F.NL, k)])
        end = toks.index(F.TRESPEND, b)
        if not {F.LINKED, F.REMOVED, F.LINKEDN, F.REMOVEDN} & set(toks[b:end]):
            continue
        pair = (index[path(F.EQNAME)], index[path(F.EQTARGET)])
        if any(t in call for t in F.UNLINK_TOKENS):
            links.discard(pair)
        else:
            links.add(pair)
    return links


def save(files, path, links=()):
    path = Path(path)
    (path / "files").mkdir(parents=True, exist_ok=True)
    recs = np.zeros(len(files), REC)
    for r, (i, (name, v, segs)) in enumerate(sorted(files.items())):
        recs[r]["name"] = list(name) + [NO_TOKEN] * (F.NSLOT - len(name))
        recs[r]["i"], recs[r]["v"] = i, v
        ids = np.array([t for seg in segs for t in seg], dtype="<u4")
        np.stack([ids & 255, ids >> 8 & 255, ids >> 16 & 255], 1).astype(np.uint8).tofile(
            path / "files" / f"{i}.tok")
        np.array([len(seg) for seg in segs], dtype="<u2").tofile(path / "files" / f"{i}.seg")
    recs.tofile(path / "index.bin")
    np.array(sorted(links), dtype=np.uint8).reshape(-1, 2).tofile(path / "links.bin")


def load(path):
    """A stored image as ({I: (path, V, segments)}, links), the form `compact` and `save` use."""
    path = Path(path)
    files = {}
    for r in np.fromfile(path / "index.bin", dtype=REC):
        i = int(r["i"])
        name = tuple(int(t) for t in r["name"] if int(t) != NO_TOKEN)
        files[i] = (name, int(r["v"]), load_segments(path, i)[0])
    links = np.fromfile(path / "links.bin", dtype=np.uint8).reshape(-1, 2)
    return files, {(int(a), int(b)) for a, b in links.tolist()}


def load_tokens(f):
    b = np.fromfile(f, dtype=np.uint8).reshape(-1, 3).astype(np.int64)
    return (b[:, 0] | b[:, 1] << 8 | b[:, 2] << 16).tolist(), b.size


def load_segments(path, i):
    """File i's segments (token lists) and its bytes on disk."""
    toks, nbytes = load_tokens(Path(path) / "files" / f"{i}.tok")
    seg = Path(path) / "files" / f"{i}.seg"
    lens = np.fromfile(seg, dtype="<u2").tolist() if seg.exists() else [len(toks)]
    out, k = [], 0
    for n in lens:
        out.append(toks[k:k + n])
        k += n
    return out, nbytes


def pmv(x, n):
    return torch.tensor(F.pm(x, n), dtype=torch.float32)


def slot_codes(rec):
    return [F.code((1 << F.IDB) - 1 if int(t) == NO_TOKEN else int(t)) for t in rec["name"]]


@torch.no_grad()
def kv(att, x):
    """The rows a full-attention layer stores for features x (n, H): rotary dims are zero."""
    x = x.to(att.k_proj.weight.device, torch.bfloat16)[None]
    n_kv = att.k_proj.out_features // F.HD
    k = att.k_norm(att.k_proj(x).view(1, x.shape[1], n_kv, F.HD)).transpose(1, 2)
    v = att.v_proj(x).view(1, x.shape[1], n_kv, F.HD).transpose(1, 2)
    return k, v


def copy_rows(model, i, v, body):
    """Copy-layer rows of version v of file i: its tokens, then the newline and "</" that close
    the parameter (a read copies up to that "</", which comes out as </tool_response>)."""
    toks = body + [F.NL, F.LTS]
    x = torch.zeros(len(toks), F.H)
    x[:, F.ONE] = 1
    x[:, F.INTEXT] = 1
    x[:, F.HW[F.VB:F.VB + F.IB]] = pmv(i, F.IB)      # the write's ack, held since the path
    x[:, F.HW[:F.VB]] = pmv(v, F.VB)
    for n, t in enumerate(toks, 1):
        x[n - 1, F.KOFF] = pmv(n, F.OB)
        x[n - 1, F.IDC] = pmv(t, F.IDB)
    return kv(model.model.layers[COPY].self_attn, x)


def word_table(model):
    """The word forms and flags the model's embeddings carry (as qfs.words.table has them)."""
    if not hasattr(model, "_qfs_words"):
        E = model.model.embed_tokens.weight
        bits = 1 << torch.arange(F.IDB, device=E.device)
        slots = torch.stack([((E[:, F.WFORM[r]] > 0).long() * bits).sum(1) for r in range(F.S)], 1)
        flags = E[:, [F.WD, F.SP, F.X1]] > 0.5
        model._qfs_words = (slots.cpu().numpy(), flags.cpu().numpy())
    return model._qfs_words


def seek_rows(model, files, links):
    """Seek-layer rows of a stored image: a search row per distinct word of each file, with the
    features its end had when the text was written, and a link row (+1) per link."""
    slots, flags = word_table(model)
    rows = [(i, base(v, segs), form) for i, (_, v, segs) in sorted(files.items())
            for form in sorted({f for seg in segs for f in words.words(seg, slots, flags).values()})
            if words.NEVER not in form]
    links = sorted(links)
    x = torch.zeros(len(rows) + len(links), F.H)
    x[:, F.ONE] = 1
    for n, (i, v, form) in enumerate(rows):
        x[n, F.XT] = 1
        x[n, F.XI] = pmv(i, F.IB)
        x[n, F.XV] = pmv(v, F.VB)
        for r in range(F.S):
            x[n, F.XW[r]] = pmv(form[r], F.IDB)
    for n, (a, b) in enumerate(links, len(rows)):
        x[n, F.LT], x[n, F.LV] = 1, 1
        x[n, F.LS], x[n, F.LD] = pmv(a, F.IB), pmv(b, F.IB)
    return kv(model.model.layers[SEEK].self_attn, x)


def lookup_rows(model, recs):
    """Lookup-layer rows, one per file: its path -> its index (as at a write's ack)."""
    x = torch.zeros(len(recs), F.H)
    x[:, F.ONE] = 1
    x[:, F.ISACKW] = 1
    for r, rec in enumerate(recs):
        for s, c in enumerate(slot_codes(rec)):
            x[r, F.NAME[s]] = torch.tensor(c, dtype=torch.float32)
        x[r, F.ACKI] = pmv(int(rec["i"]), F.IB)
    return kv(model.model.layers[LOOKUP].self_attn, x)


def dual(bits01):
    """A register row for 0/1 bits as the circuit's registers store them: dual rails, then the
    constant rails that pad the set rails to 2, 8, 32 or 128."""
    out = []
    for bit in bits01:
        out += [20.0 * bit, 20.0 * (1 - bit)]
    n_set = next(n for n in (2, 8, 32, 128) if n >= len(bits01))
    return out + [20.0] * (n_set - len(bits01))


class FSCache(DynamicCache):
    """The cache for a model with FS layers. The FS attention layers hold rows that are not
    tokens (mounted paths, paged-in text), so their lengths are not the sequence's: positions and
    mask sizes come from Qwen's first full-attention layer instead (in the circuit alone, which
    has no Qwen layers, from the first layer as usual), and `attach` masks the FS layers."""

    def __init__(self, config):
        super().__init__(config=config)
        types = config.layer_types
        self.pos = next((i for i in range(len(F.LAYERS), len(types))
                         if types[i] == "full_attention"), None)

    def _at(self, layer_idx):
        return (self.pos if self.pos is not None and layer_idx in (0, LOOKUP, SEEK, COPY)
                else layer_idx)

    def get_seq_length(self, layer_idx=0):
        return super().get_seq_length(self._at(layer_idx))

    def get_mask_sizes(self, query_length, layer_idx):
        return super().get_mask_sizes(query_length, self._at(layer_idx))


def attach(model):
    """Masks for the FS attention layers: every stored row is visible to the new tokens, which
    see each other causally. (A mask from token positions does not fit rows that are not tokens.)"""
    if getattr(model, "_qfs_masks", False):
        return

    def hook(li):
        def pre(mod, args, kw):
            h = kw["hidden_states"] if "hidden_states" in kw else args[0]
            q = h.shape[1]
            if q == 1:
                kw["attention_mask"] = None
                return args, kw
            c = kw.get("past_key_values")
            n = c.layers[li].get_seq_length() if c is not None else 0
            m = torch.ones(q, n + q, dtype=torch.bool, device=h.device)
            m[:, n:] = torch.ones(q, q, dtype=torch.bool, device=h.device).tril()
            kw["attention_mask"] = m[None, None]
            return args, kw
        return pre

    for li in (LOOKUP, SEEK, COPY):
        model.model.layers[li].self_attn.register_forward_pre_hook(hook(li), with_kwargs=True)
    model._qfs_masks = True


def mount(model, path, dev):
    """A fresh cache holding a stored filesystem: the tables and lookup rows from its index, an
    empty copy layer (files page in on their read acks), everything else idle. Returns the cache
    and {I: V} of the stored files."""
    attach(model)
    path = Path(path)
    cache = FSCache(model.config)
    recs = np.fromfile(path / "index.bin", dtype=REC)
    net = model.model.layers[0].linear_attn
    conv = torch.zeros(1, net.conv_dim, net.conv_kernel_size, dtype=torch.bfloat16, device=dev)
    state = {li: torch.zeros(1, net.num_v_heads, F.DK, F.DK, dtype=torch.float32, device=dev)
             for li, kind in enumerate(F.LAYERS) if kind == "linear_attention"}
    files, links = load(path)
    for r in recs:
        i = int(r["i"])
        row = dual(F.bits(int(r["v"]), F.VB))                     # version table (two heads)
        state[TABLE][0, 0, i, :len(row)] = torch.tensor(row)
        state[TABLE][0, 3 + F.NSLOT, i, :len(row)] = torch.tensor(row)
        b = dual(F.bits(base(int(r["v"]), files[i][2]), F.VB))   # base table (two heads)
        state[TABLE][0, 1 + F.NSLOT, i, :len(b)] = torch.tensor(b)
        state[TABLE][0, 2 + F.NSLOT, i, :len(b)] = torch.tensor(b)
        for j, c in enumerate(slot_codes(r)):                     # name table: a head per slot
            vals = dual([(b + 1) // 2 for b in c])
            state[TABLE][0, 1 + j, i, :len(vals)] = torch.tensor(vals)
    if recs.size:                                                  # file count
        row = dual(F.bits(len(recs), F.CB))
        state[COUNTL][0, F.NSLOT + 1, 0, :len(row)] = torch.tensor(row)
    for li, rec in state.items():
        layer = cache.layers[li]
        layer.lazy_initialization(conv_states=conv, recurrent_states=rec, state_idx=0)
        layer.conv_states[0].copy_(conv)
        layer.recurrent_states[0].copy_(rec)
        layer.has_previous_state[0] = True
    put(cache, LOOKUP, *lookup_rows(model, recs))
    put(cache, SEEK, *seek_rows(model, files, links))
    n_kv = model.model.layers[COPY].self_attn.k_proj.out_features // F.HD
    empty = torch.zeros(1, n_kv, 0, F.HD, dtype=torch.bfloat16, device=dev)
    put(cache, COPY, empty, empty)                    # the copy rows page in on reads
    return cache, {i: (base(v, segs), v) for i, (_, v, segs) in files.items()}


def put(cache, layer, k, v):
    c = cache.layers[layer]
    c.keys, c.values = k, v
    c.dtype, c.device = k.dtype, k.device
    c.is_initialized = True


class Pager:
    """Pages a stored segment's text into the copy layer when the model emits its read ack (a
    segment written in this session is already live). A read of several segments pages each in
    at its own ack."""

    def __init__(self, model, path, version):
        """version: {I: (base, V)} of the stored files."""
        self.m, self.path, self.version = model, Path(path), version
        self.loaded, self.load_s, self.load_bytes = set(), 0.0, 0

    def __call__(self, cache, tok):
        """Page in the segment `tok` acks, if it is stored and not yet loaded; returns (I, bytes,
        rows) when it paged one."""
        if not (F.ACKR0 <= tok < F.NONE):
            return None
        i, v = (tok - F.ACKR0) >> F.VB, (tok - F.ACKR0) & ((1 << F.VB) - 1)
        if (i, v) in self.loaded or i not in self.version:
            return None
        b, top = self.version[i]
        if not b <= v <= top:
            return None
        t0 = time.perf_counter()
        body = load_segments(self.path, i)[0][v - b]
        k, v_ = copy_rows(self.m, i, v, body)
        c = cache.layers[COPY]
        put(cache, COPY, torch.cat([c.keys, k], 2), torch.cat([c.values, v_], 2))
        self.load_s += time.perf_counter() - t0
        self.load_bytes += 3 * len(body)
        self.loaded.add((i, v))
        return i, 3 * len(body), k.shape[2]


class Session:
    """A fresh session on a stored filesystem, one token per step."""

    def __init__(self, model, path, dev, paging=True):
        self.m, self.dev = model, dev
        self.cache, version = mount(model, path, dev)
        self.pager = Pager(model, path, version) if paging else None

    @property
    def loaded(self):
        return self.pager.loaded if self.pager else set()

    @property
    def load_s(self):
        return self.pager.load_s if self.pager else 0.0

    @property
    def load_bytes(self):
        return self.pager.load_bytes if self.pager else 0

    @torch.no_grad()
    def step(self, tok):
        """Feed one token; returns the model's next token."""
        if self.pager:
            self.pager(self.cache, tok)
        out = self.m(input_ids=torch.tensor([[tok]], device=self.dev), past_key_values=self.cache,
                     use_cache=True)
        self.cache = out.past_key_values
        return int(out.logits[0, -1].argmax())
