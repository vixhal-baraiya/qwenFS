"""The filesystem on disk: only paths, token ids and links are stored; mounting and paging
rebuild the circuit's rows and tables with the model's own projections.

Disk layout (DIR):
  index.bin   one record per file: its path (NSLOT token ids, empty slots 0xFFFFFFFF), its index
              I and latest version V
  files/I.tok the latest version's text as 3-byte token ids (an empty file is a delete)
  links.bin   the links, as (source, target) byte pairs (a set: link and unlink applied)

Only the latest version is kept, so overwrites and deletes free their old text. The image is cut
from a session's token log by `compact`, using the write acks the model emitted (they carry I
and V) and the paths and texts Qwen wrote in its tool calls.

Mounting fills what a fresh session needs from the index: the lookup layer's rows (path -> I),
the version table (I -> V), the name table (I -> path, for list_files) and the file count.
Everything else starts empty: the idle state. Paging a file in, when the model emits its read
ack, builds its copy-layer rows from its token ids: the features the copy head saw when the
text was written (data flag, I, V, offset, token id) through the layer's k_proj, k_norm, v_proj.
"""
import time
from pathlib import Path

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from qfs import circuit as F

LOOKUP, PAIR, COPY, TABLE, COUNTL = F.LOOKUP, F.PAIR, F.COPY, F.TABLE, F.COUNTL
NO_TOKEN = 0xFFFFFFFF
REC = np.dtype([("name", "<u4", F.NSLOT), ("i", "u1"), ("v", "u1")])


def compact(toks):
    """{I: (path, V, text)} of the latest version of each file, from a session's token log."""
    files = {}
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
        files[i] = (tuple(toks[ns:toks.index(F.NL, ns)]), v, toks[ts:te])
    return files


def compact_links(toks, files):
    """{(S, D)}: the links after every link_files / unlink_files call the model answered
    "linked" / "removed"."""
    index = {name: i for i, (name, _, _) in files.items()}
    links, p = set(), 0
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
        if F.LINKED not in toks[b:end] and F.REMOVED not in toks[b:end]:
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
    for r, (i, (name, v, body)) in enumerate(sorted(files.items())):
        recs[r]["name"] = list(name) + [NO_TOKEN] * (F.NSLOT - len(name))
        recs[r]["i"], recs[r]["v"] = i, v
        ids = np.array(body, dtype="<u4")
        np.stack([ids & 255, ids >> 8 & 255, ids >> 16 & 255], 1).astype(np.uint8).tofile(
            path / "files" / f"{i}.tok")
    recs.tofile(path / "index.bin")
    np.array(sorted(links), dtype=np.uint8).reshape(-1, 2).tofile(path / "links.bin")


def load_tokens(f):
    b = np.fromfile(f, dtype=np.uint8).reshape(-1, 3).astype(np.int64)
    return (b[:, 0] | b[:, 1] << 8 | b[:, 2] << 16).tolist(), b.size


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


class Session:
    """A fresh session on a stored filesystem, one token per step."""

    def __init__(self, model, path, dev, paging=True):
        self.m, self.path, self.dev, self.paging = model, Path(path), dev, paging
        self.cache = DynamicCache(config=model.config)
        recs = np.fromfile(self.path / "index.bin", dtype=REC)
        self.version = {int(r["i"]): int(r["v"]) for r in recs}
        net = model.model.layers[0].linear_attn
        conv = torch.zeros(1, net.conv_dim, net.conv_kernel_size, dtype=torch.bfloat16, device=dev)
        state = {li: torch.zeros(1, net.num_v_heads, F.DK, F.DK, dtype=torch.float32, device=dev)
                 for li, kind in enumerate(F.LAYERS) if kind == "linear_attention"}
        for r in recs:
            i = int(r["i"])
            row = dual(F.bits(int(r["v"]), F.VB))                     # version table
            state[TABLE][0, 0, i, :len(row)] = torch.tensor(row)
            for j, c in enumerate(slot_codes(r)):                     # name table: a head per slot
                vals = dual([(b + 1) // 2 for b in c])
                state[TABLE][0, 1 + j, i, :len(vals)] = torch.tensor(vals)
        links = np.fromfile(self.path / "links.bin", dtype=np.uint8).reshape(-1, 2)
        for s_, d_ in links.tolist():                                  # link bits
            r = (s_ % 4) * F.NFILE + d_
            state[PAIR][0, s_ // 4, r, d_] = 20.0                      # the target's column
            state[PAIR][0, 8 + s_ // 4, r, s_] = 20.0                  # the source's column
        if recs.size:                                                  # file count
            row = dual(F.bits(len(recs), F.IB))
            state[COUNTL][0, F.NSLOT + 1, 0, :len(row)] = torch.tensor(row)
        for li, rec in state.items():
            layer = self.cache.layers[li]
            layer.lazy_initialization(conv_states=conv, recurrent_states=rec, state_idx=0)
            layer.conv_states[0].copy_(conv)
            layer.recurrent_states[0].copy_(rec)
            layer.has_previous_state[0] = True
        self._put(LOOKUP, *lookup_rows(model, recs))
        n_kv = model.model.layers[COPY].self_attn.k_proj.out_features // F.HD
        empty = torch.zeros(1, n_kv, 0, F.HD, dtype=torch.bfloat16, device=dev)
        self._put(COPY, empty, empty)
        self.loaded, self.load_s, self.load_bytes = set(), 0.0, 0

    def _put(self, layer, k, v):
        c = self.cache.layers[layer]
        c.keys, c.values = k, v
        c.dtype, c.device = k.dtype, k.device
        c.is_initialized = True

    def _page(self, tok):
        if not self.paging or not (F.ACKR0 <= tok < F.NONE):
            return
        i, v = (tok - F.ACKR0) >> F.VB, (tok - F.ACKR0) & ((1 << F.VB) - 1)
        if (i, v) in self.loaded or self.version.get(i) != v:
            return                                 # written in this session: already live
        t0 = time.perf_counter()
        body, nbytes = load_tokens(self.path / "files" / f"{i}.tok")
        k, v_ = copy_rows(self.m, i, v, body)
        c = self.cache.layers[COPY]
        self._put(COPY, torch.cat([c.keys, k], 2), torch.cat([c.values, v_], 2))
        self.load_s += time.perf_counter() - t0
        self.load_bytes += nbytes
        self.loaded.add((i, v))

    @torch.no_grad()
    def step(self, tok):
        """Feed one token; returns the model's next token."""
        self._page(tok)
        out = self.m(input_ids=torch.tensor([[tok]], device=self.dev), past_key_values=self.cache,
                     use_cache=True)
        self.cache = out.past_key_values
        return int(out.logits[0, -1].argmax())
