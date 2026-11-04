"""The circuit on scripted Qwen tool calls, against a reference filesystem.

Sessions are written as Qwen3.5 writes tool calls (write_file with name and text, read_file with
name, and the rest); everything the circuit must produce (the write ack where the text starts,
the tool response frames, the read-backs, <none> for missing names, listings, search results) is
checked: in one prefill, and by greedy decoding with the cache from each read's </tool_call>.
Searches are checked against qfs.words, the rule in Python.

Usage: python -m tests.test_circuit [--seed 0] [--files 10] [--maxlen 300] [--model MERGED]
"""
import argparse
import random
import time

import torch

from qfs import circuit as F
from qfs import words as W

TC, LT, FUNCTION, EQ, WRITE, EQREAD, FILE, PARAMETER = 248058, 27, 1628, 28, 4775, 86779, 2378, 15704
SKIP = {F.EQNAME, F.EQTEXT, F.EQQUERY}


def text(rng, n):
    out = []
    while len(out) < n:
        t = rng.randrange(1000, 200000)
        if t not in SKIP:
            out.append(t)
    return out


def names(rng, n):
    out = []
    while len(out) < n:
        if out and rng.random() < 0.3:
            base = list(rng.choice(out))
            nm = tuple(base[:-1]) if len(base) > 1 and rng.random() < 0.5 else \
                tuple(base + text(rng, 1))[:F.NSLOT]
        else:
            nm = tuple(text(rng, rng.randrange(1, F.NSLOT + 1)))
        if nm and nm not in out:
            out.append(nm)
    return out


def call_head(func, name, nl=F.NL):
    """nl: the newline that starts the name (Qwen writes \n or, sometimes, \n\n)."""
    return [TC, F.NL, LT, FUNCTION, *func, F.GT, F.NL, LT, PARAMETER, F.EQNAME, F.GT, nl,
            *name, F.NL, F.LTS, PARAMETER, F.GT, F.NL]


CALL_TAIL = [F.LTS, FUNCTION, F.GT, F.NL, F.TCEND]
HEAD = [F.NL, F.IMSTART, F.USER, F.NL, F.TRESP, F.NL]
TURN = [F.NL, F.IMSTART, F.ASSISTANT, F.NL, F.THINK, F.NL2, F.THINKEND, F.NL2]


class Session:
    words = None                       # (slots, flags) for searches: qfs.words.table

    def __init__(self, rng):
        self.rng, self.index, self.files, self.links = rng, {}, {}, set()
        self.toks, self.forced, self.reads = text(rng, 20), {}, []

    def force(self, ts):
        for t in ts:
            self.forced[len(self.toks)] = t
            self.toks.append(t)

    def write(self, nm, body):
        if nm not in self.index:
            self.index[nm] = (len(self.index), 0)
        i, v = self.index[nm]
        self.index[nm] = (i, v + 1)
        self.files[nm] = body
        nl = lambda: F.NL2 if self.rng.random() < 0.3 else F.NL
        self.toks += call_head([EQ, WRITE, FILE], nm, nl()) + [LT, PARAMETER, F.EQTEXT, F.GT, nl()]
        self.toks += body + [F.NL, F.LTS, PARAMETER, F.GT, F.NL] + CALL_TAIL
        self.force(HEAD + [F.ackw(i, v + 1), F.SAVED, F.NL, F.MARK0 + i, *nm, F.NL, F.TRESPEND]
                   + TURN)
        self.toks += text(self.rng, self.rng.randrange(1, 10))

    def listing(self):
        self.toks += [TC, F.NL, LT, FUNCTION, EQ, 1001, F.GT, F.NL, F.LTS, FUNCTION, F.GT, F.NL,
                      F.TCEND]
        start = len(self.toks)
        paths = sorted(self.index.items(), key=lambda kv: kv[1][0])
        if paths:
            body = [t for nm, (i, _) in paths for t in [F.MARK0 + i, *nm, F.NL]]
        else:
            body = [F.EMPTYW, F.NL]
        want = HEAD + body + [F.TRESPEND] + TURN
        self.force(want)
        self.reads.append((start, want))
        self.toks += text(self.rng, self.rng.randrange(1, 10))

    def param(self, eq, path):
        nl = F.NL2 if self.rng.random() < 0.3 else F.NL
        return [LT, PARAMETER, eq, F.GT, nl, *path, F.NL, F.LTS, PARAMETER, F.GT, F.NL]

    def link(self, nm, target, unlink=False):
        func = [EQ, 64778 if unlink else 2012, FILE]          # "=", "unlink"/"link", "_files"
        self.toks += [TC, F.NL, LT, FUNCTION, *func, F.GT, F.NL]
        self.toks += self.param(F.EQNAME, nm) + self.param(F.EQTARGET, target) + CALL_TAIL
        start = len(self.toks)
        if nm in self.index and target in self.index:
            (self.links.discard if unlink else self.links.add)((nm, target))
            body = [F.REMOVED if unlink else F.LINKED, F.NL, F.MARK0 + self.index[nm][0], *nm,
                    F.NL, F.TO, F.NL, F.MARK0 + self.index[target][0], *target, F.NL]
        else:
            body = [F.NONE, F.NOT, F.FOUND_, F.NL]
        self.respond(start, body)

    def unlink(self, nm, target):
        self.link(nm, target, unlink=True)

    def respond(self, start, body):
        want = HEAD + body + [F.TRESPEND] + TURN
        self.force(want)
        self.reads.append((start, want))
        self.toks += text(self.rng, self.rng.randrange(1, 10))

    def listed(self, nm, which):
        if nm not in self.index:
            return [F.NONE, F.NOT, F.FOUND_, F.NL]
        other = sorted({b for a, b in self.links if a == nm} if which == "links" else
                       {a for a, b in self.links if b == nm}, key=lambda x: self.index[x][0])
        if not other:
            return [F.EMPTYW, F.NL]
        return [t for x in other for t in [F.MARK0 + self.index[x][0], *x, F.NL]]

    def list_links(self, nm):
        self.toks += [TC, F.NL, LT, FUNCTION, EQ, 1003, F.GT, F.NL] + self.param(F.EQFILE, nm)
        self.toks += CALL_TAIL
        self.respond(len(self.toks), self.listed(nm, "links"))

    def list_backlinks(self, nm):
        self.toks += [TC, F.NL, LT, FUNCTION, EQ, 1004, F.GT, F.NL] + self.param(F.EQPATH, nm)
        self.toks += CALL_TAIL
        self.respond(len(self.toks), self.listed(nm, "back"))

    def word_forms(self, nm):
        return set(W.words(self.files[nm], *self.words).values())

    def search(self, q):
        """search_files with the query's tokens q."""
        self.toks += [TC, F.NL, LT, FUNCTION, EQ, 1005, FILE, F.GT, F.NL]
        self.toks += self.param(F.EQQUERY, q) + CALL_TAIL
        form, body = W.query(q), []
        if form is not None:
            for nm, (i, _) in sorted(self.index.items(), key=lambda kv: kv[1][0]):
                body += [F.MARK0 + i, *nm, F.NL] if form in self.word_forms(nm) else [F.SKIP0 + i]
        if not any(F.MARK0 <= t < F.MARK0 + F.NFILE for t in body):
            body += [F.EMPTYW, F.NL]
        self.respond(len(self.toks), body)

    def read(self, nm):
        self.toks += call_head([EQREAD, FILE], nm, F.NL2 if self.rng.random() < 0.3 else F.NL) + CALL_TAIL
        start = len(self.toks)
        if nm in self.index:
            i, v = self.index[nm]
            want = HEAD + [F.ackr(i, v)] + self.files[nm] + [F.NL, F.TRESPEND] + TURN
        else:
            want = HEAD + [F.NONE, F.NOT, F.FOUND_, F.NL, F.TRESPEND] + TURN
        self.force(want)
        self.reads.append((start, want))
        self.toks += text(self.rng, self.rng.randrange(1, 10))


def make(seed, nfiles, maxlen, misses=3):
    rng = random.Random(seed)
    s = Session(rng)
    s.listing()                                    # nothing stored yet
    all_names = names(rng, nfiles + misses)
    stored, missing = all_names[:nfiles], all_names[nfiles:]
    for nm in stored:
        s.write(nm, text(rng, rng.randrange(1, maxlen)))
    for nm in rng.sample(stored, max(1, nfiles // 4)):
        s.write(nm, text(rng, rng.randrange(1, maxlen)))
    for nm in rng.sample(stored, max(1, nfiles // 8)):
        s.write(nm, [])
    s.listing()
    pool = stored + missing[:1]
    for _ in range(2 * nfiles):                    # links (some repeated, some to missing names)
        s.link(rng.choice(stored[:4] + pool), rng.choice(pool))
    for a, b in rng.sample(sorted(s.links, key=str), min(4, len(s.links))):   # unlinks
        s.unlink(a, b)
    s.unlink(stored[0], stored[-1])                # perhaps never linked: still ok
    s.unlink(missing[0], stored[0])                # a missing file: not found
    for nm in stored[:4] + missing[:1] + stored[-2:]:
        s.list_links(nm)
        s.list_backlinks(nm)
    order = stored + missing
    rng.shuffle(order)
    for k, nm in enumerate(order):
        s.read(nm)
        if k == len(order) // 2:
            s.listing()
    if s.words is not None:
        searches(rng, s, stored)
    return s


def searches(rng, s, stored):
    """Words of current files (one, two and three tokens), words only an older version had, words
    no file has, too long and empty queries; then a file rewritten and searched again."""
    def form_toks(f):
        return [t for t in reversed(f) if t != W.EMPTY]

    live = [f for nm in stored for f in s.word_forms(nm) if W.NEVER not in f]
    by_len = {n: [f for f in live if len(form_toks(f)) == n] for n in (1, 2, 3)}
    qs = [form_toks(rng.choice(fs)) for n, fs in by_len.items() for _ in range(2) if fs]
    qs += [list(text(rng, 1)), list(text(rng, 4)), []]
    for q in qs:
        s.search(q)
    # words several files share: " Paris" (one token) and "milk" ("m" "ilk", at a line's start);
    # one file has them only in its old text
    paris, milk = [11751, 11], [F.NL, 76, 32062, 11]
    shared = rng.sample(stored, min(4, len(stored)))
    for k, nm in enumerate(shared):
        body = text(rng, rng.randrange(5, 40))
        cut = rng.randrange(len(body))
        s.write(nm, body[:cut] + paris + milk + body[cut:])
    s.write(shared[-1], text(rng, 30))
    for q in ([57590], [76, 32062], [57590, 76]):     # "Paris", "milk", "Parism"
        s.search(q)
    nm = stored[0]
    if s.files[nm]:
        q = form_toks(next(iter(s.word_forms(nm)), (W.EMPTY,) * W.S))
        s.write(nm, text(rng, 20))
        if q:
            s.search(q)                    # the old text had it: not found unless the new one has


@torch.no_grad()
def check_prefill(model, s, dev):
    pred = model(input_ids=torch.tensor([s.toks], device=dev)).logits[0].argmax(-1).tolist()
    bad = sorted(p for p, t in s.forced.items() if pred[p - 1] != t)
    return len(s.forced) - len(bad), len(s.forced), bad[:5]


@torch.no_grad()
def check_generate(model, s, dev):
    ok = 0
    for start, want in s.reads:
        out = model(input_ids=torch.tensor([s.toks[:start]], device=dev), use_cache=True)
        cache, got = out.past_key_values, []
        nxt = int(out.logits[0, -1].argmax())
        while len(got) < len(want) + 5:
            got.append(nxt)
            if got[-2:] == [F.THINKEND, F.NL2]:
                break
            out = model(input_ids=torch.tensor([[nxt]], device=dev), past_key_values=cache,
                        use_cache=True)
            cache, nxt = out.past_key_values, int(out.logits[0, -1].argmax())
        if got == want:
            ok += 1
        else:
            first = next((i for i, (a, b) in enumerate(zip(got, want)) if a != b),
                         min(len(got), len(want)))
            print(f"    read at {start}: wanted {len(want)}, first difference at {first}: got "
                  f"{got[first] if first < len(got) else None}, want "
                  f"{want[first] if first < len(want) else None}")
    return ok, len(s.reads)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--files", type=int, default=10)
    ap.add_argument("--maxlen", type=int, default=300)
    ap.add_argument("--model", help="a merged checkpoint instead of the standalone circuit")
    ap.add_argument("--no-generate", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "mps"
    t0 = time.time()
    if a.model:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        model = Qwen3_5ForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    else:
        model = F.build(dev, F.calibrate(dev), W.table(W.tokenizer(), F.VOCAB))
    print(f"model ready in {time.time() - t0:.0f}s")
    Session.words = W.table(W.tokenizer(a.model), F.VOCAB)
    s = make(a.seed, a.files, a.maxlen)
    print(f"session: {len(s.toks)} tokens, {len(s.reads)} reads, {len(s.forced)} forced tokens")
    good, total, bad = check_prefill(model, s, dev)
    print(f"prefill:  {good}/{total} forced tokens right" + (f", first wrong at {bad}" if bad else ""))
    if bad:
        p = bad[0]
        print("   context:", s.toks[max(0, p - 6):p + 1], "want", s.forced[p])
    if not a.no_generate:
        t0 = time.time()
        ok, n = check_generate(model, s, dev)
        print(f"generate: {ok}/{n} reads exact ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
