"""The filesystem from disk: session A writes files through tool calls, its image is cut from the log,
and a fresh session B (empty context) lists, reads, writes and lists again, paging files in when
the model emits their read acks, and searches (the search rows are built at mount, so a search
finds unpaged files too). B runs one token at a time; every forced token is checked
against the model's own prediction. The control runs B without paging. Last, B's chat is saved
over the image it mounted (as the app's disk does after every turn), twice.

Usage: python -m tests.test_disk [--seed 0] [--files 12] [--maxlen 300] [--model MERGED]
"""
import argparse
import random
import shutil
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import torch

from qfs import bridge, disk
from qfs import circuit as F
from qfs import words as W
from qfs.chat import Chat
from tests.test_circuit import Session, names, text


def run(sess, s):
    right, wrong = 0, set()
    for p, tok in enumerate(s.toks[:-1]):
        pred = sess.step(tok)
        if p + 1 in s.forced:
            if pred == s.forced[p + 1]:
                right += 1
            else:
                wrong.add(p + 1)
    exact = sum(not any(st <= w < st + len(want) for w in wrong) for st, want in s.reads)
    return right, exact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--files", type=int, default=12)
    ap.add_argument("--maxlen", type=int, default=300)
    ap.add_argument("--model")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "mps"
    if a.model:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
        model = Qwen3_5ForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).to(dev).eval()
    else:
        model = F.build(dev, F.calibrate(dev), W.table(W.tokenizer(), F.VOCAB))
    Session.words = W.table(W.tokenizer(a.model), F.VOCAB)
    rng = random.Random(a.seed)
    path = Path(tempfile.mkdtemp(prefix="qfs4-"))

    all_names = names(rng, a.files + 6)
    stored, missing, fresh = all_names[:a.files], all_names[a.files:a.files + 4], all_names[-2:]
    A = Session(rng)
    for nm in stored:
        A.write(nm, text(rng, rng.randrange(1, a.maxlen)))
    for nm in rng.sample(stored, a.files // 4):
        A.write(nm, text(rng, rng.randrange(1, a.maxlen)))
    for nm in rng.sample(stored, max(1, a.files // 8)):
        A.write(nm, [])
    for nm in rng.sample(stored, a.files // 3):                  # appends, and a long file
        A.append(nm, text(rng, rng.randrange(1, a.maxlen)))
    for _ in range(3):
        A.append(stored[0], text(rng, a.maxlen))
    for _ in range(2 * a.files):
        A.link(rng.choice(stored[:4]), rng.choice(stored))
    for x, y in rng.sample(sorted(A.links, key=str), 3):
        A.unlink(x, y)
    with torch.no_grad():
        pred = model(input_ids=torch.tensor([A.toks], device=dev)).logits[0].argmax(-1).tolist()
    ok = sum(pred[p - 1] == t for p, t in A.forced.items())
    files = disk.compact(A.toks)
    assert {i: (nm, v, segs) for i, (nm, v, segs) in files.items()} == \
        {i: (nm, v, A.files[nm]) for nm, (i, v) in A.index.items()}, \
        "compact disagrees with the reference"
    links = disk.compact_links(A.toks, files)
    idx = {nm: i for nm, (i, _) in A.index.items()}
    assert links == {(idx[x], idx[y]) for x, y in A.links}, "compact_links disagrees"
    disk.save(files, path, links)
    nbytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    live = sum(len(seg) for _, _, segs in files.values() for seg in segs)
    print(f"session A: {len(A.toks)} tokens, forced {ok}/{len(A.forced)} right; disk {nbytes} bytes "
          f"for {len(files)} files, {live} live tokens ({nbytes / max(live, 1):.2f} B/token)")

    B = Session(rng)
    B.index, B.files = dict(A.index), {nm: list(segs) for nm, segs in A.files.items()}
    B.links = set(A.links)
    B.listing()
    for nm in stored[:4] + missing[:1]:
        B.list_links(nm)
        B.list_backlinks(nm)
    order = stored + missing
    rng.shuffle(order)
    for nm in order:
        B.read(nm)
    B.write(fresh[0], text(rng, 150))
    B.write(rng.choice(stored[1:]), text(rng, 120))
    B.append(stored[0], text(rng, 80))      # a stored file's segments, then one written here
    B.read(stored[0])
    B.link(fresh[0], stored[0])
    B.link(stored[1], fresh[0])
    B.unlink(*sorted(A.links, key=str)[0])
    B.listing()
    for nm in (fresh[0], stored[1], stored[0]):
        B.list_links(nm)
        B.list_backlinks(nm)
    for nm in (fresh[0], fresh[1]):
        B.read(nm)
    forms = [f for nm in stored for f in B.word_forms(nm) if W.NEVER not in f]
    for f in rng.sample(forms, min(6, len(forms))) + [(W.EMPTY,) * W.S]:
        B.search([t for t in reversed(f) if t != W.EMPTY])     # (the last: an empty query)
    print(f"session B: {len(B.toks)} tokens, {len(B.reads)} reads/listings, {len(B.forced)} forced")
    for paging in (True, False):
        t0 = time.time()
        sess = disk.Session(model, path, dev, paging=paging)
        mount = time.time() - t0
        t0 = time.time()
        right, exact = run(sess, B)
        label = "paged from disk" if paging else "control, no paging"
        print(f"  {label}: {right}/{len(B.forced)} forced right, {exact}/{len(B.reads)} exact "
              f"({time.time() - t0:.0f}s; mount {mount * 1000:.0f} ms)")
        if paging:
            print(f"    paged {len(sess.loaded)} files: {sess.load_bytes} bytes, rows built in "
                  f"{sess.load_s * 1000:.0f} ms")
    persisted(path, B)
    shutil.rmtree(path)


def persisted(path, B):
    """B's chat saved over the image it mounted holds B's filesystem, and saving again changes
    nothing."""
    chat = SimpleNamespace(mounted=path, ids=B.toks)
    chat.save = Chat.save.__get__(chat)
    sess = SimpleNamespace(chat=chat)
    images = []
    for _ in range(2):
        bridge.Session.save(sess, path)
        assert not path.with_name(path.name + ".new").exists(), "the new image was left beside it"
        assert not path.with_name(path.name + ".old").exists(), "the old image was left beside it"
        images.append({f.relative_to(path): f.read_bytes() for f in path.rglob("*")
                       if f.is_file()})
    files, links = disk.load(path)
    idx = {nm: i for nm, (i, _) in B.index.items()}
    ok = ({i: (nm, v) for i, (nm, v, _) in files.items()}
          == {i: (nm, v) for nm, (i, v) in B.index.items()}
          and {i: segs for i, (_, _, segs) in files.items()}
          == {idx[nm]: segs for nm, segs in B.files.items()}
          and links == {(idx[x], idx[y]) for x, y in B.links})
    same = images[0] == images[1]
    print(f"  saved over the mount: {'B' if ok else 'NOT B'}'s filesystem; "
          f"saved again: {'unchanged' if same else 'CHANGED'}")
    assert ok and same


if __name__ == "__main__":
    main()
