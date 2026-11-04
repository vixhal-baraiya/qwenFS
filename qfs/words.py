"""Words as search_files matches them: the token table the circuit's embeddings carry, and the same
rule in Python (the reference the tests check the circuit against, and the disk's search rows).

A word is a run of word tokens (letters and digits only, after a leading space) in a written text:
its first token may start with a space, the others may not, and it ends where the next token is
not a word token or starts a new word. A word is written the way Qwen writes the word alone (a
query): a token with a leading space stands for the tokens of its text without the space
(" designer" -> "design", "er"), every other token for itself. A word of more than S tokens that
way never matches. Matching is exact and case-sensitive.

Each token's row: its form right-aligned in S slots (the last token in slot 0, EMPTY after the
first), and three flags: a word token (WD), one that starts with a space (SP), and one whose form
is a single token (X1).
"""
import hashlib
from pathlib import Path

import numpy as np

from qfs.circuit import IDB

S = 3                                   # a word's tokens, at most
EMPTY = (1 << IDB) - 1                  # an unused slot (the circuit's all +1 code)
NEVER = (1 << IDB) - 2                  # a form no query has
WD, SP, X1 = 0, 1, 2
CACHE = Path.home() / ".cache" / "qfs"


def build(tok, vocab):
    """(slots (vocab, S) int, flags (vocab, 3) bool) for a Qwen3.5 tokenizer; ids past the
    tokenizer's (the FS tokens) are not words."""
    n = len(tok)
    pieces = tok.convert_ids_to_tokens(list(range(n)))
    special = set(tok.all_special_ids)
    texts = [tok.convert_tokens_to_string([p]) if p is not None else "" for p in pieces]
    slots = np.full((vocab, S), EMPTY, dtype=np.int64)
    flags = np.zeros((vocab, 3), dtype=bool)
    spaced = []
    for t, s in enumerate(texts):
        w = s.removeprefix(" ")
        if t in special or not w or not w.isalnum():
            continue
        flags[t, WD] = True
        if s.startswith(" "):
            flags[t, SP] = True
            spaced.append((t, w))
        else:
            slots[t, 0] = t
            flags[t, X1] = True
    forms = tok([w for _, w in spaced], add_special_tokens=False)["input_ids"]
    for (t, _), form in zip(spaced, forms):
        if len(form) > S:
            slots[t, 0] = NEVER
            continue
        for r, u in enumerate(reversed(form)):
            slots[t, r] = u
        flags[t, X1] = len(form) == 1
    return slots, flags


def table(tok, vocab):
    """`build`, cached by the tokenizer's vocabulary."""
    key = hashlib.sha1(repr(sorted(tok.get_vocab().items())).encode()).hexdigest()[:16]
    f = CACHE / f"words-{key}-{vocab}-{S}-{IDB}.npz"
    if f.exists():
        z = np.load(f)
        return z["slots"], z["flags"]
    slots, flags = build(tok, vocab)
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(f, slots=slots, flags=flags)
    return slots, flags


def tokenizer(src=None):
    """The Qwen3.5 tokenizer: from `src` (a checkpoint), else the merged build, else the Hub."""
    from transformers import AutoTokenizer

    for p in (src, "build/fs-qwen-2b"):
        if p and (Path(p) / "tokenizer.json").exists():
            return AutoTokenizer.from_pretrained(p)
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")


def words(toks, slots, flags):
    """The words of a text (its tokens) as {position of the token after it: its form}, the
    form right-aligned in S slots, as the circuit keys them. The text is followed by a newline."""
    toks = list(toks) + [-1]
    wd = lambda t: t >= 0 and flags[t, WD]
    sp = lambda t: t >= 0 and flags[t, SP]
    out = {}
    for b in range(1, len(toks)):
        if wd(toks[b]) and not sp(toks[b]):
            continue                       # not the end of a word
        run = []
        for k in range(b - 1, max(-1, b - S - 1), -1):
            if not wd(toks[k]):
                break
            run.append(toks[k])
            if sp(toks[k]):
                break
        if not run:
            continue
        m, first = len(run), run[-1]
        before = b - m - 1
        if not sp(first) and before >= 0 and wd(toks[before]):
            continue                       # the word is longer than S tokens
        if m == 1:
            out[b] = tuple(int(x) for x in slots[first])
        elif flags[first, X1]:
            out[b] = tuple(int(slots[t, 0]) for t in run) + (EMPTY,) * (S - m)
    return out


def query(toks):
    """A query's form (Qwen writes the word alone), right-aligned; None if it has more than S
    tokens or none."""
    if not 0 < len(toks) <= S:
        return None
    return tuple(reversed([int(t) for t in toks])) + (EMPTY,) * (S - len(toks))
