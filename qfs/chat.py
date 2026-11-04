"""Chat with the merged model; Qwen saves and reads files through the filesystem in its weights.

There is no host: the loop only decodes and stops at <|im_end|>. When Qwen writes a tool call,
the circuit prints the tool response inside the same turn. A chat can start on a stored
filesystem (`mount`); its files page in when the model emits their read acks.

Usage: python -m qfs.chat MERGED "first message" ["second message" ...]
"""
import argparse
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from qfs import circuit as F
from qfs import disk

TOOLS = [
    {"type": "function", "function": {
        "name": "write_file", "description": "Save a text file at a path (replaces it). Use "
        "folders to organise files, e.g. people/alice.txt.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "the file path, e.g. notes/shopping.txt"},
            "text": {"type": "string", "description": "the file's text"}},
            "required": ["name", "text"]}}},
    {"type": "function", "function": {
        "name": "append_file", "description": "Add text to the end of a file, on a new line "
        "(creates the file if it is missing). Use it to add to a file without writing it again.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "the file path"},
            "text": {"type": "string", "description": "the text to add"}},
            "required": ["name", "text"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a saved text file, exactly as it was saved.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "the file path"}},
            "required": ["name"]}}},
    {"type": "function", "function": {
        "name": "list_files", "description": "List the paths of all saved files.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "link_files", "description": "Link a file to a related file (e.g. a person to "
        "a project they work on).",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "the file to link from"},
            "target": {"type": "string", "description": "the related file to link to"}},
            "required": ["name", "target"]}}},
    {"type": "function", "function": {
        "name": "unlink_files", "description": "Remove the link from a file to another.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "description": "the file the link is from"},
            "target": {"type": "string", "description": "the file the link is to"}},
            "required": ["name", "target"]}}},
    {"type": "function", "function": {
        "name": "list_links", "description": "List the files a file is linked to.",
        "parameters": {"type": "object", "properties": {
            "file": {"type": "string", "description": "the file path"}},
            "required": ["file"]}}},
    {"type": "function", "function": {
        "name": "list_backlinks", "description": "List the files that link to a file (e.g. "
        "who works on a project).",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "the file path"}},
            "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "search_files", "description": "Find the files whose text has a word (exact "
        "and case-sensitive, e.g. Paris).",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "one word"}},
            "required": ["query"]}}},
]
IM_END = 248046


def show(tok, ids):
    out = []
    for t in ids:
        if F.ACKW0 <= t < F.ACKR0:
            x = t - F.ACKW0
            out.append(f"[w {x >> F.VB} {x & 63}]")
        elif F.ACKR0 <= t < F.NONE:
            x = t - F.ACKR0
            out.append(f"[r {x >> F.VB} {x & 63}]")
        elif t == F.NONE:
            out.append("[none]")
        elif F.MARK0 <= t < F.MARK0 + F.NFILE:
            out.append(f"[m {t - F.MARK0}]")
        elif F.SKIP0 <= t < F.SKIP0 + F.NFILE:
            out.append(f"[s {t - F.SKIP0}]")
        elif t == -1:
            out.append(" [cut at the token limit]")
        else:
            out.append(tok.decode([t]))
    return "".join(out)


class Chat:
    def __init__(self, path, dev, temp=0.0, top_p=0.9, seed=0, system=None):
        self.dev, self.temp, self.top_p = dev, temp, top_p
        self.tok = AutoTokenizer.from_pretrained(path)
        self.m = Qwen3_5ForCausalLM.from_pretrained(path, dtype=torch.bfloat16).to(dev).eval()
        disk.attach(self.m)
        self.reset(seed, system)

    def reset(self, seed=0, system=None, mount=None):
        """A new conversation on the same model: on an empty filesystem, or on the one stored
        at `mount`."""
        self.gen = torch.Generator(device="cpu").manual_seed(seed)
        self.system = system
        self.ids, self.pager, self.mounted = [], None, mount
        if mount is None:
            self.cache = disk.FSCache(self.m.config)
        else:
            self.cache, version = disk.mount(self.m, mount, self.dev)
            self.pager = disk.Pager(self.m, mount, version)
        self.paged = None            # (I, bytes, rows) of a file the last feed paged in
        self.pages = []              # every page-in: (token position, rows, I)

    @torch.no_grad()
    def feed(self, new):
        """Run new tokens through the model; returns the next token."""
        self.paged = None
        if self.pager is not None and len(new) == 1:   # read acks are generated, one at a time
            self.paged = self.pager(self.cache, new[0])
            if self.paged:
                self.pages.append((len(self.ids), self.paged[2], self.paged[0]))
        out = self.m(input_ids=torch.tensor([new], device=self.dev), past_key_values=self.cache,
                     use_cache=True)
        self.cache = out.past_key_values
        self.ids += new
        return self.pick(out.logits[0, -1].float())

    def save(self, path):
        """Store the filesystem at `path`: the mounted image, if any, with this chat's writes,
        links and unlinks applied. Returns the image's files ({I: (path, V, text)}) and links."""
        files, links = disk.load(self.mounted) if self.mounted else ({}, set())
        files = disk.compact(self.ids, files)
        links = disk.compact_links(self.ids, files, links)
        disk.save(files, path, links)
        return files, links

    def pick(self, logits):
        """Greedy, or nucleus sampling at temperature temp. A forced token (an ack, a read-back,
        the response frame) leads by hundreds of logits, so sampling still takes it."""
        if self.temp <= 0:
            return int(logits.argmax())
        p = (logits / self.temp).softmax(-1).cpu()
        sp, si = p.sort(descending=True)
        keep = sp.cumsum(0) - sp < self.top_p
        sp = sp * keep
        return int(si[torch.multinomial(sp / sp.sum(), 1, generator=self.gen)])

    def prompt(self, user):
        if not self.ids:
            msgs = ([{"role": "system", "content": self.system}] if self.system else [])
            return self.tok.apply_chat_template(msgs + [{"role": "user", "content": user}],
                                                tools=TOOLS, tokenize=False,
                                                add_generation_prompt=True, enable_thinking=False)
        return (f"\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
                "<think>\n\n</think>\n\n")

    @torch.no_grad()
    def stream(self, user, limit=600):
        """One assistant turn, a token at a time: yields each token the model emits, then -1 if
        the turn was cut at the limit (a cut turn is closed too)."""
        nxt = self.feed(self.tok(self.prompt(user)).input_ids)
        n = 0
        while nxt != IM_END and n < limit:
            yield nxt
            nxt = self.feed([nxt])
            n += 1
        self.feed([IM_END])
        if nxt != IM_END:
            yield -1

    def turn(self, user, limit=600):
        t0 = time.time()
        reply = list(self.stream(user, limit))
        return show(self.tok, reply), len(reply), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merged")
    ap.add_argument("messages", nargs="+")
    ap.add_argument("--temp", type=float, default=0.7, help="0 = greedy (a 2B model loops on tool calls when greedy)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mount", help="start on the filesystem stored in this directory")
    ap.add_argument("--doc", default=str(Path(__file__).parents[1] / "docs" / "fs-agent.md"),
                    help="the system prompt (default: docs/fs-agent.md; '' for none)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "mps"
    chat = Chat(a.merged, dev, temp=a.temp, seed=a.seed,
                system=open(a.doc).read() if a.doc else None)
    if a.mount:
        chat.reset(a.seed, chat.system, mount=a.mount)
    for msg in a.messages:
        reply, n, dt = chat.turn(msg)
        print(f"\n### user: {msg}\n### assistant ({n} tokens, {dt:.0f}s):\n{reply}")


if __name__ == "__main__":
    main()
