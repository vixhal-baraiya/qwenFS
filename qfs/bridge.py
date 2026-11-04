"""The chat session behind the TUI: Qwen and the filesystem in its weights, as a stream of events.

`Session.send` runs one assistant turn and yields Events as the tokens come out:
  text      a piece of Qwen's text (a tool call it is writing included, as text)
  token     every token, with its phase (qwen, call, response) and the call's tool: the panel's
            lights follow these
  call      a tool call Qwen finished writing (</tool_call>): its function and parameters
  response  the circuit's tool response, once printed: its lines, the file it acked, and the
            file paged in from disk for it, if any
  fs        the filesystem as the model now holds it, read from its state (qfs.monitor.read)
  saved     the filesystem stored to the session's disk, after a turn that changed it
  done      the turn's end
The loop decodes only: every response token is the model's own prediction.

Run: uv run python -m qfs.bridge MERGED "Save notes/todo.txt: call the dentist"
"""
import json
import re
import shutil
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import torch

from qfs import circuit as F
from qfs import disk, monitor
from qfs.chat import Chat

DOC = Path(__file__).parents[1] / "docs" / "fs-agent.md"


@dataclass
class Event:
    kind: str
    data: dict = field(default_factory=dict)


class Cancelled(Exception):
    pass


def parse_call(text):
    """(function, {parameter: value}) of a Qwen3.5 tool call's text."""
    f = re.search(r"<function=([^>\s]+)>", text)
    params = dict(re.findall(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", text, re.DOTALL))
    return (f.group(1) if f else "?"), params


# the calls that change the filesystem
CHANGES = {"write_file", "append_file", "link_files", "unlink_files"}


def stored(path):
    """Whether `path` holds a filesystem image."""
    return path is not None and (Path(path) / "index.bin").exists()


class Session:
    """With a `disk`, the filesystem persists: the session starts on the image stored there (if
    any), saves to it after every turn that changed the filesystem, and an interrupt keeps it."""

    def __init__(self, merged, device=None, doc=True, temp=0.7, seed=0, mount=None, disk=None):
        self.dev = device or ("cuda" if torch.cuda.is_available() else
                              "mps" if torch.backends.mps.is_available() else "cpu")
        self.merged = Path(merged)
        try:
            self.meta = json.loads((self.merged / "fs.json").read_text())
        except (OSError, ValueError):
            self.meta = {}
        self.name = self.meta.get("base", "Qwen3.5")
        self.doc = doc
        self.chat = Chat(str(merged), self.dev, temp=temp, seed=seed)
        n = self.chat.m.config.num_hidden_layers
        self.meta.setdefault("fs_layers", len(F.LAYERS))
        self.meta["qwen_layers"] = n - self.meta["fs_layers"]
        self.tok = self.chat.tok
        self.cancel = False
        self.seed = seed
        self.disk = Path(disk) if disk else None
        self.reset(mount or (self.disk if stored(self.disk) else None))

    # the filesystem -------------------------------------------------------------------------
    def reset(self, mount=None):
        """A new chat: on an empty filesystem, or on the one stored at `mount`."""
        system = DOC.read_text() if self.doc else None
        self.chat.reset(self.seed, system, mount=mount)
        self.mounted = Path(mount) if mount else None
        self.where = str(self.mounted) if mount else None   # how the panel names the mount
        self.image = disk.load(mount)[0] if mount else {}
        # rows the mount put in the copy layer before any token (the search rows)
        c = self.chat.cache.layers[disk.COPY]
        self.mount_rows = c.keys.shape[-2] if getattr(c, "keys", None) is not None else 0
        self.stats = {"turns": 0, "calls": 0, "forced": 0, "tokens": 0, "seconds": 0.0,
                      "paged": 0, "paged_bytes": 0}
        self._sizes()

    def path(self, name):
        return self.tok.decode(list(name))

    def state(self):
        """The filesystem as the model holds it: files ({I: (path, V, text tokens)}), links,
        and the copy layer's rows, read from the model's state; text sizes from the log."""
        st = monitor.read(self.chat.cache)
        files = {i: (self.path(n), v, self.sizes.get(i, 0)) for i, (n, v) in st["files"].items()}
        return {"files": files, "links": st["links"], "rows": st["rows"], "lookup": st["lookup"],
                "vbits": st["vbits"], "copy": self.copy_map(st["files"]), "mounted": self.where}

    def copy_map(self, live):
        """The copy layer's rows as segments (start, end, kind, I): every token is a row
        ("context"), a written text's rows hold that file ("file", a current segment, or "old"),
        and a segment paged in from disk adds its rows where it was paged ("paged"). Rows the
        mount made (the search rows) come first and are left out."""
        pages, m = self.chat.pages, self.mount_rows

        def row(pos):
            return m + pos + sum(n for p, n, _ in pages if p <= pos)

        segs, shift = [], m
        for p, n, i in pages:
            segs.append((p + shift, p + shift + n, "paged", i))
            shift += n
        files = disk.compact(self.chat.ids, self.image)
        for i, v, _, ts, te, _ in disk.writes(self.chat.ids):
            if te > ts:
                top = live.get(i, (None, -1))[1]
                current = i in files and disk.base(*files[i][1:]) <= v <= top
                segs.append((row(ts), row(te - 1) + 1, "file" if current else "old", i))
        return sorted(segs)

    def restore(self):
        """After an interrupt: a new chat on the files so far (the context is cleared, the
        filesystem is kept), through a stored image: the disk's, or a temporary one."""
        import tempfile

        path = self.disk or Path(tempfile.mkdtemp(prefix="qfs-"))
        self.save(path)
        stats = self.stats
        self.reset(path)
        self.stats = stats
        if not self.disk:
            self.where = "your files, kept after the interrupt"

    def erase(self):
        """Erase the disk's image (the chat goes on until the next reset)."""
        if self.disk:
            shutil.rmtree(self.disk, ignore_errors=True)

    def save(self, path):
        """Store the filesystem at `path`, replacing any image there in one step: it is written
        beside it first, so an interrupted save leaves the old image whole. `path` may be the
        mounted image itself (the chat's writes are applied to it)."""
        path = Path(path)
        tmp, old = path.with_name(path.name + ".new"), path.with_name(path.name + ".old")
        shutil.rmtree(tmp, ignore_errors=True)
        files, links = self.chat.save(tmp)
        shutil.rmtree(old, ignore_errors=True)
        if path.exists():
            path.rename(old)
        tmp.rename(path)
        shutil.rmtree(old, ignore_errors=True)
        nbytes = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return {"files": len(files), "links": len(links), "bytes": nbytes, "path": str(path)}

    # a turn ---------------------------------------------------------------------------------
    def send(self, user) -> Iterator[Event]:
        self.cancel = False
        t0 = time.time()
        phase, func, seg, said = "qwen", None, [], ""
        body, call_text, n = None, "", 0
        paged, changed = None, False
        for t in self.chat.stream(user):
            if self.cancel and phase in ("qwen", "call"):   # never inside a response
                self.restore()
                raise Cancelled
            if t == -1:
                yield Event("cut")
                break
            n += 1
            if self.chat.paged is not None:
                paged = self.chat.paged
            if phase == "qwen" and t == F.TCSTART:
                phase, call_text = "call", ""
            yield Event("token", {"t": t, "phase": phase, "func": func})
            if phase in ("qwen", "call"):
                seg.append(t)
                text = self.tok.decode(seg)
                if not text.endswith("�"):        # a character split across tokens: wait
                    yield Event("text", {"text": text[len(said):]})
                    said = text
                if phase == "call":
                    call_text = text.split("<tool_call>", 1)[-1]
                if t == F.TCEND:
                    func, params = parse_call(call_text)
                    self.stats["calls"] += 1
                    yield Event("call", {"func": func, "params": params})
                    phase, body, paged = "response", None, None
                    seg, said = [], ""
                continue
            self.stats["forced"] += 1                  # the response: every token forced
            if t == F.TRESP:
                body = []
            elif t == F.TRESPEND:
                yield Event("response", self._response(func, body or [], paged))
                changed |= func in CHANGES
                if paged:
                    self.stats["paged"] += 1
                    self.stats["paged_bytes"] += paged[1]
                body = None
                self._sizes()
                yield Event("fs", self.state())
            elif body is not None:
                body.append(t)
            if t == F.THINKEND:
                phase = "after"
            elif phase == "after" and t == F.NL2:
                phase, func = "qwen", None             # the new assistant turn's text
        self._sizes()
        yield Event("fs", self.state())
        if changed and self.disk:
            yield Event("saved", self.save(self.disk))
        self.stats["turns"] += 1
        self.stats["tokens"] += n
        self.stats["seconds"] += time.time() - t0
        yield Event("done", {"tokens": n, "seconds": time.time() - t0})

    def _response(self, func, body, paged):
        """A response's text as Qwen reads it (a marker reads as "-"; a skip as a newline, so it
        is left out), and its ack."""
        ack, pieces = None, [[]]
        for t in body:
            if F.ACKW0 <= t < F.NONE:
                x = t - (F.ACKW0 if t < F.ACKR0 else F.ACKR0)
                ack = {"kind": "write" if t < F.ACKR0 else "read", "i": x >> F.VB,
                       "v": x & ((1 << F.VB) - 1)}
            elif t == F.NONE:
                ack = {"kind": "none"}
            elif F.MARK0 <= t < F.MARK0 + F.NFILE:
                pieces.append([])
            elif F.SKIP0 <= t < F.SKIP0 + F.NFILE:
                pass                                   # a file a search passed over
            else:
                pieces[-1].append(t)
        text = self.tok.decode(pieces[0]) + "".join("- " + self.tok.decode(p) for p in pieces[1:])
        out = {"func": func, "lines": text.strip("\n").split("\n"), "ack": ack, "paged": None}
        if paged:
            out["paged"] = {"i": paged[0], "bytes": paged[1], "rows": paged[2]}
        return out

    def _sizes(self):
        """Each file's text, in tokens (all its segments): the mounted image's, then this chat's
        writes and appends."""
        self.sizes = {i: sum(map(len, segs))
                      for i, (_, _, segs) in disk.compact(self.chat.ids, self.image).items()}


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("merged")
    ap.add_argument("messages", nargs="+")
    ap.add_argument("--mount")
    a = ap.parse_args()
    s = Session(a.merged, mount=a.mount)
    for msg in a.messages:
        print(f"\n### user: {msg}")
        for e in s.send(msg):
            if e.kind == "text":
                print(e.data["text"], end="", flush=True)
            elif e.kind == "response":
                print(f"\n[response {e.data['func']}: {e.data['lines']} ack {e.data['ack']} "
                      f"paged {e.data['paged']}]")
            elif e.kind == "fs":
                print("[fs]", {i: f[:2] for i, f in e.data["files"].items()}, e.data["links"])
        print()


if __name__ == "__main__":
    main()
