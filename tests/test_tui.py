"""The chat app, headless, on a scripted session (no model): every event renders, the panel
follows the filesystem, and the slash commands work.

Usage: python -m tests.test_tui [--screenshot out.svg]
"""
import argparse
import asyncio
import shutil
import tempfile
import time
from pathlib import Path

from qfs import circuit as F
from qfs import tui
from qfs.bridge import Event

ALICE = "Alice is a designer in Pune."


class FakeSession:
    """Plays back the events of real turns (as qfs.bridge emits them)."""

    name, dev = "fake", "cpu"
    meta = {"base": "Qwen3.5-2B", "fs_layers": 8, "qwen_layers": 24}
    delay = 0.0

    def __init__(self, merged, device=None, doc=True, mount=None):
        self.doc, self.cancel = doc, False
        self.saved = None
        self.reset(mount)

    def reset(self, mount=None):
        self.mounted = Path(mount) if mount else None
        self.where = str(mount) if mount else None
        self.files, self.links = ({0: ("people/alice.txt", 1, 7)}, set()) if mount else ({}, set())
        self.stats = {"calls": 0}

    def state(self):
        vbits = [F.bits(self.files[i][1], F.VB) if i in self.files else [0] * F.VB
                 for i in range(F.NFILE)]
        rows = 1500 + 40 * len(self.files)
        copy = [(1400 + 40 * k, 1407 + 40 * k, "file", i) for k, i in enumerate(self.files)]
        if self.mounted:
            copy.append((1450, 1457, "paged", 0))
        return {"files": dict(self.files), "links": set(self.links), "rows": rows,
                "lookup": 1, "vbits": vbits, "copy": copy, "mounted": self.where}

    def save(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "index.bin").write_bytes(b"x")
        self.saved = path
        return {"files": len(self.files), "links": len(self.links), "bytes": 123,
                "path": str(path)}

    def _call(self, func, params, lines, ack, paged=None):
        text = "<tool_call>\n<function=" + func + ">\n" + "".join(
            f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items()) + "</function>\n"
        for k in range(0, len(text), 6):
            time.sleep(self.delay)
            yield Event("token", {"t": F.TCSTART if k == 0 else 100, "phase": "call",
                                  "func": None})
            yield Event("text", {"text": text[k:k + 6]})
        yield Event("token", {"t": F.TCEND, "phase": "call", "func": None})
        self.stats["calls"] += 1
        yield Event("call", {"func": func, "params": params})
        for t in (F.TRESP, F.ACKW0, 100, F.MARK0, 100, F.TRESPEND, F.THINKEND):
            yield Event("token", {"t": t, "phase": "response", "func": func})
        yield Event("response", {"func": func, "lines": lines, "ack": ack, "paged": paged})
        yield Event("fs", self.state())

    def _say(self, text):
        for w in text.split(" "):
            yield Event("token", {"t": 100, "phase": "qwen", "func": None})
            yield Event("text", {"text": w + " "})

    def send(self, msg):
        if "Alice" in msg and "Link" not in msg:
            self.files[0] = ("people/alice.txt", 1, 7)
            yield from self._call("write_file", {"name": "people/alice.txt", "text": ALICE},
                                  ["saved", "- people/alice.txt"], {"kind": "write", "i": 0, "v": 1})
            yield from self._say("I saved Alice in **people/alice.txt**.")
        elif "cafe" in msg:
            self.files[1] = ("projects/cafe.txt", 1, 6)
            yield from self._call("write_file", {"name": "projects/cafe.txt", "text": "A cafe."},
                                  ["saved", "- projects/cafe.txt"], {"kind": "write", "i": 1, "v": 1})
            self.links.add((0, 1))
            yield from self._call("link_files", {"name": "people/alice.txt",
                                                 "target": "projects/cafe.txt"},
                                  ["linked", "- people/alice.txt", "to", "- projects/cafe.txt"],
                                  None)
            yield from self._say("Done: Alice is linked to the cafe project.")
        elif "read" in msg.lower():
            yield from self._call("read_file", {"name": "people/alice.txt"}, [ALICE],
                                  {"kind": "read", "i": 0, "v": 1}, paged={"i": 0, "bytes": 21})
            yield from self._say(ALICE)
        elif "missing" in msg:
            yield from self._call("read_file", {"name": "notes/x.txt"}, ["not found"],
                                  {"kind": "none"})
            yield from self._say("There is no such file.")
        elif "slow" in msg:
            for _ in range(200):
                time.sleep(0.02)
                if self.cancel:
                    from qfs.bridge import Cancelled
                    self.where = "your files, kept after the interrupt"
                    raise Cancelled
                yield from self._say("word")
        else:
            yield from self._say("Hello!")
        yield Event("done", {"tokens": 10, "seconds": 0.1})


async def wait_idle(pilot, app, limit=5.0):
    t0 = time.time()
    await pilot.pause(0.1)
    while app.busy and time.time() - t0 < limit:
        await pilot.pause(0.05)
    await pilot.pause(0.2)


def text_of(app):
    out = []
    for w in app.query("Static"):
        r = w.render()
        out.append(r.plain if hasattr(r, "plain") else str(r))
    return "\n".join(out)


async def run(shot=None):
    tui.Session = FakeSession
    tmp = Path(tempfile.mkdtemp(prefix="qfs-tui-"))
    app = tui.QFS("fake", None)
    checks = []

    def check(name, ok):
        checks.append((name, bool(ok)))
        print(f"  {'ok ' if ok else 'FAIL'} {name}")

    async with app.run_test(size=(150, 46)) as pilot:
        await wait_idle(pilot, app)
        fs = app.query_one(tui.FSView)
        check("loaded: the prompt is enabled", not app.query_one("#prompt").disabled)
        check("welcome names the checkpoint", "8 layers in front of Qwen's 24" in text_of(app))
        app.send(tui.EXAMPLES[0])
        await wait_idle(pilot, app)
        tools = list(app.query(tui.Tool))
        check("a write renders as one tool line", len(tools) == 1 and tools[0].label == "write_file")
        check("its response is under it", "saved" in text_of(app) and "file 0 · version 1" in text_of(app))
        check("the panel shows the file", "alice.txt" in fs.draw_tree().plain
              and "people/" in fs.draw_tree().plain and "people/alice.txt"
              in fs.draw_table().plain)
        check("Qwen's answer renders", any(a.shown.startswith("I saved Alice") for a in app.query(tui.Answer)))
        check("no draft is left", not app.query(tui.Draft))
        app.send("Save projects/cafe.txt and Link Alice to it")
        await wait_idle(pilot, app)
        check("a link sets its bit in the panel", fs.state["links"] == {(0, 1)}
              and "1 set" in fs.draw_links().plain)
        check("the copy layer shows file rows", " file " in fs.draw_copy().plain)
        check("the hint counts files and links", "2 files" in str(app.query_one("#hint").render())
              and "1 link" in str(app.query_one("#hint").render()))
        app.send("missing file")
        await wait_idle(pilot, app)
        check("not found marks the call red", list(app.query(tui.Tool))[-1].state == "err")
        app.send("/help")
        await pilot.pause(0.2)
        check("/help lists the commands", "/mount" in text_of(app))
        app.send(f"/save {tmp}")
        await pilot.pause(0.2)
        check("/save stores it", app.session.saved == tmp and "saved" in fs.draw_disk().plain)
        app.send(f"/mount {tmp}")
        await pilot.pause(0.3)
        check("/mount starts a new chat on it", app.session.where == str(tmp)
              and app.query(".welcome") and "mounted" in fs.draw_disk().plain)
        app.send(f"/mount {tmp}/nothing")
        await pilot.pause(0.2)
        check("/mount of a missing image says so", "No filesystem" in text_of(app))
        app.send("read alice")
        await wait_idle(pilot, app)
        check("a paged read says so", "paged in from disk: 21 B" in text_of(app))
        app.send("/doc off")
        await pilot.pause(0.2)
        check("/doc off starts a new chat", app.session.doc is False and app.query(".welcome"))
        # the command menu
        await pilot.press("/", "d")
        await pilot.pause(0.1)
        check("the menu opens on /d", app.menu == ["/doc"])
        await pilot.press("escape")
        await pilot.pause(0.1)
        # an interrupt
        app.send("slow")
        await pilot.pause(0.3)
        await pilot.press("escape")
        await wait_idle(pilot, app)
        check("escape interrupts, the files kept", not app.busy and "kept" in text_of(app))
        # a narrow window hides the panel
        await pilot.resize_terminal(80, 40)
        await pilot.pause(0.2)
        check("a narrow window hides the panel", app.query_one("#side").has_class("hidden"))
        await pilot.resize_terminal(150, 46)
        await pilot.pause(0.2)
        if shot:
            app.send("/new")
            await pilot.pause(0.2)
            FakeSession.delay = 0.0
            for m in (tui.EXAMPLES[0], "Save projects/cafe.txt and Link Alice to it", "read alice"):
                app.send(m)
                await wait_idle(pilot, app)
            app.save_screenshot(shot)
            print(f"  screenshot: {shot}")
    shutil.rmtree(tmp, ignore_errors=True)
    bad = [n for n, ok in checks if not ok]
    print(f"{len(checks) - len(bad)}/{len(checks)} checks passed")
    return not bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screenshot")
    a = ap.parse_args()
    raise SystemExit(0 if asyncio.run(run(a.screenshot)) else 1)


if __name__ == "__main__":
    main()
