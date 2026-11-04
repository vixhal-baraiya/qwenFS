"""qfs chat: talk to Qwen, which keeps its files in the filesystem inside its own weights.

Your message sits in a box on the right, Qwen's text follows a dot, and each tool call is one
"● write_file(people/alice.txt)" line with the circuit's response indented under "⎿". Settings
are slash commands typed in the input (/help lists them). The panel on the right is the
filesystem as the model holds it, read from its state after every response (qfs/monitor.py):
its files and folders, its links, the FS layers lighting up as they act, and the disk.

The filesystem persists on a disk (fs/ by default): the app starts on it and saves to it after
every turn that changes it.

Run: uv run qfs [MERGED] [--device mps] [--disk DIR | --no-disk] [--no-doc]
"""

import time
from collections import deque
from pathlib import Path
from typing import ClassVar

from rich.cells import cell_len
from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from rich.theme import Theme
from textual import events, on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Input, Static

from qfs import circuit as F
from qfs.bridge import Cancelled, Event, Session, stored
from qfs.monitor import LAYERS, Monitor

EXAMPLES = [
    "Remember my friend Alice, a designer in Paris. Save her in the people folder.",
    "Save my shopping list at lists/shopping.txt: milk, eggs.",
    "Add bread to my shopping list.",
    "Save projects/cafe.txt: a new cafe opening in March. Link Alice's file to it.",
    "Who works on the cafe project? Check which files link to it.",
]
TOOLS = [  # (tool, what it does)
    ("write_file", "save a file (folders: people/alice.txt)"),
    ("read_file", "its text, token for token"),
    ("list_files", "every path"),
    ("link_files", "link a file to another; unlink_files removes it"),
    ("list_links", "a file's links; list_backlinks: the files linking to it"),
]
DISK = "fs"  # where the filesystem persists by default
COMMANDS = [  # (command, arguments, what it does)
    ("/save", "dir", "store a copy of the filesystem in dir"),
    ("/mount", "dir", "a new chat on the filesystem stored in dir, now the disk"),
    ("/new", "", "a new chat on your files"),
    ("/format", "", "erase the disk: a new chat on an empty filesystem"),
    ("/doc", "on|off", "the filesystem reference as Qwen's system prompt"),
    ("/panel", "", "show or hide the filesystem panel"),
    ("/help", "", "commands and keys"),
    ("/quit", "", "exit"),
]
OPTIONS = {"/doc": ("on", "off")}  # a command's choices, picked with left/right in the menu
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
STAR = "·✢✳✶✻✽✻✶✳✢"  # the status line's pulse
ACCENT, OK, ERR = "#d7875f", "#87af87", "#d77575"  # muted: one warm accent, a calm green/red
SPARK = " ▁▂▃▄▅▆▇█"
FPS = 12
NARROW = 100  # columns: below this the panel hides itself
SAMPLE = 6  # frames per throughput sample (half a second)

# the panel's lights: each part has a dark base and a lit colour
LIT, STAGE, QWEN = (240, 190, 90), (230, 140, 90), (110, 160, 240)
FILE_ON, REG, OLD = (120, 200, 140), (90, 160, 115), (60, 92, 76)
LINK, MEM, CTX = (90, 140, 220), (110, 160, 240), (58, 64, 74)
OFF, DIM, ERR_RGB = (44, 48, 54), (110, 116, 126), (215, 117, 117)
BLOCK_OFF, TILE_OFF, GRID, GRID_USED = (40, 44, 52), (30, 33, 39), (26, 29, 34), (40, 45, 54)
HUES = [(120, 200, 140), (110, 160, 230), (200, 150, 230), (90, 190, 200), (230, 200, 110),
        (230, 140, 150)]  # a folder's colour
ROOT = (180, 186, 196)
CTX_KEY = (78, 86, 100)  # the legend's context swatch, a shade lighter than the rows


def hard_breaks(text: str) -> str:
    """Qwen's line breaks kept: Markdown joins single lines into one paragraph, so each line
    outside a code block ends in a hard break."""
    out, code = [], False
    for ln in text.split("\n"):
        code ^= ln.lstrip().startswith("```")
        out.append(ln if code or ln.lstrip().startswith("```") or not ln.strip() else ln + "  ")
    return "\n".join(out)


def short_text(s: str, n: int) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[:max(n - 1, 0)] + "…"


def light(h: float, on: tuple, base: tuple) -> str:
    """A light's color: `base` when dark, `on` at full brightness."""
    h = max(0.0, min(1.0, h))
    r, g, b = (int(x + (y - x) * h) for x, y in zip(base, on))
    return f"rgb({r},{g},{b})"


def spark(vals, width: int) -> str:
    vals = list(vals)[-width:]
    top = max(max(vals, default=0), 1e-9)
    s = "".join(SPARK[min(8, round(v / top * 8))] if v > 0 else SPARK[1] for v in vals)
    return s.rjust(width, SPARK[1])


def size(n: int) -> str:
    return f"{n:,} B" if n < 10_000 else f"{n / 1024:.1f} KB"


def call_arg(func: str, params: dict) -> str:
    """A call as its one line shows it: the path, or both paths of a link."""
    if "target" in params:
        return f"{params.get('name', '?').strip()} → {params['target'].strip()}"
    for k in ("name", "file", "path"):
        if k in params:
            return params[k].strip()
    return ""


# --- chat widgets -----------------------------------------------------------------------------


class UserMsg(Horizontal):
    """Your message: a box with round corners on the right, as in a chat app."""

    def __init__(self, msg: str, reply: Text | None = None) -> None:
        super().__init__()
        self.msg, self.reply = msg, reply  # reply: a slash command's outcome, under it

    def compose(self) -> ComposeResult:
        t = Text(self.msg)
        if self.reply is not None:
            t.append("\n")
            t.append_text(self.reply)
        yield Static(t, classes="bubble")


class Notice(Static):
    pass


class Example(Static):
    """A clickable example prompt."""

    def __init__(self, prompt: str) -> None:
        super().__init__(Text.assemble(("  › ", "dim"), prompt))
        self.prompt = prompt

    def on_click(self, _: events.Click) -> None:
        self.app.send(self.prompt)


class Holder(Vertical):
    """A container that takes widgets before it is mounted: they wait for it (events can
    arrive faster than Textual mounts)."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.waiting: list = []
        self.ready = False  # on_mount has run: mount directly (is_mounted is set later)

    def add(self, *ws) -> None:
        if self.ready:
            self.mount(*ws)
        else:
            self.waiting += ws

    def on_mount(self) -> None:
        self.ready = True
        if self.waiting:
            self.mount(*self.waiting)
            self.waiting = []


class Answer(Horizontal):
    """Qwen's text: a dot, then Markdown (a tool call it is writing is the Draft's)."""

    def __init__(self) -> None:
        super().__init__(classes="answer")
        self.text, self.shown, self.body = "", "", Static("", classes="body")

    def compose(self) -> ComposeResult:
        yield Static("●", classes="dot")
        yield self.body

    def write(self, piece: str) -> None:
        self.text += piece
        self.shown = self.text.split("<tool_call>")[0].strip()
        self.body.update(RichMarkdown(hard_breaks(self.shown)) if self.shown else "")
        self.display = bool(self.shown)


class Draft(Vertical):
    """A tool call while Qwen is still writing it: the tool's name, then each parameter as it
    streams in. The call event replaces it with a Tool."""

    def __init__(self) -> None:
        super().__init__(classes="tool")
        self.head, self.body = Static("", classes="head"), Static("", classes="source")
        self.call = ""  # the call's text so far, after <tool_call>

    def compose(self) -> ComposeResult:
        yield self.head
        yield self.body

    def draw(self, spin: str) -> None:
        import re

        call = re.sub(r"\n?<[^>]*$", "", self.call)  # a tag still being written: not text
        name = re.search(r"<function=([^>\s]+)>", call)
        params = re.findall(r"<parameter=([^>\s]+)>\n?(.*?)(?:\n?</parameter>|$)", call,
                            re.DOTALL)
        self.head.update(Text.assemble((f"{spin} ", ACCENT), (name.group(1) if name else "tool",
                                                                "bold"),
                                       ("  writing the call…", "dim")))
        self.body.update(Text("\n".join(f"{k}: {v.strip()}" for k, v in params), style="dim"))


class Tool(Holder):
    """One tool call, as a "● name(argument)" line, the text it writes, and the circuit's
    response under "⎿"."""

    def __init__(self, func: str, arg: str = "", text: str | None = None) -> None:
        super().__init__(classes="tool")
        self.label, self.arg, self.text = func, arg, text
        self.state = "run"  # run, ok, err
        self.head, self.n = Static(self._head(), classes="head"), 0

    def _head(self) -> Text:
        dot = {"run": "dim", "ok": OK, "err": ERR}[self.state]
        t = Text.assemble(("● ", dot), (self.label, "bold"))
        t.append(f"({self.arg})", style="dim" if self.state == "run" else "")
        return t

    def compose(self) -> ComposeResult:
        yield self.head
        if self.text:
            yield Static(Text(self.text.strip(), style="dim italic"), classes="source")

    def line(self, text) -> Static:
        """A line under the call: "⎿" before the first, aligned spaces after."""
        t = indented(text if isinstance(text, Text) else Text(str(text)), self.n == 0)
        self.n += 1
        w = Static(t, classes="line")
        self.add(w)
        return w

    def set(self, state: str) -> None:
        self.state = state
        self.head.update(self._head())


def indented(t: Text, first: bool = True) -> Text:
    """Text under a call: "  ⎿  " (or spaces) before its first line, spaces before the rest."""
    out = Text()
    for i, line in enumerate(t.split("\n", allow_blank=True)):
        out.append("\n" if i else "")
        out.append("  ⎿  " if first and i == 0 else "     ", style="dim")
        out.append_text(line)
    return out


# --- the filesystem panel ---------------------------------------------------------------------

class FSView(Vertical):
    """The filesystem as the model holds it, drawn as the circuit's own parts: a status card, the
    FS layers as a pipeline, the file table (the version table's 32 rows), the link bits (layer
    6's table), the copy layer's rows, the folders, and the disk."""

    def __init__(self) -> None:
        super().__init__(id="fs")
        self.mon = Monitor()
        self.state = {"files": {}, "links": set(), "rows": 0, "lookup": 0, "mounted": None,
                      "vbits": [[0] * F.VB for _ in range(F.NFILE)], "copy": []}
        self.meta: dict = {}
        self.model = ""
        self.doing = ""  # what the circuit is doing, for the status card
        self.fed = self.fed_seen = 0
        self.rates: deque = deque([0.0] * 34, maxlen=34)  # tok/s, one per SAMPLE frames
        self.touched: dict[int, float] = {}  # file -> its light
        self.linked: dict[tuple, float] = {}  # a new link -> its light
        self.unlinked: dict[tuple, float] = {}  # a removed link, fading out
        self.current: int | None = None  # the file the response is about
        self.head = 0.0  # the copy layer's write head: lit as each token adds a row
        self.last: int | None = None  # the file touched last
        self.disk: dict = {}  # the last save or mount
        self.n = 0
        self.cards = {k: Static(classes="card", id=f"f-{k}") for k in
                      ("top", "pipe", "table", "links", "copy", "tree", "disk")}

    def compose(self) -> ComposeResult:
        yield from self.cards.values()

    def on_mount(self) -> None:
        c = self.cards
        c["pipe"].border_title, c["table"].border_title = "layers", "files"
        c["links"].border_title, c["copy"].border_title = "links", "copy layer"
        c["tree"].border_title, c["disk"].border_title = "folders", "disk"
        c["table"].border_subtitle, c["links"].border_subtitle = "layer 4", "layer 6"

    def feed(self, d: dict) -> None:
        t, phase = d["t"], d["phase"]
        self.mon.feed(t, phase, d["func"])
        self.fed += 1
        self.state["rows"] += 1  # each token is a copy-layer row (the next fs event is exact)
        self.head = 1.0
        if phase != "response":
            self.current = None
        elif F.ACKW0 <= t < F.NONE:
            x = t - (F.ACKW0 if t < F.ACKR0 else F.ACKR0)
            self.current = x >> F.VB
            self.touch(self.current)
        elif F.MARK0 <= t < F.MARK0 + F.NFILE:
            self.touch(t - F.MARK0)
        elif self.current is not None and d["func"] == "read_file":
            self.touch(self.current)  # the text being copied

    def update_fs(self, st: dict) -> None:
        for pair in st["links"] - self.state["links"]:
            self.linked[pair] = 1.0
        for pair in self.state["links"] - st["links"]:
            self.unlinked[pair] = 1.0
        self.state = st

    def touch(self, i: int) -> None:
        self.touched[i] = 1.0
        self.last = i

    def frame(self) -> None:
        self.n += 1
        if self.n % SAMPLE == 0:
            self.rates.append((self.fed - self.fed_seen) * FPS / SAMPLE)
            self.fed_seen = self.fed
        c = self.cards
        c["top"].border_title = "qfs"
        c["top"].border_subtitle = self.model
        room = self.parent.size.height if self.parent else 0
        sized = room > 0  # before the first layout every card shows
        w = self.width_of(c["table"])

        def lines(x: Text) -> int:  # a card's height: its lines as wrapped, and its border
            return sum(max(1, -(-cell_len(ln) // w)) for ln in x.plain.split("\n")) + 2

        draw = {"top": self.draw_top, "pipe": self.draw_pipe, "table": self.draw_table,
                "links": self.draw_links, "copy": self.draw_copy, "tree": self.draw_tree,
                "disk": self.draw_disk}
        for k, f in draw.items():  # in order: a card that does not fit is not shown
            text = f()
            c[k].display = text is not None and (not sized or lines(text) <= room)
            if c[k].display:
                room -= lines(text)
                c[k].update(text)
        self.head *= Monitor.FADE
        self.mon.tick()
        fade = Monitor.FADE

        def dim(d: dict) -> dict:
            return {k: v * fade for k, v in d.items() if v > 0.02}
        self.touched, self.linked, self.unlinked = (dim(self.touched), dim(self.linked),
                                                   dim(self.unlinked))

    @staticmethod
    def width_of(card: Static) -> int:
        return card.content_size.width or 46

    def path(self, i: int) -> str:
        f = self.state["files"].get(i)
        return f[0] if f else f"slot {i}"

    def draw_top(self) -> Text:
        t, w = Text(), self.width_of(self.cards["top"])
        if self.doing:
            t.append("● ", style=ACCENT)
            t.append(short_text(self.doing, w - 2), style="bold")
        else:
            t.append("○ idle", style="dim")
        t.append("\n")
        t.append(spark(self.rates, w - 11), style=ACCENT if self.doing else "dim")
        t.append(f" {self.rates[-1]:4.0f} tok/s", style="bold" if self.doing else "dim")
        return t

    def draw_pipe(self) -> Text:
        """The FS layers as a strip of blocks, lit as they act on a token, then Qwen's."""
        t, w = Text(), self.width_of(self.cards["pipe"])
        lights = self.mon.lights
        bw = max(3, min(6, (w - 9) // 8 - 1))  # a block's width
        q = self.meta.get("qwen_layers", 24)
        qw = max(4, w - 8 * (bw + 1) - 2)
        t.append("".join(f"{name:<{bw + 1}}" for name, _, _ in LAYERS), style="dim")
        label = next(s for s in (f"Qwen ×{q}", f"Qwen×{q}", "Qwen", "Q") if len(s) <= qw)
        t.append(f"  {label:<{qw}}", style="dim")
        for _ in range(2):
            t.append("\n")
            for k in range(len(LAYERS)):
                t.append(" " * bw, style=f"on {light(lights[k], STAGE, BLOCK_OFF)}")
                t.append(" ")
            t.append("› ", style="dim")
            t.append(" " * qw, style=f"on {light(lights[-1], QWEN, BLOCK_OFF)}")
        names = ["lags", "parse", "path", "lookup", "tables", "forced", "links", "copy"]
        lit = [names[k] for k in range(len(LAYERS)) if lights[k] > 0.5]
        t.append("\n" + (short_text(" · ".join(lit), w) if lit else " "),
                 style=ACCENT if lit else "dim")
        return t

    def tile(self, i: int) -> tuple[str, str]:
        """A slot's colours: its folder's hue, lit when touched; dark when free."""
        f = self.state["files"].get(i)
        if f is None:
            return light(0, OFF, TILE_OFF), "rgb(78,84,94)"
        folder = f[0].rsplit("/", 1)[0] if "/" in f[0] else ""
        hue = HUES[sum(map(ord, folder)) % len(HUES)] if folder else ROOT
        return light(self.touched.get(i, 0.0), LIT, hue), "rgb(18,20,24)"

    def draw_table(self) -> Text:
        """The 32 file slots of the version table as tiles: a file's folder colour, its slot
        and its version; a slot lights as the circuit acks or lists it."""
        t, w = Text(), self.width_of(self.cards["table"])
        files = self.state["files"]
        tw = max(3, min(6, (w - 7) // 8))
        out = []
        for r in range(4):
            for line in range(2):
                row = Text()
                for c in range(8):
                    i = 8 * r + c
                    bg, fg = self.tile(i)
                    label = f"{i}" if line == 0 else (f"v{files[i][1]}" if i in files else "")
                    row.append(f"{label:^{tw}}", style=f"{fg} on {bg}")
                    row.append(" " if c < 7 else "")
                out.append(row)
            if r < 3:
                out.append(Text(""))
        t = Text("\n").join(out)
        i = self.last if self.last in files else (min(files) if files else None)
        t.append("\n")
        if i is None:
            t.append(" ", style="dim")
        else:
            path, v, n = files[i]
            t.append(short_text(path, w - 14), style="bold")
            t.append(f"  v{v} · {n} tok", style="dim")
        return t

    def draw_links(self) -> Text:
        """Layer 6's link bits as a map: a row per source slot, a column per target."""
        t, w = Text(), self.width_of(self.cards["links"])
        links, files = self.state["links"], self.state["files"]
        n = min(F.NFILE, max(8, -(-max(len(files), 1) // 8) * 8))

        def cell(s: int, d: int) -> str:
            if (s, d) in self.unlinked:
                return light(self.unlinked[(s, d)], ERR_RGB, GRID)
            if (s, d) in links:
                return light(self.linked.get((s, d), 0.0), LIT, LINK)
            used = s in files and d in files
            return light(0, GRID_USED if used else GRID, GRID_USED if used else GRID)

        gap = 3 * n + 12 <= w  # cells with a gap between them, if they fit
        rows = range(n) if n <= 16 else range(0, n, 2)
        for k, s in enumerate(rows):
            t.append("\n" if k else "")
            for d in range(n):
                if n <= 16:  # an upper half block: a square cell, with a gap below it
                    t.append("▀▀", style=cell(s, d))
                    t.append(" " if gap else "")
                else:  # two sources per line: the upper half's colour over the lower's
                    t.append("▀", style=f"{cell(s, d)} on {cell(s + 1, d)}")
            if k == 0:
                t.append("  source ↓", style="dim")
            elif k == 1:
                t.append("  target →", style="dim")
            elif k == 3 and links:
                t.append(f"  {len(links)} set", style="dim")
        return t

    def draw_copy(self) -> Text:
        """The copy layer's rows as a memory map: context, file text (in its folder's colour),
        older versions, rows paged in from disk; a read lights its file's rows, and the write
        head lights as each token adds a row."""
        t, w = Text(), self.width_of(self.cards["copy"])
        rows, segs = self.state["rows"], self.state.get("copy", [])
        per, lines = w, 4
        cells = per * lines
        size_ = max(1, -(-rows // cells))
        rank = {"paged": 3, "file": 2, "old": 1}
        owner: list = [None] * cells
        for a, b, kind, i in segs:
            for c in range(a // size_, min(cells, -(-b // size_))):
                if owner[c] is None or rank[kind] > rank[owner[c][0]]:
                    owner[c] = (kind, i)
        used = -(-rows // size_) if rows else 0
        for c in range(cells):
            t.append("\n" if c and c % per == 0 else "")
            if c >= used:
                col = light(0, TILE_OFF, TILE_OFF)
            elif c == used - 1:
                col = light(self.head, LIT, CTX)
            elif owner[c] is None:
                col = light(0, CTX, CTX)
            else:
                kind, i = owner[c]
                if kind == "file":
                    col = self.tile(i)[0]
                else:
                    col = light(self.touched.get(i, 0.0), LIT, MEM if kind == "paged" else OLD)
            t.append(" ", style=f"on {col}")
        t.append("\n")
        for label, c in (("context", CTX_KEY), ("file", HUES[0]), ("old", OLD), ("disk", MEM)):
            t.append("  ", style=f"on {light(0, c, c)}")
            t.append(f" {label}  ", style="dim")
        self.cards["copy"].border_subtitle = f"layer 7 · {rows:,} rows"
        return t

    def draw_tree(self) -> Text:
        t, w = Text(), self.width_of(self.cards["tree"])
        files = self.state["files"]
        if not files:
            t.append("no files yet", style="dim")
            return t
        tree: dict = {}
        for i, (path, v, n) in files.items():
            *folders, leaf = path.split("/")
            node = tree
            for f in folders:
                node = node.setdefault(f + "/", {})
            node[leaf] = (i, v, n)

        def walk(node: dict, depth: int) -> None:
            dirs = sorted(k for k, x in node.items() if isinstance(x, dict))
            leaves = sorted(k for k, x in node.items() if not isinstance(x, dict))
            for k in dirs:
                t.append(("\n" if t.plain else "") + "  " * depth + k, style="bold")
                walk(node[k], depth + 1)
            for k in leaves:
                i, v, n = node[k]
                h = self.touched.get(i, 0.0)
                meta = f"slot {i:<2} v{v}  " + (f"{n} tok" if n else "empty")
                name = short_text(k, max(6, w - 2 * depth - len(meta) - 2))
                t.append(("\n" if t.plain else "") + "  " * depth)
                t.append(name, style=light(h, LIT, (205, 210, 215)))
                t.append(" " * max(1, w - 2 * depth - cell_len(name) - len(meta)))
                t.append(meta, style=light(h, FILE_ON, DIM))

        walk(tree, 0)
        return t

    def draw_disk(self) -> Text | None:
        t, d, st = Text(), self.disk, self.state
        if not st.get("mounted") and not d.get("saved"):
            return None
        if st.get("mounted"):
            t.append("mounted ", style="dim")
            t.append(st["mounted"], style="bold")
            if d.get("paged"):
                t.append(f"\npaged in {d['paged']} file{'s' * (d['paged'] != 1)} · "
                         f"{size(d['paged_bytes'])}", style="dim")
            else:
                t.append("\nfiles page in when Qwen reads them", style="dim")
        if d.get("saved"):
            s = d["saved"]
            t.append("\n" if t.plain else "")
            t.append(f"saved {s['path']}: {s['files']} file{'s' * (s['files'] != 1)} · "
                     f"{s['links']} link{'s' * (s['links'] != 1)} · {size(s['bytes'])}",
                     style="dim")
        return t


# --- the input --------------------------------------------------------------------------------

class Prompt(Input):
    """The message box. While the command menu is open: up/down choose a command, left/right
    its option, enter runs it, tab completes it, escape closes the menu."""

    BINDINGS: ClassVar = [
        Binding("tab", "complete", show=False),
        Binding("up", "menu(-1)", show=False), Binding("down", "menu(1)", show=False),
        Binding("left", "option(-1)", show=False), Binding("right", "option(1)", show=False),
        Binding("escape", "close", show=False),
    ]

    def action_complete(self) -> None:
        if self.app.menu:
            self.value = self.app.menu[self.app.menu_i] + " "
            self.cursor_position = len(self.value)

    def action_menu(self, step: int) -> None:
        if self.app.menu:
            self.app.menu_i = (self.app.menu_i + step) % len(self.app.menu)
            self.app.draw_menu()

    def action_option(self, step: int) -> None:
        cmd = self.app.menu[self.app.menu_i] if self.app.menu else None
        if cmd in OPTIONS:
            self.app.menu_opt[cmd] = (self.app.menu_opt[cmd] + step) % len(OPTIONS[cmd])
            self.app.draw_menu()
        elif step < 0:  # no menu: the arrows move the cursor as usual
            self.action_cursor_left()
        else:
            self.action_cursor_right()

    def action_close(self) -> None:
        if self.app.menu:
            self.value = ""
        else:
            self.app.action_stop()


# --- the app ----------------------------------------------------------------------------------

class QFS(App):
    TITLE = "qfs"
    CSS = """
    Screen { background: $background; }
    #main { height: 1fr; }
    #chat, #side { scrollbar-size-vertical: 1; scrollbar-background: $background;
                   scrollbar-color: $panel-lighten-1; scrollbar-color-hover: $panel-lighten-2;
                   scrollbar-color-active: $panel-lighten-3; }
    #chat { width: 1fr; padding: 0 1; }
    #side { width: 34%; min-width: 46; max-width: 84; padding: 0 0 0 1; overflow-y: hidden; }
    #side.hidden { display: none; }
    .card { border: round $panel-lighten-2; border-title-color: $text-muted;
            border-subtitle-color: $text-muted; padding: 0 1; height: auto; }
    .welcome { padding: 0 1; margin: 0 0 1 0; border: round $panel-lighten-2; height: auto; }
    .welcome Static { height: auto; }
    .row { height: auto; }
    .row .row-name { width: 16; padding: 0 0 0 2; text-style: bold; }
    .row Static { width: 1fr; }
    Example:hover { background: $boost; }
    UserMsg { margin: 1 0 0 0; height: auto; align-horizontal: right; }
    UserMsg .bubble { width: auto; max-width: 75%; height: auto; padding: 0 1;
                      border: round $panel-lighten-3; background: transparent; }
    .answer { height: auto; margin: 1 0 0 0; }
    .answer .dot { width: 2; }
    .answer .body { width: 1fr; height: auto; }
    .tool { height: auto; margin: 1 0 0 0; }
    .tool Static { height: auto; }
    .tool .source { margin: 0 0 0 5; }
    Notice { color: $text-muted; margin: 1 0 0 0; }
    #status { height: 1; padding: 0 1; display: none; }
    #status.on { display: block; }
    #prompt { margin: 0; border: round $panel-lighten-2; }
    #prompt:focus { border: round $panel-lighten-3; }
    #menu { height: auto; padding: 0 2; display: none; }
    #menu.on { display: block; }
    #hint { height: 1; padding: 0 2; color: $text-muted; }
    """
    BINDINGS: ClassVar = [
        Binding("escape", "stop", "Stop", show=False),
        Binding("ctrl+b", "toggle_side", "Panel", show=False),
        Binding("ctrl+l", "redraw", "Redraw", show=False),
        Binding("ctrl+q", "quit", "Quit", priority=True, show=False),
    ]

    def __init__(self, merged: str, device: str | None, doc: bool = True,
                 disk: str | None = DISK):
        super().__init__()
        self.merged, self.device, self.doc, self.disk = merged, device, doc, disk
        self.session: Session | None = None
        self.busy, self.t0 = True, time.time()
        self.status_text = "Loading the model"
        self.frame_no = 0
        self.tool = self.answer = self.draft = None
        self.side_shown: bool | None = None  # None: by the window's width
        self.menu: list[str] = []  # the slash commands the menu shows
        self.menu_i, self.menu_opt = 0, {}  # the highlighted one; each one's chosen option

    # layout ---------------------------------------------------------------------------------
    def compose(self) -> ComposeResult:
        with Horizontal(id="main"):
            with VerticalScroll(id="chat"):
                yield self._welcome()
            with VerticalScroll(id="side"):
                yield FSView()
        yield Static("", id="status")
        yield Prompt(placeholder="Loading the model…", id="prompt", disabled=True)
        yield Static("", id="menu")
        yield Static("", id="hint")

    def _about(self) -> str:
        m = self.session.meta if self.session else {}
        base, fs, qwen = m.get("base", "Qwen3.5"), m.get("fs_layers", 8), m.get("qwen_layers")
        own = f"Qwen's {qwen}" if qwen else "Qwen's own"
        return (f"{base} with a filesystem compiled into its weights: {fs} layers in front of "
                f"{own}, every weight computed, none learned.")

    def _welcome(self) -> Vertical:
        intro = Text()
        intro.append("✻ ", style=ACCENT)
        intro.append("qfs", style="bold")
        intro.append(f"  {self._about()}\n\n", style="dim")
        intro.append("Try", style="bold")
        tools = Text()
        tools.append("\nTools", style="bold")
        tools.append("  Qwen calls them; the circuit in its weights answers", style="dim")
        rows = [Horizontal(Static(name, classes="row-name"), Static(Text(what, style="dim")),
                           classes="row")
                for name, what in TOOLS]
        s = self.session
        mount = s.where if s else None
        disk = s.disk if s else None
        where = (f"on {mount}" if mount else
                 f"on an empty filesystem · it saves to {disk}/" if disk else
                 "on an empty filesystem, kept in memory only · /save stores it")
        foot = Static(Text(f"\nA new chat {where}\n/ for commands · esc stops · ctrl+b panel",
                           style="dim"))
        return Vertical(Static(intro), *[Example(e) for e in EXAMPLES], Static(tools), *rows,
                        foot, classes="welcome")

    def _hint(self, width: int | None = None) -> None:
        s, t = self.session, Text()
        if s is None:
            self.query_one("#hint", Static).update("loading…")
            return
        st, fs = s.stats, self.query_one(FSView).state
        n, k = len(fs["files"]), len(fs["links"])
        # (priority, text): the highest go first when the line is wider than the window
        parts = [(0, f"{n} file{'s' * (n != 1)}"), (2, f"{s.name} on {s.dev}"),
                 (1, f"{k} link{'s' * (k != 1)}" if k else ""),
                 (1, f"mounted {s.where}" if s.where else ""),
                 (3, "doc on" if s.doc else "doc off"),
                 (3, f"{st['calls']} call{'s' * (st['calls'] != 1)}" if st["calls"] else ""),
                 (0, "/ for commands")]
        parts = [(p, x) for p, x in parts if x]
        room = (width or self.size.width) - 4
        while len(" · ".join(x for _, x in parts)) > room and len(parts) > 2:
            parts.remove(max(parts, key=lambda px: px[0]))
        for i, (_, x) in enumerate(parts):
            t.append(" · " if i else "", style="dim")
            t.append(x, style=ACCENT if i == 0 else "")
        self.query_one("#hint", Static).update(t)

    def _say(self, text: str) -> None:
        self.status_text = text

    # loading --------------------------------------------------------------------------------
    def on_mount(self) -> None:
        # Rich's Markdown draws `code` cyan on a block: plain bold fits the muted palette
        self.console.push_theme(Theme({"markdown.code": "bold", "markdown.code_block": "",
                                       "markdown.link": "underline", "markdown.h1": "bold",
                                       "markdown.h2": "bold", "markdown.item.bullet": "dim"}))
        self._hint()
        self.query_one("#status").add_class("on")
        self.set_interval(1 / FPS, self._frame)
        self.load()

    def _frame(self) -> None:
        """FPS times a second: the panel, the status line, a call being written."""
        if not self.query("#status"):  # the app is closing: its widgets are gone
            return
        self.frame_no += 1
        spin = SPIN[self.frame_no % len(SPIN)]
        self.query_one(FSView).frame()
        status = Text()
        if self.busy:
            status.append(f"{STAR[self.frame_no // 2 % len(STAR)]} ", style=ACCENT)
            status.append(self.status_text + "…", style=ACCENT)
            status.append(f"  ({time.time() - self.t0:.0f}s · esc to interrupt)", style="dim")
        self.query_one("#status", Static).update(status)
        if self.draft is not None:
            self.draft.draw(spin)

    @work(thread=True, exclusive=True)
    def load(self) -> None:
        import transformers

        transformers.logging.set_verbosity_error()  # its warnings would draw over the screen
        try:
            s = Session(self.merged, self.device, doc=self.doc, disk=self.disk)
        except Exception as e:  # noqa: BLE001 - shown in the app
            self.call_from_thread(self._loaded, None, str(e))
            return
        self.call_from_thread(self._loaded, s, None)

    def _loaded(self, s: Session | None, error: str | None) -> None:
        self._busy(False)
        if error:
            self._add(Notice(Text(f"Could not load the model: {error}", style=ERR)))
            return
        self.session = s
        fs = self.query_one(FSView)
        fs.meta, fs.model = s.meta, f"{s.name} · {s.dev}"
        self._new_fs()
        for w in self.query(".welcome"):  # now it can say what the checkpoint is
            w.remove()
        self.query_one("#chat").mount(self._welcome())
        prompt = self.query_one("#prompt", Input)
        prompt.disabled = False
        prompt.placeholder = "Ask anything, or / for commands"
        prompt.focus()
        self._hint()

    def _new_fs(self) -> None:
        """The panel on the session's filesystem as it now is (a new chat, a mount)."""
        fs, s = self.query_one(FSView), self.session
        fs.state = s.state()
        fs.touched, fs.linked = {}, {}
        fs.disk = {k: v for k, v in fs.disk.items() if k == "saved"}
        fs.doing = ""

    def _busy(self, on: bool) -> None:
        self.busy = on
        self.query_one("#status").set_class(on, "on")

    # chat -----------------------------------------------------------------------------------
    def _add(self, *ws) -> None:
        self.query_one("#chat").mount(*ws)
        self.query_one("#chat").scroll_end(animate=False)

    @on(Input.Changed, "#prompt")
    def typing(self, ev: Input.Changed) -> None:
        """While a slash command is being typed, the commands that match it, under the box;
        each opens on its current setting."""
        word = ev.value.split(" ")[0]
        menu = [c for c, _, _ in COMMANDS if ev.value.startswith("/") and c.startswith(word)]
        if menu != self.menu:
            self.menu, self.menu_i = menu, 0
            self.menu_opt = {c: OPTIONS[c].index(self._setting(c)) for c in OPTIONS}
        self.draw_menu()

    def _setting(self, cmd: str) -> str:
        s = self.session
        return "on" if s is None or s.doc else "off"

    def draw_menu(self) -> None:
        box = self.query_one("#menu", Static)
        box.set_class(bool(self.menu), "on")
        t = Text()
        for i, (c, a, what) in enumerate(x for x in COMMANDS if x[0] in self.menu):
            on = i == self.menu_i
            t.append("\n" if i else "")
            t.append("› " if on else "  ", style=ACCENT)
            t.append(f"{c:<9}", style="bold" if on else "")
            opts = Text()
            for k, o in enumerate(OPTIONS.get(c, ())):
                chosen = k == self.menu_opt.get(c)
                opts.append(f" {o} ", style=f"bold reverse {ACCENT}" if chosen and on else
                            "bold" if chosen else "dim")
            if c not in OPTIONS:
                opts.append(a, style="dim")
            t.append_text(opts)
            t.append(" " * max(1, 16 - opts.cell_len))
            t.append(what, style="" if on else "dim")
        if self.menu:
            t.append("\n  ↑↓ choose · ←→ option · enter run · tab complete · esc close",
                     style="dim")
        box.update(t)

    @on(Input.Submitted, "#prompt")
    def submit(self, ev: Input.Submitted) -> None:
        msg = ev.value.strip()
        if self.menu:  # the highlighted command, with the typed argument or the chosen option
            cmd = self.menu[self.menu_i]
            arg = msg.partition(" ")[2].strip() or (OPTIONS[cmd][self.menu_opt[cmd]]
                                                     if cmd in OPTIONS else "")
            msg = f"{cmd} {arg}".strip()
        if self.send(msg):
            ev.input.value = ""

    def send(self, msg: str) -> bool:
        msg = msg.strip()
        if not msg or self.session is None:
            return False
        if msg.startswith("/"):
            return self.command(msg)
        if self.busy:
            self.notify("Still working: press escape to stop.", severity="warning")
            return False
        for w in self.query(".welcome"):
            w.remove()
        self._add(UserMsg(msg))
        self.tool = self.answer = self.draft = None
        self._busy(True)
        self.t0 = time.time()
        self._say("Reading")
        self.run_chat(msg)
        return True

    def command(self, msg: str) -> bool:
        """A slash command: shown as your line, its outcome under it."""
        name, _, arg = msg.partition(" ")
        arg, s = arg.strip(), self.session

        def reply(text: str, err: bool = False) -> None:
            self._add(UserMsg(msg, Text(text, style=ERR if err else "dim")))

        hits = [c for c, _, _ in COMMANDS if c.startswith(name)]
        name = hits[0] if len(hits) == 1 else name
        if self.busy and name in ("/save", "/mount", "/new", "/format", "/doc"):
            reply("Still working: press escape first.", True)
        elif name == "/save":
            path = Path(arg) if arg else s.disk or Path(DISK)
            try:
                r = s.save(path)
            except OSError as e:
                reply(f"Could not save: {e}", True)
            else:
                fs = self.query_one(FSView)
                fs.disk["saved"] = r
                reply(f"Saved {r['files']} file{'s' * (r['files'] != 1)} and {r['links']} "
                      f"link{'s' * (r['links'] != 1)} to {path}/ "
                      f"({size(r['bytes'])}).")
        elif name == "/mount":
            path = Path(arg) if arg else s.disk or Path(DISK)
            if not stored(path):
                reply(f"No filesystem in {path}/: /save stores one.", True)
            else:
                s.disk = s.disk and path  # the disk follows the mount; no disk stays none
                self.action_new_chat(mount=path)
                fs = self.query_one(FSView)
                n = len(fs.state["files"])
                reply(f"A new chat on {path}/: {n} file{'s' * (n != 1)}, "
                      f"{len(fs.state['links'])} link{'s' * (len(fs.state['links']) != 1)}. "
                      "Files page in when Qwen reads them.")
        elif name == "/doc":
            v = {"on": True, "off": False, "": not s.doc}.get(arg.lower())
            if v is None:
                reply("Usage: /doc on|off", True)
            else:
                s.doc = v
                self.action_new_chat()
                reply(f"The filesystem reference: {'on' if v else 'off'}. A new chat"
                      + (f" on {s.mounted}/." if s.mounted else "."))
        elif name == "/new":
            self.action_new_chat()
        elif name == "/format":
            s.erase()
            self.action_new_chat(mount=None)
            reply("Erased" + (f" {s.disk}/" if s.disk else " the filesystem")
                  + ": a new chat on an empty filesystem.")
        elif name == "/panel":
            self.action_toggle_side()
        elif name == "/help":
            lines = [f"{c:<9}{a:<9}{w}" for c, a, w in COMMANDS]
            lines += ["", ("esc stop · tab complete a command · ctrl+b panel · ctrl+l redraw · "
                           "ctrl+q quit")]
            reply("\n".join(lines))
        elif name in ("/quit", "/exit"):
            self.exit()
        else:
            reply(f"Unknown command {name}. /help lists them.", True)
        self._hint()
        return True

    @work(thread=True, exclusive=True, group="chat")
    def run_chat(self, msg: str) -> None:
        try:
            for e in self.session.send(msg):
                self.call_from_thread(self.show, e)
        except Cancelled:
            self.call_from_thread(self._stopped)
        except Exception as e:  # noqa: BLE001 - shown in the app
            self.call_from_thread(self._failed, repr(e))

    def show(self, e: Event) -> None:
        d = e.data
        fs = self.query_one(FSView)
        if e.kind == "token":
            fs.feed(d)
            return
        if e.kind == "text":
            if self.answer is None:
                self.answer = Answer()
                self._add(self.answer)
            self.answer.write(d["text"])
            if "<tool_call>" in self.answer.text:  # Qwen is writing a call: show it as it goes
                if self.draft is None:
                    self.draft = Draft()
                    self._add(self.draft)
                self.draft.call = self.answer.text.split("<tool_call>", 1)[1]
                self._say("Writing a tool call")
            else:
                self._say("Writing")
        elif e.kind == "call":
            if self.answer is not None and not self.answer.display:
                self.answer.remove()  # the turn was only the call
            if self.draft is not None:
                self.draft.remove()  # the call, written: the Tool below shows it
            self.answer = self.draft = None
            func, params = d["func"], d["params"]
            self.tool = Tool(func, call_arg(func, params), params.get("text"))
            self._add(self.tool)
            arg = call_arg(func, params)
            fs.doing = {"write_file": f"saving {arg}", "read_file": f"reading {arg}",
                        "list_files": "listing the files", "link_files": f"linking {arg}",
                        "unlink_files": f"unlinking {arg}", "list_links": f"links of {arg}",
                        "list_backlinks": f"backlinks of {arg}"}.get(func, func)
            self._say("The filesystem is answering")
        elif e.kind == "response":
            ack, lines = d["ack"], d["lines"]
            missing = ack is not None and ack["kind"] == "none"
            body = "\n".join(lines) if lines else "(empty)"
            self.tool.line(Text(body, style=ERR if missing else ""))
            note = []
            if ack and ack["kind"] in ("write", "read"):
                note.append(f"file {ack['i']} · version {ack['v']}")
                fs.touch(ack["i"])
            if d["paged"]:
                p = d["paged"]
                note.append(f"paged in from disk: {size(p['bytes'])}")
                fs.disk["paged"] = fs.disk.get("paged", 0) + 1
                fs.disk["paged_bytes"] = fs.disk.get("paged_bytes", 0) + p["bytes"]
            if note:
                self.tool.line(Text(" · ".join(note), style="dim"))
            self.tool.set("err" if missing else "ok")
            fs.doing = ""
            self._say("Reading the response")
        elif e.kind == "fs":
            fs.update_fs(d)
            self._hint()
        elif e.kind == "saved":
            fs.disk["saved"] = d
        elif e.kind == "cut":
            self._add(Notice(Text("The turn reached the token limit.", style=ERR)))
        elif e.kind == "done":
            self._busy(False)
            fs.doing = ""
            self._hint()
            self.query_one("#prompt", Input).focus()
        self.query_one("#chat").scroll_end(animate=False)

    def _stopped(self) -> None:
        self._busy(False)
        self._add(Notice("Interrupted. A new chat on your files: the context was cleared, "
                         "the filesystem kept."))
        fs = self.query_one(FSView)
        fs.state = self.session.state()
        fs.doing = ""
        self._hint()

    def _failed(self, err: str) -> None:
        self._busy(False)
        self._add(Notice(Text(f"Error: {err}. Started a new chat on the last saved files.",
                              style=ERR)))
        self.session.reset(self._home())
        self._new_fs()
        self._hint()

    def _home(self) -> Path | None:
        """The image a new chat starts on: the disk's, else the one mounted (if any)."""
        s = self.session
        return s.disk if stored(s.disk) else s.mounted

    # actions --------------------------------------------------------------------------------
    def action_new_chat(self, mount: Path | None | bool = True) -> None:
        """A new chat on `mount`: by default, on your files."""
        if self.session is None or self.busy:
            return
        self.session.reset(self._home() if mount is True else mount)
        self.tool = self.answer = self.draft = None
        self._new_fs()
        chat = self.query_one("#chat")
        chat.remove_children()
        chat.mount(self._welcome())
        self._hint()

    def action_stop(self) -> None:
        if self.session and self.busy:
            self.session.cancel = True

    def action_toggle_side(self) -> None:
        """/panel: show or hide the panel, overriding the automatic choice."""
        self.side_shown = self.query_one("#side").has_class("hidden")
        self._place_side()

    def on_resize(self, ev: events.Resize) -> None:
        self._place_side(ev.size.width)
        self._hint(ev.size.width)
        # Some terminals reflow their cells after the resize reaches the app, leaving pieces of
        # the old layout where nothing changed for the app to repaint: clear and draw it all.
        self.set_timer(0.15, self.action_redraw)
        self.set_timer(0.6, self.action_redraw)

    def action_redraw(self) -> None:
        """Clear the terminal and draw every cell again (ctrl+l)."""
        if self._driver is not None:
            self._driver.write("\x1b[2J")
        self.screen.refresh()

    def _place_side(self, width: int | None = None) -> None:
        """The panel shows when the window is wide enough for it and the chat (or when
        /panel asked for it); a narrow window gives the chat the whole width."""
        width = width or self.size.width
        shown = width >= NARROW if self.side_shown is None else self.side_shown
        self.query_one("#side").set_class(not shown, "hidden")


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser(description="Chat with Qwen; it keeps its files in the "
                                             "filesystem inside its weights")
    ap.add_argument("merged", nargs="?", default="build/fs-qwen-2b")
    ap.add_argument("--device", help="cuda, mps or cpu (default: the fastest available)")
    ap.add_argument("--disk", default=DISK,
                    help=f"where the filesystem persists (default {DISK}/): the app starts on it "
                         "and saves to it after every turn that changes it")
    ap.add_argument("--no-disk", action="store_true",
                    help="keep the filesystem in memory only (/save still stores a copy)")
    ap.add_argument("--no-doc", action="store_true",
                    help="without the filesystem reference as the system prompt")
    a = ap.parse_args()
    # Loading the model creates a multiprocessing semaphore, which starts Python's resource
    # tracker; started under Textual (stdio replaced) it fails with "bad value(s) in
    # fds_to_keep". Start it while stdio is still the terminal.
    from multiprocessing import resource_tracker

    resource_tracker.ensure_running()
    QFS(a.merged, a.device, not a.no_doc, None if a.no_disk else a.disk).run()


if __name__ == "__main__":
    main()
