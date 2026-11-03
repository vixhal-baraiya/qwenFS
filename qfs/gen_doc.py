"""Write docs/fs-agent.md: the filesystem reference Qwen gets in its system prompt.

The worked examples are one session run on the merged model: the example calls are fed in as
Qwen's own tool calls, and each response shown is the one the circuit printed (acks are blank
to Qwen, markers read as "-"). Calls are written as write_file(name="...") rather than tool-call
syntax, which the circuit would run.

Usage: python -m qfs.gen_doc MERGED [--out docs/fs-agent.md]
"""
import argparse
from pathlib import Path

import torch

from qfs import circuit as F
from qfs.chat import TOOLS, Chat

DOC = """# Your files

You have a small filesystem: your exact memory. A file keeps its text exactly; read_file returns
it word for word. Your memory of the conversation can be wrong or out of date. The file is the
truth, so to answer about something saved, read it first.

## Tools

| Tool | Use it to | The response |
|---|---|---|
| write_file: name, text | save a file; it replaces the whole file | saved, then the path |
| read_file: name | get a file's text | the text, or "not found" |
| list_files (no parameters) | see every file | one path per line, or "empty" |
| link_files: name, target | link a file to a related file | linked, the two paths |
| unlink_files: name, target | remove that one link | removed, the two paths |
| list_links: file | see the files a file links to | one path per line, or "empty" |
| list_backlinks: path | see the files that link to a file | one path per line, or "empty" |

A missing file gives "not found".

## How to use them

- Name files folder/name.txt, in lowercase: people/sam.txt, projects/website.txt,
  lists/shopping.txt, notes/ideas.txt. Keep paths short.
- Make one call per action. When the response says saved, linked or removed, it is done:
  do not repeat the call. Then answer the user.
- Do what the user asked and nothing more: link files only when asked.
- To change a file, read it, then write its complete new text.
- To remove one link, unlink exactly that pair; the other links stay.
- Limits: 32 files, about 500 words per file.

## Examples

These ran on your filesystem; each response is what it printed. Write each call in the
usual tool-call format: the examples only list the function and its parameters.

{examples}"""

EXAMPLES = [
    ("Save my shopping list: milk, eggs.",
     [("write_file", {"name": "lists/shopping.txt", "text": "milk\neggs"})],
     "I saved your shopping list to lists/shopping.txt."),
    ("Add bread to my shopping list.",
     [("read_file", {"name": "lists/shopping.txt"}),
      ("write_file", {"name": "lists/shopping.txt", "text": "milk\neggs\nbread"})],
     "I added bread. Your list is now milk, eggs and bread."),
    ("What's on my shopping list?",
     [("read_file", {"name": "lists/shopping.txt"})],
     "Your shopping list has milk, eggs and bread."),
    ("Remember my colleague Sam, an engineer in Berlin.",
     [("write_file", {"name": "people/sam.txt", "text": "Sam is an engineer in Berlin."})],
     "I saved Sam in people/sam.txt."),
    ("Save a project note: the website launch is on May 5.",
     [("write_file", {"name": "projects/website.txt", "text": "The website launch is on May 5."})],
     "I saved the note to projects/website.txt."),
    ("Sam works on the website project. Link them.",
     [("link_files", {"name": "people/sam.txt", "target": "projects/website.txt"})],
     "I linked Sam to the website project."),
    ("Who works on the website project?",
     [("list_backlinks", {"path": "projects/website.txt"})],
     "Sam works on the website project."),
    ("Sam moved to another project. Remove his link.",
     [("unlink_files", {"name": "people/sam.txt", "target": "projects/website.txt"})],
     "I removed Sam's link to the website project."),
    ("What files do I have?",
     [("list_files", {})],
     "You have lists/shopping.txt, people/sam.txt and projects/website.txt."),
    ("What's in notes/ideas.txt?",
     [("read_file", {"name": "notes/ideas.txt"})],
     "There is no file notes/ideas.txt."),
]


def call_text(func, params):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in params.items())
    return f"<tool_call>\n<function={func}>\n{body}</function>\n</tool_call>"


def shown(func, params):
    """A call as the doc shows it: the function, then a "param: value" line each (a compact
    call notation leaked into Qwen's real calls as <function=name(...)>)."""
    lines = [func] + [f"  {k}: " + v.replace("\n", "\n" + " " * (len(k) + 4)) for k, v in params.items()]
    return "\n".join(lines)


def render(tok, ids):
    """Response tokens as Qwen reads them: acks blank, markers as "-"."""
    out = []
    for t in ids:
        if F.ACKW0 <= t <= F.NONE:
            continue
        out.append("-" if F.MARK0 <= t < F.MARK0 + F.NFILE else tok.decode([t]))
    return "".join(out).strip("\n")


@torch.no_grad()
def run(chat):
    tok = chat.tok
    parts = []
    for user, calls, answer in EXAMPLES:
        if not chat.ids:
            prompt = tok.apply_chat_template([{"role": "user", "content": user}],
                                             tools=TOOLS, tokenize=False,
                                             add_generation_prompt=True, enable_thinking=False)
        else:
            prompt = (f"\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
                      "<think>\n\n</think>\n\n")
        chat.feed(tok(prompt).input_ids)
        lines = [f"User: {user}"]
        for func, params in calls:
            nxt = chat.feed(tok(call_text(func, params)).input_ids)
            got = []
            while True:                  # the circuit's response, up to the new turn's header
                got.append(nxt)
                nxt = chat.feed([nxt])
                if got[-2:] == [F.THINKEND, F.NL2] or len(got) > 400:
                    break
            body = got[got.index(F.TRESP) + 1:got.index(F.TRESPEND)]
            lines += [f"Tool call: {shown(func, params)}", "Response:", render(tok, body)]
        chat.feed(tok(answer + "<|im_end|>").input_ids)
        lines.append(f"Answer: {answer}")
        parts.append("\n".join(lines))
    return parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merged")
    ap.add_argument("--out", default="docs/fs-agent.md")
    a = ap.parse_args()
    chat = Chat(a.merged, "cuda" if torch.cuda.is_available() else "mps", temp=0.0)
    parts = run(chat)
    Path(a.out).parent.mkdir(exist_ok=True)
    Path(a.out).write_text(DOC.format(examples="```\n" + "\n\n".join(parts) + "\n```\n"))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
