"""Does Qwen use the filesystem well? Scenario chats on the merged model, scored per turn.

Each scenario is a fresh conversation (and an empty filesystem). A turn passes when Qwen made
the calls the turn needs, none it must not (linking unasked, unlinking the wrong file), its
answer has the facts asked for, and it finished within the token limit. Calls beyond the needed
ones are counted as extra.

Usage: python -m qfs.evaluate MERGED [--doc docs/fs-agent.md] [--seeds 0,1,2] [--out result.json]
       python -m qfs.evaluate --rescore result.json ...   (score saved replies again)
"""
import argparse
import json
import re
import time

import torch

from qfs.chat import Chat


def calls_of(text):
    out = []
    for blk in re.findall(r"<tool_call>(.*?)</tool_call>", text, re.S):
        f = re.search(r"<function=(\w+)>", blk)
        params = dict(re.findall(r"<parameter=(\w+)>\n(.*?)\n</parameter>", blk, re.S))
        out.append((f.group(1) if f else "?", params))
    return out


def path(p, *parts):
    """A call parameter is a path naming all of `parts` (case-insensitive)."""
    return lambda c: p in c[1] and all(x in c[1][p].lower() for x in parts)


def call(func, *preds):
    return lambda c: c[0] == func and all(pr(c) for pr in preds)


def text_has(*words):
    return lambda c: all(w in c[1].get("text", "").lower() for w in words)


W, R, L, U = "write_file", "read_file", "link_files", "unlink_files"
LF, LL, LB = "list_files", "list_links", "list_backlinks"
NO_LINKS = [call(L), call(U)]

SCENARIOS = {
    "recall": [
        ("Save a note at notes/dentist.txt: dentist appointment on Friday at 3pm.",
         [call(W, path("name", "dentist"))], NO_LINKS, [], []),
        ("When is my dentist appointment? Read the note.",
         [call(R, path("name", "dentist"))], NO_LINKS + [call(W)], ["friday", "3"], []),
    ],
    "update": [
        ("Save my shopping list at lists/shopping.txt: milk, eggs.",
         [call(W, path("name", "shopping"))], NO_LINKS, [], []),
        ("Add bread to my shopping list.",
         [call(W, path("name", "shopping"), text_has("milk", "eggs", "bread"))], NO_LINKS, [], []),
        ("What is on my shopping list? Read the file.",
         [call(R, path("name", "shopping"))], NO_LINKS + [call(W)], ["milk", "eggs", "bread"], []),
    ],
    "organize": [
        ("Remember my friend Alice, a designer in Paris. Save her in the people folder.",
         [call(W, path("name", "people/", "alice"))], NO_LINKS, [], []),
        ("Also remember Bob, a chef in Tokyo, in the people folder.",
         [call(W, path("name", "people/", "bob"))], NO_LINKS, [], []),
        ("What files do I have saved?",
         [call(LF)], NO_LINKS + [call(W)], ["alice", "bob"], []),
    ],
    "links": [
        ("Save people/alice.txt: Alice is a designer. Then save projects/cafe.txt: a new cafe "
         "opening in March.",
         [call(W, path("name", "alice")), call(W, path("name", "cafe"))], NO_LINKS, [], []),
        ("Alice works on the cafe project. Link her file to it.",
         [call(L, path("name", "alice"), path("target", "cafe"))],
         [call(U), call(L, lambda c: "alice" not in c[1].get("name", "").lower())], [], []),
        ("Who works on the cafe project? Check which files link to it.",
         [call(LB, path("path", "cafe"))], NO_LINKS + [call(W)], ["alice"], []),
    ],
    "unlink": [
        ("Save people/alice.txt: Alice, designer. Save people/bob.txt: Bob, chef. Save "
         "projects/cafe.txt: a new cafe.",
         [call(W, path("name", "alice")), call(W, path("name", "bob")),
          call(W, path("name", "cafe"))], NO_LINKS, [], []),
        ("Alice and Bob both work on the cafe project. Link both of them to it.",
         [call(L, path("name", "alice"), path("target", "cafe")),
          call(L, path("name", "bob"), path("target", "cafe"))], [call(U)], [], []),
        ("Bob left the project. Remove only his link.",
         [call(U, path("name", "bob"), path("target", "cafe"))],
         [call(U, path("name", "alice")), call(L)], [], []),
        ("Who works on the cafe project now? Check the links.",
         [call(LB, path("path", "cafe"))], NO_LINKS + [call(W)], ["alice"], []),
    ],
    "missing": [
        ("What is in notes/recipes.txt? Read it.",
         [call(R, path("name", "recipes"))], NO_LINKS + [call(W)], [], []),
    ],
}
NOT_FOUND = ("not", "no ", "doesn't", "does not", "couldn't", "unable", "empty")


def score_turn(reply, need, forbid, words, absent, scenario):
    calls = calls_of(reply)
    final = reply.rsplit("</think>", 1)[-1].lower() if "</think>" in reply else reply.lower()
    if "<tool_call>" in final:
        final = final.rsplit("</tool_call>", 1)[-1]
    needed = all(any(n(c) for c in calls) for n in need)
    bad = [c for c in calls if any(f(c) for f in forbid)]
    said = all(w in final for w in words) and not any(w in final for w in absent)
    if scenario == "missing":
        said = any(w in final for w in NOT_FOUND)
    cut = "[cut at the token limit]" in reply
    extra = sum(1 for c in calls if not any(n(c) for n in need))
    return {"pass": needed and not bad and said and not cut, "needed": needed, "forbidden": len(bad),
            "said": said, "cut": cut, "extra": extra, "calls": [c[0] for c in calls]}


def rescore(files):
    for f in files:
        rows = json.load(open(f))
        for r in rows:
            msg, need, forbid, words, absent = SCENARIOS[r["scenario"]][r["turn"]]
            r.update(score_turn(r["reply"], need, forbid, words, absent, r["scenario"]))
        scen = {}
        for r in rows:
            scen.setdefault((r["seed"], r["scenario"]), []).append(r["pass"])
        print(f"{f}: turns {sum(r['pass'] for r in rows)}/{len(rows)}; scenarios "
              f"{sum(all(v) for v in scen.values())}/{len(scen)}; extra calls "
              f"{sum(r['extra'] for r in rows)}; forbidden {sum(r['forbidden'] for r in rows)}; "
              f"cut {sum(r['cut'] for r in rows)}; tokens {sum(r['tokens'] for r in rows)}")
        for r in rows:
            if not r["pass"]:
                print(f"   fail: seed {r['seed']} {r['scenario']} turn {r['turn']} "
                      f"(needed {r['needed']}, forbidden {r['forbidden']}, said {r['said']}, "
                      f"cut {r['cut']}) {r['calls']}")


def main():
    import sys
    if sys.argv[1:2] == ["--rescore"]:
        return rescore(sys.argv[2:])
    ap = argparse.ArgumentParser()
    ap.add_argument("merged")
    ap.add_argument("--doc")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--out")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "mps"
    system = open(a.doc).read() if a.doc else None
    chat = Chat(a.merged, dev, temp=a.temp, system=system)
    rows = []
    t0 = time.time()
    for seed in map(int, a.seeds.split(",")):
        for name, turns in SCENARIOS.items():
            chat.reset(seed, system)
            for k, (msg, need, forbid, words, absent) in enumerate(turns):
                reply, n, dt = chat.turn(msg)
                r = score_turn(reply, need, forbid, words, absent, name)
                r.update(seed=seed, scenario=name, turn=k, tokens=n, reply=reply)
                rows.append(r)
                print(f"seed {seed} {name:9s} turn {k}: {'PASS' if r['pass'] else 'fail'} "
                      f"(needed {r['needed']}, forbidden {r['forbidden']}, said {r['said']}, "
                      f"cut {r['cut']}, extra {r['extra']}) {r['calls']}", flush=True)
    passed = sum(r["pass"] for r in rows)
    scen = {}
    for r in rows:
        scen.setdefault((r["seed"], r["scenario"]), []).append(r["pass"])
    print(f"\n{'doc' if a.doc else 'no doc'}: turns {passed}/{len(rows)} passed; scenarios "
          f"{sum(all(v) for v in scen.values())}/{len(scen)} fully passed; extra calls "
          f"{sum(r['extra'] for r in rows)}; forbidden calls {sum(r['forbidden'] for r in rows)}; "
          f"cut turns {sum(r['cut'] for r in rows)} ({time.time() - t0:.0f}s)")
    if a.out:
        json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
