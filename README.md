# qwenFS: Qwen keeps its own files

**qwenFS is an experimental model: Qwen3.5-2B with a filesystem compiled and inserted directly
into its weights.** It is not a plugin, a tool server or a database behind an API. The
filesystem is 8 transformer layers in front of Qwen's own 24, and none of their weights are
learned: every one is set by construction. Qwen saves, reads, organizes and links its own files
through its normal tool calls, and the model answers every call itself.

![qwenFS saving a file and reading it back: the FS layers, the file table and the copy layer, token by token](docs/qfs-demo.gif)

## How it works

**The calls.** Qwen writes ordinary Qwen3.5 tool calls. The circuit reads each call as Qwen
writes it, keying on the single tokens Qwen's tool-call format always uses (`=name`, `=text`,
`=target`, `>`, newline, `</tool_call>`, …). A call's kind comes from its parameters; only
`unlink_files` is told apart by its function-name token. After `</tool_call>`, the circuit
prints the tool response itself, in a user turn as Qwen's chat template has it, then opens a new
assistant turn, and Qwen answers.

| Tool | What it does | The circuit's response |
|---|---|---|
| `write_file(name, text)` | save a file (replaces it) | `saved` and the path |
| `read_file(name)` | read a file, token for token | the stored text, or `not found` |
| `list_files()` | every file | one path per line, or `empty` |
| `link_files(name, target)` | link a file to another | `linked` and both paths |
| `unlink_files(name, target)` | remove that link | `removed` and both paths |
| `list_links(file)` | files a file links to | one path per line, or `empty` |
| `list_backlinks(path)` | files that link to a file | one path per line, or `empty` |
| `search_files(query)` | files whose text has a word | one path per line, or `empty` |

Paths make folders (`people/alice.txt`, `projects/cafe.txt`); links and backlinks make a graph.

**Where the filesystem lives.** Every part of the model has a job:

- **Attention** holds the files. A write's text stays in the copy layer's attention log, keyed by
  (file index, version, offset); a read is an exact copy, token by token. A second attention head
  looks paths up: a path finds its file's index, or falls back to "not found". A third
  searches: every word of a written text is a row, keyed by (file index, version, the word's
  tokens), and a search checks one file per response token, against that file's current
  version, so a word only an old version had is not found.
- **Gated DeltaNet state** holds the tables and registers: the version table, the path table
  (for listings), the file count, the call's state, and the links. A link is a row of state per
  (source, target) pair: `link_files` writes it, `unlink_files` empties it, so links never
  duplicate. Rows store their bit in the target's column and in the source's, and a multi-hot
  query over a file's rows reads all its links, or all its backlinks, at once.
- **MLPs** parse the tool calls and pick every forced token: acks, the response frame, echoed
  paths, the next link to list, a search's next match. A file a search passes over gets a skip
  token, which Qwen reads as a newline.
- **Offsets** come from a decaying DeltaNet counter, read in the head norm's eps regime and
  decoded exactly to 600 tokens.

**Exactness.** Every token carries a ballast dim, so each norm in the FS layers is a fixed scale.
Registers pad their state rows to 2, 8, 32 or 128 set rails, so reads come back as exactly 0/1
or ±1. A key that is off sits far below zero, since a DeltaNet key's l2-norm would rescale any
small value to full strength. `qfs/circuit.py` describes each layer.

**The merge.** `qfs.merge` widens Qwen's residual from 2048 to 3584, zero-pads its matrices,
scales its norm gains by √(2048/3584), puts the FS layers in front of Qwen's and appends the FS
tokens to the vocabulary. The FS layers clear their dims before Qwen's layers run, except a small
output code that the tied lm_head reads at the end, so a forced token wins and Qwen's logits are
untouched everywhere else.

**Disk.** A filesystem saves to a compact image: the paths, the latest text of each file as
3-byte token ids, and the links (about 3.3–3.8 bytes per stored token). A fresh session mounts
the image (the tables, the lookup rows and the search rows, one per distinct word of each file)
and pages a file's text in only when the model emits that file's read ack. Paging rebuilds the copy rows from the token ids with the model's own
projections. See `qfs/disk.py`.

## What runs where

- **The weights**: parsing each call, every lookup, every table update, every forced token,
  every copied token of a read. Nothing outside the model decides what a response says.
- **The chat loop** (`qfs/chat.py`): decodes tokens and stops at `<|im_end|>`. It samples at
  temperature 0.7: with greedy decoding, a 2B model tends to repeat tool calls. Forced tokens lead
  by hundreds of logits, so sampling never changes them.
- **Disk paging** (`qfs/disk.py`): when the model emits a read ack, the runtime loads that
  file's bytes, like a disk controller. The model picks the file and copies the text.
- **The app's panel** (`qfs/monitor.py`): it reads the filesystem out of the model's state (the
  version, path and link tables) and keeps no copy of its own.

## The doc

`docs/fs-agent.md` is Qwen's system prompt by default: a reference for the tools, their
responses and conventions, with worked examples. Every example response in it was printed by the
merged model (`qfs/gen_doc.py`). In a scenario eval (save and recall, updating a list,
organizing, links and backlinks, unlinking one file, a missing file; 3 seeds, 48 turns):

| | Turns passed | Scenarios passed | Unneeded calls |
|---|---|---|---|
| No doc | 46/48 | 17/18 | 6 |
| With the doc | 48/48 | 18/18 | 0 |

The sample is small; the drop in unneeded calls is the clearer effect.

## Results

On Qwen3.5-2B, on Apple silicon (MPS, bf16):

- **Scripted sessions** of writes, overwrites, deletes, missing names, links, repeated links,
  unlinks and listings (16 files): every forced token exact, 74/74 calls per session.
- **A dense graph**: one file linked to and from all 31 others, a duplicate and a self-link, then
  unlinks at the start, middle and end: exact.
- **From disk**, in a fresh session with an empty context: exact.
- **Merged into Qwen3.5-2B**, in context and from disk: exact.
- **Qwen's text**: the top-1 prediction is unchanged at all 139 positions checked; KL about 2e-3
  (from rounding the rescaled norm gains to bf16).

## Limits

- 32 files, paths of up to 8 tokens, about 590 tokens per write, 63 versions per file; a text
  cannot contain `</`.
- Search finds one whole word, exactly and case-sensitively (`Paris`, not `paris`), of up to 3
  tokens as Qwen writes the word alone; longer words are not found.
- No partial edits: changing a file means writing it again.
- Qwen's judgment is the weak part, not the filesystem: it sometimes answers from the
  conversation instead of reading a file, or makes calls it wasn't asked for.
- About 20 tokens/s on MPS (PyTorch fallback kernels for Gated DeltaNet).

## Quickstart

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/),
[just](https://github.com/casey/just), and about 16 GB of free memory for the merged model
(Apple silicon, an NVIDIA GPU or a CPU).

```bash
git clone https://github.com/vixhal-baraiya/qwenFS && cd qwenFS
just setup
just build
just chat
```

`just build` downloads Qwen3.5-2B and compiles the filesystem into it. `just chat` opens the
chat app: each tool call shows as one line with the circuit's response under it, and the panel
on the right is the filesystem as the model holds it, decoded from its DeltaNet state after
every response: the files and folders, the links, the FS layers lighting up as they act, and
the disk. The filesystem persists in `fs/`: the app starts on it, and saves to it after every
turn that changes it, so your files are there in the next chat. A stored file pages in only when
Qwen reads it. `/new` starts a new chat on your files, `/format` erases them, `--disk DIR` keeps
them elsewhere and `--no-disk` keeps them in memory only; `/help` lists the rest.

`just ask "Save notes/todo.txt: call the dentist" "What's on my todo list?"` sends messages
without the app. `just` lists every recipe: `test` (the circuit alone, no Qwen needed),
`test-disk`, `test-tui` (the app on a scripted session), `check` (the merged model) and `eval`
(add `--doc docs/fs-agent.md` to use the doc).

## Repository

| Path | What it is |
|---|---|
| `qfs/circuit.py` | The filesystem as Qwen3.5 layers: every weight, layer by layer |
| `qfs/merge.py` | Building the merged Qwen3.5 checkpoint |
| `qfs/disk.py` | The disk image: saving, mounting and paging files in |
| `qfs/tui.py` | The chat app: the chat, and the filesystem panel |
| `qfs/bridge.py` | The app's session: a turn as a stream of events |
| `qfs/monitor.py` | The filesystem decoded from the model's state; the panel's lights |
| `qfs/chat.py` | The chat loop, and a command-line chat |
| `qfs/evaluate.py` | The scenario eval |
| `qfs/gen_doc.py` | Writes `docs/fs-agent.md`, running its examples on the model |
| `docs/fs-agent.md` | The filesystem reference Qwen gets as its system prompt |
| `tests/` | The circuit on scripted sessions, the disk image, the merged model |

## Related

[qwenGPU](https://github.com/vixhal-baraiya/qwenGPU) compiles a SIMT GPU into the same model.
qwenFS shares no code with it: its layers are built directly in Qwen3.5's own layer types.

## License

The code is MIT, see [LICENSE](LICENSE). A checkpoint built with `qfs.merge` is a modified
[Qwen3.5](https://huggingface.co/Qwen/Qwen3.5-2B) model (its config is widened and filesystem
layers are added; Qwen's own weights are unchanged apart from rescaled norm gains), so its
weights fall under Qwen's Apache-2.0 license.
