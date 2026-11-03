model := "Qwen3.5-2B"
merged := "build/fs-qwen-2b"

# list the recipes
default:
    @just --list

# install the dependencies
setup:
    uv sync

# download Qwen3.5-2B and compile the filesystem into it
build:
    uv run hf download Qwen/{{model}} --local-dir models/{{model}}
    uv run python -m qfs.merge models/{{model}} {{merged}}

# chat with Qwen and its filesystem (e.g. just chat --mount fs)
chat *args:
    uv run qfs {{merged}} {{args}}

# send messages without the TUI (e.g. just ask "Save notes/todo.txt: call the dentist")
ask *args:
    uv run python -m qfs.chat {{merged}} {{args}}

# the circuit alone on scripted tool-call sessions (no Qwen needed)
test *args:
    uv run python -m tests.test_circuit {{args}}

# the filesystem saved to disk and read back in a fresh session
test-disk *args:
    uv run python -m tests.test_disk {{args}}

# the chat app, headless, on a scripted session (no model needed)
test-tui *args:
    uv run python -m tests.test_tui {{args}}

# the merged model: Qwen's text unchanged, the filesystem exact
check:
    uv run python -m tests.test_merged {{merged}} models/{{model}}

# how well Qwen uses the filesystem: scenario chats, with and without the doc
eval *args:
    uv run python -m qfs.evaluate {{merged}} {{args}}
