"""Checks of the merged checkpoint (qfs.merge).

  text   on ordinary text (no FS tokens), the merged model's next-token distributions against
         stock Qwen3.5: KL and top-1 agreement
  fs     a scripted session (tests.test_circuit) inside the merged model: every forced token must win
         over Qwen's own prediction, in prefill and in greedy generation

Usage: python -m tests.test_merged MERGED SRC [--files 8] [--maxlen 200] [--skip-text]
"""
import argparse
import time

import torch
from transformers import AutoTokenizer
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from qfs import merge as M
from tests import test_circuit as tc

TEXTS = [
    "The quick brown fox jumps over the lazy dog. In 1969, Apollo 11 landed on the Moon, and "
    "Neil Armstrong became the first person to walk on its surface.",
    "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    "
    "return a\n\nprint([fibonacci(i) for i in range(10)])",
    "<|im_start|>user\nWhat is the capital of France, and why is it famous?<|im_end|>\n"
    "<|im_start|>assistant\n",
    "Photosynthesis converts light energy into chemical energy. Plants take in carbon dioxide "
    "and water, and with sunlight they produce glucose and oxygen.",
]


@torch.no_grad()
def text_check(merged, src, dev):
    tok = AutoTokenizer.from_pretrained(src)
    base = M.base_model(src).to(dev)
    kls, agree, n = [], 0, 0
    for t in TEXTS:
        ids = tok(t, return_tensors="pt").input_ids.to(dev)
        p = base(input_ids=ids).logits[0].float().log_softmax(-1)
        q = merged(input_ids=ids).logits[0, :, :p.shape[-1]].float().log_softmax(-1)
        kls.append((p.exp() * (p - q)).sum(-1).cpu())
        agree += (p.argmax(-1) == q.argmax(-1)).sum().item()
        n += ids.shape[1]
    kl = torch.cat(kls)
    del base
    print(f"text: {n} positions, KL(base || merged) mean {kl.mean():.2e} max {kl.max():.2e}; "
          f"top-1 agrees {agree}/{n}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("merged")
    ap.add_argument("src")
    ap.add_argument("--files", type=int, default=8)
    ap.add_argument("--maxlen", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-text", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "mps"
    merged = Qwen3_5ForCausalLM.from_pretrained(a.merged, dtype=torch.bfloat16).to(dev).eval()
    if not a.skip_text:
        text_check(merged, a.src, dev)
    s = tc.make(a.seed, a.files, a.maxlen)
    print(f"fs session: {len(s.toks)} tokens, {len(s.reads)} reads, {len(s.forced)} forced tokens")
    good, total, bad = tc.check_prefill(merged, s, dev)
    print(f"  prefill:  {good}/{total} forced tokens right" + (f", first wrong at {bad}" if bad else ""))
    t0 = time.time()
    ok, n = tc.check_generate(merged, s, dev)
    print(f"  generate: {ok}/{n} reads exact ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
