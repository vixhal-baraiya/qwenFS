"""Merge the filesystem into a stock Qwen3.5 checkpoint.

The FS layers (qfs.circuit) go in front of Qwen's 24, the residual is widened from
2048 to 3072 (Qwen's matrices are zero-padded; its decoder and final norm gains are scaled by
sqrt(2048/3072), so its text computation is unchanged up to bf16 rounding of those gains), and
the FS tokens are appended to the vocabulary. The FS exit clears every FS dim except the output
code, which it leaves EPS-sized in the id dims: Qwen's layers barely see it, and the final norm
scales it up so that a forced token (an ack, a read-back) wins over Qwen's own prediction. Where
nothing is forced the code is 0 and Qwen's logits are untouched.

Run: python -m qfs.merge SRC DST
"""
import argparse
import json
import math
import re
import shutil
from pathlib import Path

import torch
from safetensors import safe_open
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

from qfs import circuit as F
from qfs import words

PREFIX = "model.language_model."
FS_LAYERS = F.LAYERS
WIDE = math.sqrt(2048 / F.H)        # Qwen's norm gains on the wide residual
ALIAS = 0.25                        # ack tokens' Qwen part: a quarter of the newline's


def text_config(src):
    tc = json.loads((Path(src) / "config.json").read_text())["text_config"]
    tc.pop("model_type", None)
    tc["tie_word_embeddings"] = True
    return tc


def qwen_tensors(src):
    """Qwen's text weights, keyed as in Qwen3_5ForCausalLM ("model.layers.N...")."""
    src = Path(src)
    index = json.loads((src / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for f in sorted(set(index.values())):
        with safe_open(str(src / f), "pt") as sf:
            for k in sf.keys():
                if k.startswith(PREFIX):
                    out["model." + k[len(PREFIX):]] = sf.get_tensor(k)
    return out


def empty_model(cfg):
    with torch.device("meta"):
        model = Qwen3_5ForCausalLM(cfg)
    model = model.to_empty(device="cpu").to(torch.bfloat16)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    return model


def base_model(src):
    """Stock Qwen3.5 as a text-only model (the reference for the KL check)."""
    cfg = Qwen3_5TextConfig(**text_config(src))
    model = empty_model(cfg)
    model.load_state_dict(qwen_tensors(src), strict=False)
    model.tie_weights()
    return model.eval()


def merge(src, xs, gain):
    tc = text_config(src)
    n0, d0 = tc["num_hidden_layers"], tc["hidden_size"]
    assert d0 == F.OFF, d0
    tc.update(hidden_size=F.H, num_hidden_layers=len(FS_LAYERS) + n0, vocab_size=F.VOCAB,
              layer_types=FS_LAYERS + tc["layer_types"])
    tc.pop("full_attention_interval", None)
    cfg = Qwen3_5TextConfig(**tc)
    model = empty_model(cfg)
    params = dict(model.named_parameters())
    shift = re.compile(r"^model\.layers\.(\d+)\.")
    norm_re = re.compile(r"(input_layernorm|post_attention_layernorm)\.weight$|^model\.norm\.weight$")
    with torch.no_grad():
        for k, t in qwen_tensors(src).items():
            k2 = shift.sub(lambda m: f"model.layers.{int(m[1]) + len(FS_LAYERS)}.", k)
            p = params[k2]
            if norm_re.search(k2):                 # decoder and final norms see the wide residual
                p.fill_(-1)
                p[:d0] = ((1 + t.float()) * WIDE - 1).to(p.dtype)
            elif p.shape == t.shape:
                p.copy_(t)
            else:
                p[tuple(slice(0, s) for s in t.shape)] = t
        E = model.model.embed_tokens.weight
        F.embeddings(E, words.table(words.tokenizer(src), F.VOCAB))
        # ack tokens look like a newline to Qwen's layers (RMSNorm undoes the scale); scaled
        # down, their Qwen logit stays below the newline's, so Qwen never writes one itself
        E[F.ACKW0:F.NONE + 1, :d0] = (ALIAS * E[F.NL, :d0].float()).to(E.dtype)
        # list markers look like "-" (a list bullet), scaled down for the same reason
        E[F.MARK0:F.MARK0 + F.NFILE, :d0] = (ALIAS * E[F.DASH, :d0].float()).to(E.dtype)
        # search skips look like a newline, as acks do
        E[F.SKIP0:F.SKIP0 + F.NFILE, :d0] = (ALIAS * E[F.NL, :d0].float()).to(E.dtype)
        F.fill(model.model.layers[:len(FS_LAYERS)], xs)
        F.code_gain(model.model.norm, gain)
    model.tie_weights()
    return model.eval()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--gain", type=float, default=4096.0, help="the output code's final-norm gain")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "mps"
    xs = F.calibrate(dev)
    model = merge(a.src, xs, a.gain)
    dst = Path(a.dst)
    model.save_pretrained(dst, max_shard_size="2GB")
    for f in ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja"):
        if (Path(a.src) / f).exists():
            shutil.copy(Path(a.src) / f, dst / f)
    (dst / "fs.json").write_text(json.dumps({
        "base": Path(a.src).name, "fs_layers": len(FS_LAYERS), "fs_dims_offset": F.OFF,
        "hidden": F.H, "eps": F.EPS, "code_gain": a.gain, "vocab": F.VOCAB}, indent=2))
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
