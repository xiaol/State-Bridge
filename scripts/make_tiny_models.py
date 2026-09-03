"""Create two tiny, randomly initialised Qwen3-architecture models with a shared byte-level
BPE tokenizer and a minimal chat template.  Used by the smoke config and the tests so
that the whole pipeline can run offline on CPU in under a minute."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>{{ m['content'] }}<|end|>{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


def make_tokenizer(texts: list[str], vocab_size: int = 512) -> PreTrainedTokenizerFast:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>", "<pad>", "<|end|>", "<|user|>", "<|assistant|>"], initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer)
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<|end|>", chat_template=CHAT_TEMPLATE)


def make_model(tokenizer, hidden: int, layers: int, heads: int, seed: int) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=len(tokenizer), hidden_size=hidden, intermediate_size=hidden * 3, num_hidden_layers=layers, num_attention_heads=heads,
                      num_key_value_heads=heads, head_dim=hidden // heads, max_position_embeddings=1024, tie_word_embeddings=True,
                      eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    return Qwen3ForCausalLM(cfg)


def main(out: str) -> None:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from state_bridge.data import load_synthetic

    texts = [ex.user_prompt + " " + ex.solution for ex in load_synthetic("train", 512)]
    tok = make_tokenizer(texts)
    for name, hidden, layers, heads, seed in (("sender", 96, 4, 4, 1), ("receiver", 64, 3, 4, 2)):
        d = Path(out) / name
        m = make_model(tok, hidden, layers, heads, seed)
        m.save_pretrained(d)
        tok.save_pretrained(d)
        print(f"wrote {d}: {sum(p.numel() for p in m.parameters())/1e6:.2f}M params")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/tiny")
    main(ap.parse_args().out)
