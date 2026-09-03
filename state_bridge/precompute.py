"""Let the sender write full solutions for the training split.

Used for hand-off-aware bridge training (the bridge then also sees sender states over
partially written reasoning) and, optionally, as distillation targets.  Supports
sharding so several GPUs can generate in parallel: ``--shard i/n``.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import run_dir
from .data import load_examples
from .evaluate import generate_plain
from .models import load_model


def run_precompute(cfg: dict, shard: str = "0/1", device: str | None = None, split: str = "train") -> Path:
    out = run_dir(cfg)
    i, n = (int(x) for x in shard.split("/"))
    mcfg = cfg["models"]["sender"]
    sender = load_model(mcfg["path"], device or mcfg["device"], mcfg["dtype"], "sender")
    examples = load_examples(cfg["data"], split, cfg["data"]["train_limit"], cfg["seed"])[i::n]
    bs, mnt = cfg["eval"]["batch_size"], cfg["eval"]["sender_max_new_tokens"]
    path = out / (f"sender_gen_{split}.{i}.jsonl" if n > 1 else f"sender_gen_{split}.jsonl")
    with open(path, "w") as f:
        for j in range(0, len(examples), bs):
            batch = examples[j : j + bs]
            texts = generate_plain(sender, [sender.chat_prompt(ex.user_prompt) for ex in batch], mnt)
            for ex, t in zip(batch, texts):
                f.write(json.dumps({"id": ex.id, "text": t}) + "\n")
            f.flush()
            print(f"[precompute {shard}] {min(j+bs,len(examples))}/{len(examples)}", flush=True)
    print(f"wrote {path}")
    return path
