"""Model-written solutions for the training split, and training targets built from them.

Why: fine-tuning a bridge (or a soft prompt) on the terse gold GSM8K rationales pulls the
receiver away from its own chain-of-thought style and *lowers* its accuracy, which would
confound "gap closed".  Instead the training targets are

* the receiver's own solution when it is correct (no style shift), otherwise
* the sender's solution when the sender is correct (the large model's knowledge, written out
  once at training time), otherwise
* the gold rationale.

``precompute`` supports sharding (``--shard i/n``) so several GPUs can generate in parallel,
and ``--subset wrong`` restricts the sender to problems the receiver got wrong.
"""

from __future__ import annotations

import json
from pathlib import Path

from .config import run_dir
from .data import extract_answer, is_correct, load_examples
from .evaluate import generate_plain
from .models import load_model


def _read_rows(paths) -> dict[str, dict]:
    rows = {}
    for p in paths:
        with open(p) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    rows[r["id"]] = r
    return rows


def run_precompute(cfg: dict, role: str = "sender", shard: str = "0/1", device: str | None = None, split: str = "train",
                   subset: str | None = None, tag: str | None = None) -> Path:
    """``subset``: ``wrong`` = problems the receiver got wrong (all receiver files), or
    ``wrong:<file>`` to use one specific receiver generation file.  ``tag`` names the output."""
    out = run_dir(cfg)
    i, n = (int(x) for x in shard.split("/"))
    mcfg = cfg["models"][role]
    lm = load_model(mcfg["path"], device or mcfg["device"], mcfg["dtype"], role)
    examples = load_examples(cfg["data"], split, cfg["data"]["train_limit"], cfg["seed"])
    if subset and subset.startswith("wrong"):
        files = [subset.split(":", 1)[1]] if ":" in subset else sorted(out.glob(f"gen_receiver_{split}.*.jsonl"))
        recv = _read_rows(files)
        if not recv:
            raise FileNotFoundError("subset=wrong needs receiver generations first")
        examples = [ex for ex in examples if ex.id in recv and not recv[ex.id]["correct"]]
    examples = examples[i::n]
    bs = cfg["eval"]["batch_size"]
    mnt = cfg["eval"]["sender_max_new_tokens"] if role == "sender" else cfg["eval"]["max_new_tokens"]
    path = out / f"gen_{role}_{split}.{tag or f'{i}of{n}'}.jsonl"
    n_ok = 0
    with open(path, "w") as f:
        for j in range(0, len(examples), bs):
            batch = examples[j : j + bs]
            texts = generate_plain(lm, [lm.chat_prompt(ex.user_prompt) for ex in batch], mnt)
            for ex, t in zip(batch, texts):
                pred = extract_answer(t)
                ok = is_correct(pred, ex.answer)
                n_ok += ok
                f.write(json.dumps({"id": ex.id, "text": t.strip(), "pred": pred, "correct": ok}) + "\n")
            f.flush()
            done = min(j + bs, len(examples))
            print(f"[precompute {role} {shard}] {done}/{len(examples)} acc={n_ok/done:.3f}", flush=True)
    print(f"wrote {path}")
    return path


def build_targets(cfg: dict, split: str = "train") -> Path:
    out = run_dir(cfg)
    recv = _read_rows(sorted(out.glob(f"gen_receiver_{split}.*.jsonl")))
    send = _read_rows(sorted(out.glob(f"gen_sender_{split}.*.jsonl")))
    examples = load_examples(cfg["data"], split, cfg["data"]["train_limit"], cfg["seed"])
    counts = {"receiver": 0, "sender": 0, "gold": 0}
    path = out / f"targets_{split}.jsonl"
    with open(path, "w") as f:
        for ex in examples:
            r, s = recv.get(ex.id), send.get(ex.id)
            if r and r["correct"] and "\\boxed" in r["text"]:
                src, text = "receiver", r["text"]
            elif s and s["correct"] and "\\boxed" in s["text"]:
                src, text = "sender", s["text"]
            else:
                src, text = "gold", ex.solution
            counts[src] += 1
            f.write(json.dumps({"id": ex.id, "solution": text, "source": src}) + "\n")
    print(f"wrote {path}: {counts}")
    return path
