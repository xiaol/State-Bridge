"""Datasets, prompt formatting, answer extraction and difficulty buckets.

GSM8K is the default benchmark: short grade-school word problems with a numeric
answer, which makes "did the receiver get it right" unambiguous.  A synthetic
arithmetic task is included for offline smoke tests, and any JSONL with
``question`` / ``answer`` (and optionally ``solution``) fields can be used.
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

USER_TEMPLATE = "{question}\nPlease reason step by step, and put your final answer within \\boxed{{}}."


@dataclass
class Example:
    id: str
    question: str
    answer: str  # canonical gold answer string
    solution: str  # target text for training (rationale ending in \boxed{answer})
    n_steps: int = 1
    extra: dict = field(default_factory=dict)

    @property
    def user_prompt(self) -> str:
        return USER_TEMPLATE.format(question=self.question)


_BOXED = re.compile(r"\\boxed\{")
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def _last_boxed(text: str) -> str | None:
    starts = [m.end() for m in _BOXED.finditer(text)]
    if not starts:
        return None
    i = starts[-1]
    depth, j = 1, i
    while j < len(text) and depth:
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
        j += 1
    return text[i : j - 1] if depth == 0 else text[i:]


def normalize_number(s: str | None) -> float | None:
    if s is None:
        return None
    s = s.replace(",", "").replace("$", "").replace("%", "").strip()
    s = s.rstrip(".")
    m = _NUM.findall(s)
    if not m:
        return None
    try:
        return float(m[-1])
    except ValueError:
        return None


def extract_answer(text: str) -> float | None:
    """Last \\boxed{...} if present, else the last number in the text."""
    boxed = _last_boxed(text)
    if boxed is not None:
        v = normalize_number(boxed)
        if v is not None:
            return v
    return normalize_number(text)


def is_correct(pred: float | None, gold: str) -> bool:
    g = normalize_number(gold)
    if pred is None or g is None:
        return False
    return abs(pred - g) <= 1e-4 * max(1.0, abs(g))


def difficulty_bucket(n_steps: int) -> str:
    if n_steps <= 3:
        return "easy (<=3 steps)"
    if n_steps <= 5:
        return "medium (4-5 steps)"
    return "hard (>=6 steps)"


# ---------------------------------------------------------------- GSM8K


_CALC = re.compile(r"<<[^>]*>>")


def _gsm8k_example(row: dict, idx: int, split: str) -> Example:
    sol = row["answer"]
    body, _, final = sol.partition("####")
    final = final.strip()
    lines = [l for l in body.strip().splitlines() if l.strip()]
    body_clean = _CALC.sub("", body).strip()
    target = f"{body_clean}\nThe final answer is \\boxed{{{final}}}."
    return Example(id=f"gsm8k-{split}-{idx}", question=row["question"].strip(), answer=final, solution=target, n_steps=max(1, len(lines)))


def load_gsm8k(split: str, limit: int | None = None, path: str | None = None) -> list[Example]:
    """GSM8K from the hub, or from a local directory holding ``main/{train,test}-*.parquet``."""
    from datasets import load_dataset

    if path:
        files = sorted(str(p) for p in Path(path).glob(f"**/{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(f"no {split} parquet files under {path}")
        ds = load_dataset("parquet", data_files={split: files}, split=split)
    else:
        ds = load_dataset("openai/gsm8k", "main", split=split)
    out = [_gsm8k_example(r, i, split) for i, r in enumerate(ds)]
    return out[:limit] if limit else out


# ---------------------------------------------------------------- synthetic


def load_synthetic(split: str, limit: int | None = None, seed: int = 0) -> list[Example]:
    """Two-step arithmetic word problems; used for offline smoke tests."""
    rng = random.Random(seed + (1 if split == "test" else 0))
    n = limit or (512 if split == "train" else 64)
    names = ["Ana", "Ben", "Chen", "Dee", "Eli", "Fay"]
    items = ["apples", "coins", "books", "cards", "stamps", "shells"]
    out = []
    for i in range(n):
        a, b, c = rng.randint(2, 40), rng.randint(2, 40), rng.randint(2, 9)
        nm, it = rng.choice(names), rng.choice(items)
        q = f"{nm} has {a} {it}. {nm} buys {b} more and then gives away {c}. How many {it} does {nm} have now?"
        ans = a + b - c
        sol = f"{a} + {b} = {a + b}\n{a + b} - {c} = {ans}\nThe final answer is \\boxed{{{ans}}}."
        out.append(Example(id=f"syn-{split}-{i}", question=q, answer=str(ans), solution=sol, n_steps=2))
    return out


# ---------------------------------------------------------------- JSONL


def load_jsonl(path: str, split: str, limit: int | None = None) -> list[Example]:
    p = Path(path)
    if p.is_dir():
        p = p / f"{split}.jsonl"
    out = []
    with open(p) as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            r = json.loads(line)
            sol = r.get("solution") or f"The final answer is \\boxed{{{r['answer']}}}."
            out.append(Example(id=r.get("id", f"{split}-{i}"), question=r["question"], answer=str(r["answer"]), solution=sol, n_steps=int(r.get("n_steps", 1))))
    return out[:limit] if limit else out


def load_examples(dcfg: dict, split: str, limit: int | None = None, seed: int = 0) -> list[Example]:
    name = dcfg["name"]
    if name == "gsm8k":
        out = load_gsm8k(split, limit, dcfg.get("path"))
    elif name == "synthetic":
        out = load_synthetic(split, limit, seed)
    elif name == "jsonl":
        out = load_jsonl(dcfg["path"], split, limit)
    else:
        raise ValueError(f"unknown dataset {name!r}")
    if split == "train" and dcfg.get("targets"):
        apply_targets(out, dcfg["targets"])
    return out


def apply_targets(examples: list[Example], path: str) -> None:
    """Replace gold solutions with model-written targets (see precompute.build_targets)."""
    targets = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                targets[r["id"]] = r
    n = 0
    for ex in examples:
        t = targets.get(ex.id)
        if t:
            ex.solution = t["solution"]
            ex.extra["target_source"] = t.get("source")
            n += 1
    print(f"applied {n}/{len(examples)} model-written targets from {path}")


def load_sender_generations(path: str | None) -> dict[str, str]:
    """id -> sender-generated solution text (from `precompute`); ``path`` may be a glob over shards."""
    if not path:
        return {}
    import glob

    files = sorted(glob.glob(path)) or [path]
    out = {}
    for fp in files:
        with open(fp) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    out[r["id"]] = r["text"]
    return out


def summarize_accuracy(rows: list[dict]) -> dict:
    """rows have keys correct (bool) and n_steps (int)."""
    total = len(rows)
    acc = sum(r["correct"] for r in rows) / max(1, total)
    by = {}
    for r in rows:
        b = difficulty_bucket(r["n_steps"])
        by.setdefault(b, []).append(r["correct"])
    buckets = {b: {"n": len(v), "acc": sum(v) / len(v)} for b, v in sorted(by.items())}
    return {"n": total, "acc": acc, "buckets": buckets}
