"""Evaluate receiver alone, sender alone, the bridged pair, and the controls.

Controls matter for the claim.  ``bridged_shuffled`` feeds the receiver the sender's
state for a *different* problem; ``bridged_ablated`` replaces the slots with their
dataset mean.  If the bridged accuracy is above both, the gain came through the channel.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch

from .bridge import load_bridge
from .config import run_dir
from .data import Example, extract_answer, is_correct, load_examples, summarize_accuracy
from .injection import build_receiver_batch, generate
from .models import LoadedModel, tokenize_batch
from .train import BridgeSystem

MODES_NEED_SENDER = {"sender", "bridged", "bridged_shuffled", "bridged_ablated"}
MODES_NEED_BRIDGE = {"bridged", "bridged_shuffled", "bridged_ablated"}


@torch.no_grad()
def generate_plain(lm: LoadedModel, prompts: list[str], max_new_tokens: int) -> list[str]:
    enc = tokenize_batch(lm, prompts, padding_side="left")
    out = lm.model.generate(
        input_ids=enc["input_ids"].to(lm.device),
        attention_mask=enc["attention_mask"].to(lm.device),
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=lm.tokenizer.pad_token_id,
    )
    new = out[:, enc["input_ids"].shape[1] :]
    return lm.tokenizer.batch_decode(new, skip_special_tokens=True)


def score_rows(batch: list[Example], texts: list[str], mode: str, extra: dict | None = None) -> list[dict]:
    rows = []
    for ex, txt in zip(batch, texts):
        pred = extract_answer(txt)
        rows.append({"id": ex.id, "mode": mode, "n_steps": ex.n_steps, "gold": ex.answer, "pred": pred, "correct": is_correct(pred, ex.answer), "text": txt, **(extra or {})})
    return rows


class Evaluator:
    def __init__(self, cfg: dict, modes: list[str]):
        self.cfg = cfg
        need_sender = bool(MODES_NEED_SENDER & set(modes))
        bridge = None
        if MODES_NEED_BRIDGE & set(modes):
            ckpt = cfg["eval"]["checkpoint"] or str(run_dir(cfg) / "bridge.pt")
            bridge, bcfg = load_bridge(ckpt)
            # the checkpoint's bridge/model settings win over the eval config
            cfg["bridge"] = bcfg["bridge"]
            cfg["models"]["sender_layers"] = bcfg["models"]["sender_layers"]
            if bridge.kind == "prompt_tuning":
                need_sender = "sender" in modes
            print(f"loaded bridge {bridge.kind} from {ckpt}")
        self.system = BridgeSystem(cfg, need_sender=need_sender, bridge=bridge)
        self.system.bridge.eval()
        self.mean_slots = None

    @torch.no_grad()
    def _bridged(self, batch: list[Example], variant: str, max_new_tokens: int) -> list[str]:
        sys_ = self.system
        slots, slot_mask = sys_.slots_for(batch)
        if variant == "shuffled" and len(batch) > 1:
            slots, slot_mask = slots.roll(1, dims=0), slot_mask.roll(1, dims=0)
        elif variant == "ablated":
            slots = self.mean_slots.unsqueeze(0).expand(len(batch), -1, -1)[:, : slots.shape[1]]
            slot_mask = torch.ones_like(slot_mask)
        rb = build_receiver_batch(sys_.receiver, sys_.receiver_prompt_ids(batch), slots, slot_mask, None, sys_.position, pad_left=True)
        return generate(sys_.receiver, rb, max_new_tokens)

    @torch.no_grad()
    def compute_mean_slots(self, examples: list[Example], bs: int) -> None:
        acc, n = None, 0
        for i in range(0, len(examples), bs):
            slots, mask = self.system.slots_for(examples[i : i + bs])
            s = (slots * mask.unsqueeze(-1)).sum(0)
            acc = s if acc is None else acc + s
            n += mask.sum(0).unsqueeze(-1)
        self.mean_slots = acc / n.clamp(min=1)

    def run(self, mode: str, examples: list[Example], out_path: Path) -> dict:
        ecfg = self.cfg["eval"]
        bs = ecfg["batch_size"]
        rows, t0, n_new = [], time.time(), 0
        if mode == "bridged_ablated" and self.mean_slots is None:
            self.compute_mean_slots(examples, bs)
        with open(out_path, "w") as f:
            for i in range(0, len(examples), bs):
                batch = examples[i : i + bs]
                if mode == "receiver":
                    texts = generate_plain(self.system.receiver, [self.system.receiver.chat_prompt(ex.user_prompt) for ex in batch], ecfg["max_new_tokens"])
                elif mode == "sender":
                    texts = generate_plain(self.system.sender, [self.system.sender.chat_prompt(ex.user_prompt) for ex in batch], ecfg["sender_max_new_tokens"])
                elif mode in ("bridged", "bridged_shuffled", "bridged_ablated"):
                    texts = self._bridged(batch, mode.split("_")[1] if "_" in mode else "plain", ecfg["max_new_tokens"])
                else:
                    raise ValueError(f"unknown eval mode {mode!r}")
                tok = self.system.receiver.tokenizer if mode != "sender" else self.system.sender.tokenizer
                n_new += sum(len(tok(t, add_special_tokens=False)["input_ids"]) for t in texts)
                for r in score_rows(batch, texts, mode):
                    rows.append(r)
                    f.write(json.dumps(r) + "\n")
                done = min(i + bs, len(examples))
                acc = sum(r["correct"] for r in rows) / len(rows)
                print(f"[{mode}] {done}/{len(examples)} acc={acc:.3f} ({time.time()-t0:.0f}s)", flush=True)
        summary = summarize_accuracy(rows)
        summary.update({"mode": mode, "seconds": time.time() - t0, "generated_tokens": n_new})
        return summary


def evaluate(cfg: dict, modes: list[str] | None = None) -> dict:
    out = run_dir(cfg)
    modes = modes or cfg["eval"]["modes"]
    examples = load_examples(cfg["data"], cfg["data"]["eval_split"], cfg["data"]["eval_limit"], cfg["seed"])
    ev = Evaluator(cfg, modes)
    summaries = {}
    for mode in modes:
        s = ev.run(mode, examples, out / f"eval_{mode}.jsonl")
        summaries[mode] = s
        with open(out / f"eval_{mode}.json", "w") as f:
            json.dump(s, f, indent=2)
        print(json.dumps({k: v for k, v in s.items() if k != "buckets"}))
    write_summary(out)
    return summaries


def write_summary(out: Path) -> str:
    """Combine every eval_<mode>.json in a run directory into summary.md with gap-closure numbers."""
    sums = {}
    for p in sorted(out.glob("eval_*.json")):
        with open(p) as f:
            s = json.load(f)
        sums[s["mode"]] = s
    if not sums:
        return ""
    lines = ["# Evaluation summary", "", "| mode | n | accuracy | easy | medium | hard | seconds |", "|---|---|---|---|---|---|---|"]
    order = ["receiver", "bridged_ablated", "bridged_shuffled", "bridged", "sender"]
    for m in [m for m in order if m in sums] + [m for m in sums if m not in order]:
        s = sums[m]
        b = s["buckets"]
        cell = lambda k: next((f"{v['acc']:.3f} (n={v['n']})" for kk, v in b.items() if kk.startswith(k)), "-")
        lines.append(f"| {m} | {s['n']} | {s['acc']:.3f} | {cell('easy')} | {cell('medium')} | {cell('hard')} | {s['seconds']:.0f} |")
    r, s_, br = sums.get("receiver"), sums.get("sender"), sums.get("bridged")
    if r and br:
        lines += ["", f"- relative uplift of receiver from the bridge: **{(br['acc']/max(r['acc'],1e-9)-1)*100:+.1f}%**"]
        if s_:
            gap = s_["acc"] - r["acc"]
            closure = (br["acc"] - r["acc"]) / gap if abs(gap) > 1e-9 else float("nan")
            lines.append(f"- gap sender-receiver: {gap*100:.1f} points; **gap closed: {closure*100:.0f}%**")
            for k in r["buckets"]:
                if k in s_["buckets"] and k in br["buckets"]:
                    g = s_["buckets"][k]["acc"] - r["buckets"][k]["acc"]
                    c = (br["buckets"][k]["acc"] - r["buckets"][k]["acc"]) / g if abs(g) > 1e-9 else float("nan")
                    up = br["buckets"][k]["acc"] / max(r["buckets"][k]["acc"], 1e-9)
                    lines.append(f"  - {k}: gap {g*100:.1f} pts, closed {c*100:.0f}%, uplift x{up:.2f}")
        for ctrl in ("bridged_shuffled", "bridged_ablated"):
            if ctrl in sums:
                lines.append(f"- control {ctrl}: {sums[ctrl]['acc']:.3f} (bridged minus control: {(br['acc']-sums[ctrl]['acc'])*100:+.1f} pts)")
    text = "\n".join(lines) + "\n"
    with open(out / "summary.md", "w") as f:
        f.write(text)
    print(text)
    return text
