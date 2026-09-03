"""The channel as a place to look.

A latent hand-off gives three objects per run: the sender's hidden state, its translated
form (the slots), and the receiver's behaviour.  We record all three, probe what
information survives translation, and then *intervene* on the transferred state to see
what changes downstream.  A feature that can be read out is merely present; a feature
whose manipulation predictably changes the receiver's answer was being used.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from .bridge import load_bridge
from .config import run_dir
from .data import Example, extract_answer, is_correct, load_examples, normalize_number
from .injection import build_receiver_batch, generate
from .train import BridgeSystem


def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    xm, ym = X.mean(0), y.mean(0)
    Xc, yc = X - xm, y - ym
    K = Xc @ Xc.T
    A = np.linalg.solve(K + alpha * np.eye(len(X)), yc)
    w = Xc.T @ A
    return w, xm, ym


def ridge_cv_r2(X: np.ndarray, y: np.ndarray, alpha: float = 1.0, folds: int = 5, seed: int = 0) -> float:
    idx = np.random.RandomState(seed).permutation(len(X))
    res, tot = 0.0, 0.0
    for f in range(folds):
        te, tr = idx[f::folds], np.setdiff1d(idx, idx[f::folds])
        w, xm, ym = ridge_fit(X[tr], y[tr], alpha)
        pred = (X[te] - xm) @ w + ym
        res += ((y[te] - pred) ** 2).sum(); tot += ((y[te] - y[tr].mean(0)) ** 2).sum()
    return float(1 - res / (tot + 1e-12))


def logistic_cv_acc(X: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0, epochs: int = 300, lr: float = 0.1) -> float:
    """Linear probe accuracy (5-fold) with a small torch logistic regression on standardised features."""
    idx = np.random.RandomState(seed).permutation(len(X))
    correct = 0
    for f in range(folds):
        te, tr = idx[f::folds], np.setdiff1d(idx, idx[f::folds])
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Xt, Xe = torch.tensor((X[tr] - mu) / sd, dtype=torch.float32), torch.tensor((X[te] - mu) / sd, dtype=torch.float32)
        yt = torch.tensor(y[tr], dtype=torch.float32)
        w = torch.zeros(X.shape[1], requires_grad=True); b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.Adam([w, b], lr=lr, weight_decay=1e-2)
        for _ in range(epochs):
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Xt @ w + b, yt)
            loss.backward(); opt.step()
        pred = ((Xe @ w + b) > 0).numpy().astype(int)
        correct += int((pred == y[te]).sum())
    return correct / len(X)


def run_observe(cfg: dict) -> dict:
    out = run_dir(cfg)
    ocfg = cfg["observe"]
    ckpt = cfg["eval"]["checkpoint"] or str(out / "bridge.pt")
    bridge, bcfg = load_bridge(ckpt)
    cfg["bridge"] = bcfg["bridge"]; cfg["models"]["sender_layers"] = bcfg["models"]["sender_layers"]
    system = BridgeSystem(cfg, need_sender=True, bridge=bridge)
    system.bridge.eval()
    examples = load_examples(cfg["data"], cfg["data"]["eval_split"], ocfg["limit"], cfg["seed"])
    bs, mnt = cfg["eval"]["batch_size"], cfg["eval"]["max_new_tokens"]

    # ---- 1. record the three objects
    sender_pooled, slots_pooled, slots_all, receiver_prompt_pooled, rows = [], [], [], [], []
    with torch.no_grad():
        for i in range(0, len(examples), bs):
            batch = examples[i : i + bs]
            hidden, mask = system.encoder.encode(system.sender_prompts(batch))
            m = mask.unsqueeze(-1).float()
            sender_pooled.append(((hidden.float() * m).sum(1) / m.sum(1)).cpu())
            slots, slot_mask = system.slots_for(batch, sender_hidden=hidden, sender_mask=mask)
            sm = slot_mask.unsqueeze(-1).float()
            slots_pooled.append(((slots.float() * sm).sum(1) / sm.sum(1)).cpu())
            slots_all.append(slots.float().cpu())
            # the receiver's own reading of the prompt (last layer, mean pooled) as a reference
            pid = system.receiver_prompt_ids(batch)
            rb = build_receiver_batch(system.receiver, pid, None, None, None, system.position, pad_left=False)
            ro = system.receiver.model(inputs_embeds=rb.inputs_embeds, attention_mask=rb.attention_mask, output_hidden_states=True, use_cache=False)
            rm = rb.attention_mask.unsqueeze(-1).float()
            receiver_prompt_pooled.append(((ro.hidden_states[-1].float() * rm).sum(1) / rm.sum(1)).cpu())
            rb = build_receiver_batch(system.receiver, pid, slots, slot_mask, None, system.position, pad_left=True)
            texts = generate(system.receiver, rb, mnt)
            for ex, t in zip(batch, texts):
                p = extract_answer(t)
                rows.append({"id": ex.id, "gold": ex.answer, "pred": p, "correct": is_correct(p, ex.answer), "n_steps": ex.n_steps, "text": t})
            print(f"[record] {min(i+bs,len(examples))}/{len(examples)}", flush=True)
    S = torch.cat(sender_pooled).numpy(); Z = torch.cat(slots_pooled).numpy(); Rp = torch.cat(receiver_prompt_pooled).numpy()
    gold_mag = np.array([math.log10(abs(normalize_number(ex.answer) or 0) + 1) for ex in examples])
    hard = np.array([1 if ex.n_steps >= 5 else 0 for ex in examples])
    correct = np.array([1 if r["correct"] else 0 for r in rows])

    # ---- 2. what survives translation: the same probes on sender state, slots, receiver-own state
    probes = {}
    for name, X in (("sender_state", S), ("translated_slots", Z), ("receiver_own_prompt_state", Rp)):
        probes[name] = {
            "answer_log_magnitude_r2": ridge_cv_r2(X, gold_mag, alpha=10.0),
            "difficulty_probe_acc": logistic_cv_acc(X, hard),
            "difficulty_majority_baseline": float(max(hard.mean(), 1 - hard.mean())),
            "receiver_correct_probe_acc": logistic_cv_acc(X, correct),
            "receiver_correct_majority_baseline": float(max(correct.mean(), 1 - correct.mean())),
        }

    # ---- 3. intervene on the channel: steer along the answer-magnitude direction
    w, xm, ym = ridge_fit(Z, gold_mag, alpha=10.0)
    direction = torch.tensor(w / (np.linalg.norm(w) + 1e-9), dtype=torch.float32)
    slot_rms = float(torch.cat(slots_all).pow(2).mean().sqrt())
    steer = {"alphas": ocfg["steer_alphas"], "mean_pred_log_magnitude": [], "accuracy": [], "parse_rate": []}
    n_sub = min(len(examples), max(64, ocfg["limit"] // 2))
    sub = examples[:n_sub]
    with torch.no_grad():
        for a in ocfg["steer_alphas"]:
            mags, corr, parsed = [], 0, 0
            for i in range(0, n_sub, bs):
                batch = sub[i : i + bs]
                slots, slot_mask = system.slots_for(batch)
                slots = slots + a * slot_rms * direction.to(slots.device, slots.dtype)
                rb = build_receiver_batch(system.receiver, system.receiver_prompt_ids(batch), slots, slot_mask, None, system.position, pad_left=True)
                for ex, t in zip(batch, generate(system.receiver, rb, mnt)):
                    p = extract_answer(t)
                    if p is not None:
                        parsed += 1; mags.append(math.log10(abs(p) + 1))
                    corr += int(is_correct(p, ex.answer))
            steer["mean_pred_log_magnitude"].append(float(np.mean(mags)) if mags else float("nan"))
            steer["accuracy"].append(corr / n_sub); steer["parse_rate"].append(parsed / n_sub)
            print(f"[steer] alpha={a:+.1f} mean log10|answer|={steer['mean_pred_log_magnitude'][-1]:.3f} acc={corr/n_sub:.3f}", flush=True)
    res = {"n": len(examples), "bridged_accuracy": float(correct.mean()), "probes": probes, "steering": steer,
           "gold_mean_log_magnitude": float(gold_mag.mean()), "slot_rms": slot_rms}
    with open(out / "observe.json", "w") as f:
        json.dump(res, f, indent=2)
    with open(out / "observe_rows.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    lines = [f"# Observability ({len(examples)} examples, bridged accuracy {correct.mean():.3f})", "", "## What survives translation (linear probes, 5-fold CV)", "",
             "| representation | answer log-magnitude R^2 | difficulty probe acc (majority) | receiver-correct probe acc (majority) |", "|---|---|---|---|"]
    for k, v in probes.items():
        lines.append(f"| {k} | {v['answer_log_magnitude_r2']:.2f} | {v['difficulty_probe_acc']:.2f} ({v['difficulty_majority_baseline']:.2f}) | {v['receiver_correct_probe_acc']:.2f} ({v['receiver_correct_majority_baseline']:.2f}) |")
    lines += ["", "## Intervening on the channel: steering the slots along the answer-magnitude direction", "",
              f"gold mean log10|answer| = {gold_mag.mean():.3f}", "", "| alpha (x slot RMS) | mean log10|predicted answer| | accuracy | parse rate |", "|---|---|---|---|"]
    for a, m, acc, pr in zip(steer["alphas"], steer["mean_pred_log_magnitude"], steer["accuracy"], steer["parse_rate"]):
        lines.append(f"| {a:+.1f} | {m:.3f} | {acc:.3f} | {pr:.2f} |")
    text = "\n".join(lines) + "\n"
    with open(out / "observe.md", "w") as f:
        f.write(text)
    print(text)
    return res
