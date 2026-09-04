"""Representation geometry: how much structure do sender and receiver already share?

For the same texts we mean-pool every layer of both models and compute
(1) linear CKA between each sender layer and each receiver layer, and
(2) the cross-validated R^2 of a ridge regression from a sender layer to a receiver
layer - a direct measure of how much of the receiver's representation a *linear* bridge
could recover.  When both models share a tokenizer we also report token-level CKA.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .config import run_dir
from .data import load_examples
from .models import SenderEncoder, load_role


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xy = np.linalg.norm(Y.T @ X) ** 2
    xx = np.linalg.norm(X.T @ X)
    yy = np.linalg.norm(Y.T @ Y)
    return float(xy / (xx * yy + 1e-12))


def ridge_r2_cv(X: np.ndarray, Y: np.ndarray, alpha: float = 1.0, folds: int = 5, seed: int = 0) -> float:
    """Cross-validated R^2 of ridge regression X -> Y (multi-output, averaged over dims)."""
    n = X.shape[0]
    idx = np.random.RandomState(seed).permutation(n)
    ss_res, ss_tot = 0.0, 0.0
    for f in range(folds):
        te = idx[f::folds]
        tr = np.setdiff1d(idx, te)
        xm, ym = X[tr].mean(0), Y[tr].mean(0)
        Xt, Yt = X[tr] - xm, Y[tr] - ym
        # dual form keeps this cheap when d >> n
        K = Xt @ Xt.T
        A = np.linalg.solve(K + alpha * np.eye(len(tr)), Yt)
        pred = (X[te] - xm) @ (Xt.T @ A) + ym
        ss_res += ((Y[te] - pred) ** 2).sum()
        ss_tot += ((Y[te] - ym) ** 2).sum()
    return float(1 - ss_res / (ss_tot + 1e-12))


@torch.no_grad()
def pooled_layers(encoder: SenderEncoder, texts: list[str], bs: int = 16) -> np.ndarray:
    outs = []
    for i in range(0, len(texts), bs):
        outs.append(encoder.all_layer_pooled(texts[i : i + bs]).cpu())
    return torch.cat(outs, 1).float().numpy()  # [L+1, N, d]


def run_geometry(cfg: dict) -> dict:
    out = run_dir(cfg)
    gcfg, mcfg = cfg["geometry"], cfg["models"]
    sender = load_role(cfg, "sender")
    receiver = load_role(cfg, "receiver")
    examples = load_examples(cfg["data"], "train", gcfg["num_texts"], cfg["seed"])
    texts = [ex.question for ex in examples]  # raw text, no chat template: the same string for both models
    S = pooled_layers(SenderEncoder(sender, [-1], mcfg["max_prompt_tokens"]), texts)
    R = pooled_layers(SenderEncoder(receiver, [-1], mcfg["max_prompt_tokens"]), texts)
    st = gcfg["layer_stride"]
    s_layers = list(range(0, S.shape[0], st)) + ([S.shape[0] - 1] if (S.shape[0] - 1) % st else [])
    r_layers = list(range(0, R.shape[0], st)) + ([R.shape[0] - 1] if (R.shape[0] - 1) % st else [])
    cka = [[linear_cka(S[i], R[j]) for j in r_layers] for i in s_layers]
    r2 = [[ridge_r2_cv(S[i], R[j], gcfg["ridge_alpha"]) for j in r_layers] for i in s_layers]
    # baseline: how well does the receiver's *own* embedding layer predict its deeper layers?
    self_r2 = [ridge_r2_cv(R[0], R[j], gcfg["ridge_alpha"]) for j in r_layers]
    res = {"sender": sender.name, "receiver": receiver.name, "num_texts": len(texts), "sender_layers": s_layers, "receiver_layers": r_layers,
           "cka": cka, "ridge_r2": r2, "receiver_self_r2_from_embeddings": self_r2}
    # token-level CKA when tokenizations agree
    same = sender.tokenizer(texts[:8])["input_ids"] == receiver.tokenizer(texts[:8])["input_ids"]
    res["same_tokenizer"] = bool(same)
    with open(out / "geometry.json", "w") as f:
        json.dump(res, f, indent=2)
    lines = [f"# Geometry: {sender.name} -> {receiver.name} ({len(texts)} texts, mean-pooled)", ""]
    for name, M in (("Linear CKA", cka), ("Ridge R^2 (5-fold CV) sender layer -> receiver layer", r2)):
        lines += [f"## {name}", "", "| sender \\ receiver | " + " | ".join(f"L{j}" for j in r_layers) + " |", "|---|" + "---|" * len(r_layers)]
        for i, row in zip(s_layers, M):
            lines.append(f"| L{i} | " + " | ".join(f"{v:.2f}" for v in row) + " |")
        lines.append("")
    lines += ["Receiver self-baseline, R^2 from its own embeddings to each layer: " + ", ".join(f"L{j}={v:.2f}" for j, v in zip(r_layers, self_r2)), ""]
    best = max((r2[a][b], s_layers[a], r_layers[b]) for a in range(len(s_layers)) for b in range(len(r_layers)))
    lines.append(f"Most linearly translatable pair: sender L{best[1]} -> receiver L{best[2]} (R^2 = {best[0]:.2f}).")
    text = "\n".join(lines) + "\n"
    with open(out / "geometry.md", "w") as f:
        f.write(text)
    print(text)
    return res
