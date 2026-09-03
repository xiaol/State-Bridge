"""Compare several bridge runs against the receiver-alone and sender-alone baselines.

usage: python scripts/compare_runs.py --base runs/<base_run> --runs runs/<run_a> runs/<run_b> ...

The base run must hold eval_receiver.json and eval_sender.json; every other run holds
eval_bridged.json (and optionally eval_bridged_shuffled.json / eval_bridged_ablated.json and a
train_log.jsonl with val_loss records).  Prints a markdown table with gap closure, relative
uplift and the controls, overall and per difficulty bucket.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(p: Path):
    return json.load(open(p)) if p.exists() else None


def final_val(run: Path):
    p = run / "train_log.jsonl"
    if not p.exists():
        return None
    vals = [json.loads(l) for l in open(p) if "val_loss" in l]
    return vals[-1]["val_loss"] if vals else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    base = Path(a.base)
    r, s = load(base / "eval_receiver.json"), load(base / "eval_sender.json")
    if r is None:
        raise SystemExit("base run lacks eval_receiver.json")
    buckets = list(r["buckets"].keys())
    lines = []
    lines.append("| system | val loss | acc | " + " | ".join(b.split(" ")[0] for b in buckets) + " | gap closed | uplift | shuffled | ablated |")
    lines.append("|---|---|---|" + "---|" * len(buckets) + "---|---|---|---|")

    def row(name, acc, bk, vl=None, shuf=None, abl=None):
        gap = (s["acc"] - r["acc"]) if s else None
        closed = f"{(acc - r['acc']) / gap * 100:.0f}%" if gap else "-"
        up = f"{(acc / r['acc'] - 1) * 100:+.1f}%"
        cells = " | ".join(f"{bk[b]['acc']:.3f}" if b in bk else "-" for b in buckets)
        return f"| {name} | {vl if vl is None else f'{vl:.3f}'} | {acc:.3f} | {cells} | {closed} | {up} | {'-' if shuf is None else f'{shuf:.3f}'} | {'-' if abl is None else f'{abl:.3f}'} |".replace("| None |", "| - |")

    lines.append(row("receiver alone", r["acc"], r["buckets"]))
    for run in a.runs:
        run = Path(run)
        b = load(run / "eval_bridged.json")
        if b is None:
            continue
        sh, ab = load(run / "eval_bridged_shuffled.json"), load(run / "eval_bridged_ablated.json")
        lines.append(row(run.name, b["acc"], b["buckets"], final_val(run), sh["acc"] if sh else None, ab["acc"] if ab else None))
    if s:
        lines.append(row("sender alone", s["acc"], s["buckets"]))
    if s:
        lines.append("")
        lines.append("Per-bucket gap closed / uplift (bridged runs):")
        for run in a.runs:
            run = Path(run)
            b = load(run / "eval_bridged.json")
            if b is None:
                continue
            parts = []
            for bk in buckets:
                g = s["buckets"][bk]["acc"] - r["buckets"][bk]["acc"]
                c = (b["buckets"][bk]["acc"] - r["buckets"][bk]["acc"]) / g if abs(g) > 1e-9 else float("nan")
                parts.append(f"{bk.split(' ')[0]}: closed {c*100:.0f}%, x{b['buckets'][bk]['acc']/max(r['buckets'][bk]['acc'],1e-9):.2f}")
            lines.append(f"- {run.name}: " + "; ".join(parts))
    text = "\n".join(lines) + "\n"
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
