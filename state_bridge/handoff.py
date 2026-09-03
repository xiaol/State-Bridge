"""Text hand-off vs latent hand-off at matched sender compute.

For each budget ``k`` the sender reads the prompt and writes ``k`` tokens of its own
solution.  Then either the *text* it wrote so far is handed to the receiver, which
continues the answer, or the sender's *hidden states* over prompt + k tokens cross the
bridge and the receiver writes the whole answer.  At k = 0 the text channel has nothing
to pass; the latent channel already carries what the sender learned from reading.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .bridge import load_bridge
from .compute import ModelCost, scenario_flops
from .config import run_dir
from .data import Example, load_examples, summarize_accuracy
from .evaluate import score_rows
from .injection import build_receiver_batch, generate
from .models import tokenize_batch
from .train import BridgeSystem


@torch.no_grad()
def sender_partial(system: BridgeSystem, batch: list[Example], k: int):
    """Sender writes k tokens.  Returns (prompt_ids list, generated ids list (eos-trimmed), decoded text list)."""
    s = system.sender
    prompts = [s.chat_prompt(ex.user_prompt) for ex in batch]
    enc = tokenize_batch(s, prompts, padding_side="left", max_length=system.max_prompt)
    prompt_ids = [[t for t, m in zip(row, mask) if m] for row, mask in zip(enc["input_ids"].tolist(), enc["attention_mask"].tolist())]
    if k == 0:
        return prompt_ids, [[] for _ in batch], ["" for _ in batch]
    out = s.model.generate(
        input_ids=enc["input_ids"].to(s.device), attention_mask=enc["attention_mask"].to(s.device),
        max_new_tokens=k, do_sample=False, temperature=None, top_p=None, top_k=None, pad_token_id=s.tokenizer.pad_token_id,
    )
    new = out[:, enc["input_ids"].shape[1] :].tolist()
    eos = set(s.model.generation_config.eos_token_id if isinstance(s.model.generation_config.eos_token_id, list) else [s.model.generation_config.eos_token_id])
    eos.add(s.tokenizer.pad_token_id)
    gen_ids = []
    for row in new:
        cut = next((i for i, t in enumerate(row) if t in eos), len(row))
        gen_ids.append(row[:cut])
    texts = s.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
    return prompt_ids, gen_ids, texts


@torch.no_grad()
def latent_handoff(system: BridgeSystem, batch: list[Example], prompt_ids, gen_ids, max_new_tokens: int) -> list[str]:
    seqs = [p + g for p, g in zip(prompt_ids, gen_ids)]
    L = max(len(x) for x in seqs)
    pad = system.sender.tokenizer.pad_token_id
    ids = torch.tensor([x + [pad] * (L - len(x)) for x in seqs])
    mask = torch.tensor([[1] * len(x) + [0] * (L - len(x)) for x in seqs])
    hidden, m = system.encoder.encode_ids(ids, mask)
    slots, slot_mask = system.slots_for(batch, sender_hidden=hidden, sender_mask=m)
    rb = build_receiver_batch(system.receiver, system.receiver_prompt_ids(batch), slots, slot_mask, None, system.position, pad_left=True)
    return generate(system.receiver, rb, max_new_tokens)


@torch.no_grad()
def text_handoff(system: BridgeSystem, batch: list[Example], sender_texts: list[str], max_new_tokens: int) -> list[str]:
    ids = system.receiver_prompt_ids(batch, prefixes=sender_texts)
    rb = build_receiver_batch(system.receiver, ids, None, None, None, system.position, pad_left=True)
    cont = generate(system.receiver, rb, max_new_tokens)
    return [s + c for s, c in zip(sender_texts, cont)]


def run_handoff(cfg: dict) -> dict:
    out = run_dir(cfg)
    hcfg = cfg["handoff"]
    ckpt = cfg["eval"]["checkpoint"] or str(out / "bridge.pt")
    bridge, bcfg = load_bridge(ckpt)
    cfg["bridge"] = bcfg["bridge"]
    cfg["models"]["sender_layers"] = bcfg["models"]["sender_layers"]
    system = BridgeSystem(cfg, need_sender=True, bridge=bridge)
    system.bridge.eval()
    examples = load_examples(cfg["data"], cfg["data"]["eval_split"], hcfg["limit"], cfg["seed"])
    bs, mnt = hcfg["batch_size"], cfg["eval"]["max_new_tokens"]
    s_cost = ModelCost("sender", system.sender.non_embedding_params, system.sender.num_layers, system.sender.hidden_size)
    r_cost = ModelCost("receiver", system.receiver.non_embedding_params, system.receiver.num_layers, system.receiver.hidden_size)
    results = {"ks": [], "text": [], "latent": [], "sender_tokens_used": [], "flops_text": [], "flops_latent": []}
    rows_all = []
    for k in hcfg["ks"]:
        rows_t, rows_l, used, plen = [], [], 0, 0
        for i in range(0, len(examples), bs):
            batch = examples[i : i + bs]
            prompt_ids, gen_ids, texts = sender_partial(system, batch, k)
            used += sum(len(g) for g in gen_ids); plen += sum(len(p) for p in prompt_ids)
            rows_t += score_rows(batch, text_handoff(system, batch, texts, mnt), "text_handoff", {"k": k})
            rows_l += score_rows(batch, latent_handoff(system, batch, prompt_ids, gen_ids, mnt), "latent_handoff", {"k": k})
            print(f"[k={k}] {min(i+bs,len(examples))}/{len(examples)} text={sum(r['correct'] for r in rows_t)/len(rows_t):.3f} latent={sum(r['correct'] for r in rows_l)/len(rows_l):.3f}", flush=True)
        acc_t, acc_l = summarize_accuracy(rows_t)["acc"], summarize_accuracy(rows_l)["acc"]
        P, kk = plen / len(examples), used / len(examples)
        fl = scenario_flops(s_cost, r_cost, int(P), mnt // 2, system.bridge.__dict__.get("num_slots", int(P)), int(kk))
        results["ks"].append(k); results["text"].append(acc_t); results["latent"].append(acc_l)
        results["sender_tokens_used"].append(kk); results["flops_text"].append(fl["text_handoff"]); results["flops_latent"].append(fl["bridged"])
        rows_all += rows_t + rows_l
        print(json.dumps({"k": k, "text_handoff_acc": acc_t, "latent_handoff_acc": acc_l, "avg_sender_tokens": kk}), flush=True)
    with open(out / "handoff_rows.jsonl", "w") as f:
        for r in rows_all:
            f.write(json.dumps(r) + "\n")
    with open(out / "handoff.json", "w") as f:
        json.dump(results, f, indent=2)
    lines = ["# Text vs latent hand-off", "", "| sender tokens k | avg used | text hand-off acc | latent hand-off acc | delta |", "|---|---|---|---|---|"]
    for k, u, t, l in zip(results["ks"], results["sender_tokens_used"], results["text"], results["latent"]):
        lines.append(f"| {k} | {u:.0f} | {t:.3f} | {l:.3f} | {(l-t)*100:+.1f} pts |")
    text = "\n".join(lines) + "\n"
    with open(out / "handoff.md", "w") as f:
        f.write(text)
    print(text)
    return results
