"""Train the bridge.  Both models stay frozen; the bridge is the only thing that learns.

Objective: next-token cross-entropy of the frozen receiver on the gold solution,
conditioned on [bridge(sender prefill states)] + [receiver's own prompt].  Gradients
flow through the frozen receiver into the slots and from there into the bridge.
"""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch

from .bridge import build_bridge, save_bridge
from .config import dump_config, run_dir
from .data import Example, load_examples, load_sender_generations
from .injection import build_receiver_batch, encode_prompts, encode_targets
from .models import LoadedModel, SenderEncoder, load_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lr_at(step: int, total: int, warmup: int, base: float) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, prog)))


class BridgeSystem:
    """Frozen sender + frozen receiver + bridge, wired together.  Shared by train / eval / observe."""

    def __init__(self, cfg: dict, need_sender: bool = True, bridge=None):
        self.cfg = cfg
        mcfg = cfg["models"]
        self.receiver: LoadedModel = load_model(mcfg["receiver"]["path"], mcfg["receiver"]["device"], mcfg["receiver"]["dtype"], name="receiver")
        self.sender: LoadedModel | None = None
        self.encoder: SenderEncoder | None = None
        if need_sender:
            self.sender = load_model(mcfg["sender"]["path"], mcfg["sender"]["device"], mcfg["sender"]["dtype"], name="sender")
            self.encoder = SenderEncoder(self.sender, mcfg["sender_layers"], mcfg["max_prompt_tokens"])
        in_dim = self.encoder.out_dim if self.encoder else 1
        if bridge is None:
            bridge = build_bridge(cfg["bridge"], in_dim, self.receiver.hidden_size, target_rms=self.receiver.embedding_rms)
        self.bridge = bridge.to(self.receiver.device)
        self.position = cfg["bridge"]["position"]
        self.max_prompt = mcfg["max_prompt_tokens"]

    # ------------------------------------------------------------ helpers
    def sender_prompts(self, batch: list[Example], prefixes: list[str] | None = None) -> list[str]:
        assert self.sender is not None
        return [self.sender.chat_prompt(ex.user_prompt, prefixes[i] if prefixes else None) for i, ex in enumerate(batch)]

    def receiver_prompt_ids(self, batch: list[Example], prefixes: list[str] | None = None) -> list[list[int]]:
        texts = [self.receiver.chat_prompt(ex.user_prompt, prefixes[i] if prefixes else None) for i, ex in enumerate(batch)]
        return encode_prompts(self.receiver, texts, self.max_prompt)

    def slots_for(self, batch: list[Example], prefixes: list[str] | None = None, sender_hidden=None, sender_mask=None):
        """Run sender prefill (unless states are given) and the bridge.  Returns (slots, slot_mask)."""
        if not self.bridge.uses_sender:
            dummy = torch.ones(len(batch), 1, dtype=torch.bool, device=self.receiver.device)
            return self.bridge(None, dummy)
        if sender_hidden is None:
            sender_hidden, sender_mask = self.encoder.encode(self.sender_prompts(batch, prefixes))
        sender_hidden = sender_hidden.to(self.receiver.device, torch.float32)
        sender_mask = sender_mask.to(self.receiver.device)
        return self.bridge(sender_hidden, sender_mask)


def train(cfg: dict) -> Path:
    set_seed(cfg["seed"])
    out = run_dir(cfg)
    dump_config(cfg, out / "config.json")
    tcfg, dcfg = cfg["train"], cfg["data"]

    system = BridgeSystem(cfg, need_sender=cfg["bridge"]["type"] != "prompt_tuning")
    bridge, receiver = system.bridge, system.receiver
    print(f"receiver {receiver.name}: {receiver.num_params/1e6:.0f}M params, d={receiver.hidden_size}")
    if system.sender:
        print(f"sender {system.sender.name}: {system.sender.num_params/1e6:.0f}M params, d={system.sender.hidden_size}, layers {system.encoder.layers}")
    print(f"bridge {bridge.kind}: {bridge.num_trainable()/1e6:.2f}M trainable params")

    examples = load_examples(dcfg, "train", dcfg["train_limit"], cfg["seed"])
    rng = random.Random(cfg["seed"])
    rng.shuffle(examples)
    val = examples[: dcfg["val_size"]]
    train_ex = examples[dcfg["val_size"] :]
    sender_gen = load_sender_generations(dcfg["sender_generations"])
    print(f"train {len(train_ex)} / val {len(val)} examples; sender generations for {len(sender_gen)}")

    bs = tcfg["batch_size"]
    steps_per_epoch = math.ceil(len(train_ex) / bs)
    total = tcfg["max_steps"] or steps_per_epoch * tcfg["epochs"]
    opt = torch.optim.AdamW(bridge.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"], betas=(0.9, 0.98))
    log = open(out / "train_log.jsonl", "a")

    def handoff_prefixes(batch: list[Example]) -> list[str] | None:
        """Randomly hand off after 0..handoff_max sender tokens of the sender's own solution."""
        if not sender_gen or not system.sender:
            return None
        prefixes = []
        tok = system.sender.tokenizer
        for ex in batch:
            text = sender_gen.get(ex.id)
            if text is None or rng.random() > dcfg["handoff_prob"]:
                prefixes.append("")
                continue
            ids = tok(text, add_special_tokens=False)["input_ids"]
            k = rng.randint(1, min(dcfg["handoff_max"], len(ids)))
            prefixes.append(tok.decode(ids[:k]))
        return prefixes

    def loss_on(batch: list[Example], prefixes=None) -> torch.Tensor:
        slots, slot_mask = system.slots_for(batch, prefixes)
        prompt_ids = system.receiver_prompt_ids(batch)
        target_ids = encode_targets(receiver, [ex.solution for ex in batch], dcfg["max_target_tokens"])
        rb = build_receiver_batch(receiver, prompt_ids, slots, slot_mask, target_ids, system.position, pad_left=False)
        with torch.autocast(device_type=receiver.device.type, dtype=torch.bfloat16, enabled=receiver.device.type == "cuda"):
            outp = receiver.model(inputs_embeds=rb.inputs_embeds, attention_mask=rb.attention_mask, labels=rb.labels, use_cache=False)
        return outp.loss

    @torch.no_grad()
    def validate() -> float:
        bridge.eval()
        losses = []
        for i in range(0, len(val), bs):
            losses.append(loss_on(val[i : i + bs]).item())
        bridge.train()
        return float(np.mean(losses)) if losses else float("nan")

    step, best = 0, float("inf")
    t0 = time.time()
    vl0 = validate()
    print(json.dumps({"step": 0, "val_loss": vl0}), flush=True)
    log.write(json.dumps({"step": 0, "val_loss": vl0}) + "\n"); log.flush()
    bridge.train()
    done = False
    while not done:
        rng.shuffle(train_ex)
        for i in range(0, len(train_ex), bs):
            batch = train_ex[i : i + bs]
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, tcfg["warmup_steps"], tcfg["lr"])
            loss = loss_on(batch, handoff_prefixes(batch))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(bridge.parameters(), tcfg["grad_clip"])
            opt.step()
            step += 1
            if step % tcfg["log_every"] == 0 or step == 1:
                rec = {"step": step, "loss": loss.item(), "grad_norm": float(gn), "lr": opt.param_groups[0]["lr"], "elapsed": time.time() - t0}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n"); log.flush()
            if step % tcfg["eval_every"] == 0 or step == total:
                vl = validate()
                rec = {"step": step, "val_loss": vl, "elapsed": time.time() - t0}
                print(json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n"); log.flush()
                if vl < best:
                    best = vl
                    save_bridge(bridge, cfg, out / "bridge_best.pt")
            if step >= total:
                done = True
                break
    save_bridge(bridge, cfg, out / "bridge.pt")
    log.close()
    print(f"saved {out/'bridge.pt'} (best val loss {best:.4f})")
    return out / "bridge.pt"
