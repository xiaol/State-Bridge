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
from .injection import build_receiver_batch, encode_prompts, encode_targets, forward_with_prefix, generate, greedy_generate_with_prefix
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
        self.position = cfg["bridge"]["position"]
        self.max_prompt = mcfg["max_prompt_tokens"]
        rc = self.receiver.model.config
        rc = getattr(rc, "text_config", None) or rc
        kv_dims = (rc.num_hidden_layers, rc.num_key_value_heads, getattr(rc, "head_dim", None) or rc.hidden_size // rc.num_attention_heads)
        fresh = bridge is None
        if fresh:
            bridge = build_bridge(cfg["bridge"], in_dim, self.receiver.hidden_size, target_rms=self.receiver.embedding_rms, kv_dims=kv_dims)
        self.bridge = bridge.to(self.receiver.device)
        self.injection = getattr(self.bridge, "injection", "embed")
        if fresh and self.bridge.kv_head is not None:
            self.bridge.kv_head.calibrate(self.receiver, self.receiver.chat_prompt("Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"))

    # ------------------------------------------------------------ helpers
    def sender_prompts(self, batch: list[Example], prefixes: list[str] | None = None) -> list[str]:
        assert self.sender is not None
        return [self.sender.chat_prompt(ex.user_prompt, prefixes[i] if prefixes else None) for i, ex in enumerate(batch)]

    def receiver_prompt_ids(self, batch: list[Example], prefixes: list[str] | None = None) -> list[list[int]]:
        texts = [self.receiver.chat_prompt(ex.user_prompt, prefixes[i] if prefixes else None) for i, ex in enumerate(batch)]
        return encode_prompts(self.receiver, texts, self.max_prompt)

    def receiver_loss(self, batch: list[Example], slots, slot_mask, targets: list[str], max_target_tokens: int | None):
        """Cross-entropy of the frozen receiver on ``targets`` given the slots (either injection mode)."""
        r = self.receiver
        prompt_ids = self.receiver_prompt_ids(batch)
        target_ids = encode_targets(r, targets, max_target_tokens)
        with torch.autocast(device_type=r.device.type, dtype=torch.bfloat16, enabled=r.device.type == "cuda"):
            if self.injection == "kv":
                rb = build_receiver_batch(r, prompt_ids, None, None, target_ids, self.position, pad_left=False)
                out = forward_with_prefix(r, self.bridge.kv_head(slots), slot_mask, rb.inputs_embeds, rb.attention_mask, rb.labels)
            else:
                rb = build_receiver_batch(r, prompt_ids, slots, slot_mask, target_ids, self.position, pad_left=False)
                out = r.model(inputs_embeds=rb.inputs_embeds, attention_mask=rb.attention_mask, labels=rb.labels, use_cache=False)
        return out.loss

    @torch.no_grad()
    def receiver_generate(self, batch: list[Example], slots, slot_mask, max_new_tokens: int, prefixes: list[str] | None = None) -> list[str]:
        """Greedy generation; ``slots=None`` means the receiver runs alone (optionally continuing ``prefixes``)."""
        r = self.receiver
        prompt_ids = self.receiver_prompt_ids(batch, prefixes)
        if slots is None:
            rb = build_receiver_batch(r, prompt_ids, None, None, None, self.position, pad_left=True)
            return generate(r, rb, max_new_tokens)
        if self.injection == "kv":
            rb = build_receiver_batch(r, prompt_ids, None, None, None, self.position, pad_left=True)
            return greedy_generate_with_prefix(r, self.bridge.kv_head(slots), slot_mask, rb.inputs_embeds, rb.attention_mask, max_new_tokens)
        rb = build_receiver_batch(r, prompt_ids, slots, slot_mask, None, self.position, pad_left=True)
        return generate(r, rb, max_new_tokens)

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
    # gates and gains: higher learning rate, no weight decay, so the sender-dependent part can grow
    fast = [p for n, p in bridge.named_parameters() if n.split(".")[-1] in ("gate", "gain")]
    slow = [p for n, p in bridge.named_parameters() if n.split(".")[-1] not in ("gate", "gain")]
    opt = torch.optim.AdamW(
        [{"params": slow, "lr": tcfg["lr"], "weight_decay": tcfg["weight_decay"]}, {"params": fast, "lr": tcfg["lr"] * tcfg.get("gate_lr_mult", 10.0), "weight_decay": 0.0}],
        betas=(0.9, 0.98),
    )
    base_lrs = [g["lr"] for g in opt.param_groups]
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
        return system.receiver_loss(batch, slots, slot_mask, [ex.solution for ex in batch], dcfg["max_target_tokens"])

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
            for g, b in zip(opt.param_groups, base_lrs):
                g["lr"] = lr_at(step, total, tcfg["warmup_steps"], b)
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
