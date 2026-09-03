"""The bridge: the only trainable part of the system.

Three variants share one interface, ``forward(sender_hidden, sender_mask) -> (slots, slot_mask)``:

* ``ResamplerBridge`` - a fixed number of learned queries cross-attend over the sender's
  token states (Perceiver-style), producing ``num_slots`` vectors in the receiver's
  input-embedding space.  Length-invariant and compact.
* ``PerTokenBridge`` - a per-position MLP, one receiver slot per sender token.  Keeps all
  positions, no compression.
* ``PromptTuningBridge`` - a control.  Same number of slots, but they are learned constants
  that ignore the sender entirely.  Any accuracy the bridged system has *above* this
  control is information that crossed the channel, not an artefact of soft prompting.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_positions(length: int, dim: int, device, dtype=torch.float32) -> torch.Tensor:
    pos = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
    return pe.to(dtype)


class OutputScale(nn.Module):
    """Rescales bridge output so its RMS matches the receiver's token embeddings at init.

    Frozen receivers are sensitive to the scale of what enters their residual stream;
    matching the embedding RMS keeps the first training steps in-distribution."""

    def __init__(self, dim: int, target_rms: float):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.scale = nn.Parameter(torch.full((dim,), float(target_rms)))

    def forward(self, x):
        return self.norm(x) * self.scale


class BridgeBase(nn.Module):
    """Shared machinery: an optional learned constant prefix (a prompt-tuning component) and a
    per-dimension gate on the sender-dependent slots.  With ``gate_init`` small the system
    starts out as plain prompt tuning and learns to add sender-dependent deviations on top,
    which is much easier to optimise through a frozen receiver than sender-dependent slots
    from scratch."""

    kind = "base"
    uses_sender = True

    def __init__(self, in_dim: int, out_dim: int, num_prefix: int = 0, gate_init: float = 1.0, target_rms: float = 1.0):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_prefix = num_prefix
        self.prefix = nn.Parameter(torch.randn(num_prefix, out_dim) * target_rms) if num_prefix > 0 else None
        self.gate = nn.Parameter(torch.full((out_dim,), float(gate_init)))

    def finalize(self, slots: torch.Tensor, slot_mask: torch.Tensor):
        slots = slots * self.gate
        if self.prefix is not None:
            B = slots.shape[0]
            slots = torch.cat([self.prefix.unsqueeze(0).expand(B, -1, -1).to(slots.dtype), slots], 1)
            slot_mask = torch.cat([torch.ones(B, self.num_prefix, dtype=torch.bool, device=slots.device), slot_mask.bool()], 1)
        return slots, slot_mask

    def forward(self, sender_hidden: torch.Tensor, sender_mask: torch.Tensor):  # pragma: no cover
        raise NotImplementedError

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class CrossAttnBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float):
        super().__init__()
        self.ln_q = nn.LayerNorm(d_model)
        self.ln_kv = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ln_self = nn.LayerNorm(d_model)
        self.self_attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.ln_mlp = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4 * d_model), nn.GELU(), nn.Linear(4 * d_model, d_model))

    def forward(self, q, kv, key_padding_mask):
        a, _ = self.attn(self.ln_q(q), self.ln_kv(kv), self.ln_kv(kv), key_padding_mask=key_padding_mask, need_weights=False)
        q = q + a
        s, _ = self.self_attn(self.ln_self(q), self.ln_self(q), self.ln_self(q), need_weights=False)
        q = q + s
        return q + self.mlp(self.ln_mlp(q))


class ResamplerBridge(BridgeBase):
    kind = "resampler"

    def __init__(self, in_dim, out_dim, num_slots=64, d_model=1024, depth=2, heads=8, dropout=0.0, target_rms=1.0, max_len=4096,
                 residual_base=False, num_prefix=0, gate_init=1.0):
        super().__init__(in_dim, out_dim, num_prefix=num_prefix, gate_init=gate_init, target_rms=target_rms)
        self.num_slots = num_slots + num_prefix
        self.d_model = d_model
        # learned constant per slot; the sender-dependent part is added on top
        self.base = nn.Parameter(torch.randn(num_slots, out_dim) * target_rms) if residual_base else None
        self.in_norm = nn.LayerNorm(in_dim)
        self.in_proj = nn.Linear(in_dim, d_model)
        self.queries = nn.Parameter(torch.randn(num_slots, d_model) * 0.02)
        self.blocks = nn.ModuleList([CrossAttnBlock(d_model, heads, dropout) for _ in range(depth)])
        self.out_proj = nn.Linear(d_model, out_dim)
        self.out_scale = OutputScale(out_dim, target_rms)
        self.register_buffer("pe", sinusoidal_positions(max_len, d_model, torch.device("cpu")), persistent=False)

    def forward(self, sender_hidden, sender_mask):
        B, T, _ = sender_hidden.shape
        x = self.in_proj(self.in_norm(sender_hidden.float()))
        x = x + self.pe[:T].to(x.device, x.dtype)
        kpm = ~sender_mask.bool()  # True = ignore
        q = self.queries.unsqueeze(0).expand(B, -1, -1)
        for blk in self.blocks:
            q = blk(q, x, kpm)
        slots = self.out_scale(self.out_proj(q)) * self.gate
        if self.base is not None:
            slots = slots + self.base.unsqueeze(0)
        mask = torch.ones(B, slots.shape[1], dtype=torch.bool, device=slots.device)
        if self.prefix is not None:
            slots = torch.cat([self.prefix.unsqueeze(0).expand(B, -1, -1), slots], 1)
            mask = torch.ones(B, slots.shape[1], dtype=torch.bool, device=slots.device)
        return slots, mask


class PerTokenBridge(BridgeBase):
    kind = "per_token"

    def __init__(self, in_dim, out_dim, d_model=1024, target_rms=1.0, dropout=0.0, num_prefix=0, gate_init=1.0, **_):
        super().__init__(in_dim, out_dim, num_prefix=num_prefix, gate_init=gate_init, target_rms=target_rms)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim), nn.Linear(in_dim, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, out_dim)
        )
        self.out_scale = OutputScale(out_dim, target_rms)

    def forward(self, sender_hidden, sender_mask):
        slots = self.out_scale(self.net(sender_hidden.float()))
        return self.finalize(slots, sender_mask.bool())


class PromptTuningBridge(BridgeBase):
    """Control: learned constant slots, sender input ignored."""

    kind = "prompt_tuning"
    uses_sender = False

    def __init__(self, in_dim, out_dim, num_slots=64, target_rms=1.0, **_):
        super().__init__(in_dim, out_dim)
        self.num_slots = num_slots
        self.slots = nn.Parameter(torch.randn(num_slots, out_dim) * target_rms)

    def forward(self, sender_hidden, sender_mask):
        B = sender_hidden.shape[0] if sender_hidden is not None else sender_mask.shape[0]
        slots = self.slots.unsqueeze(0).expand(B, -1, -1)
        return slots, torch.ones(B, self.num_slots, dtype=torch.bool, device=slots.device)


def build_bridge(bcfg: dict, in_dim: int, out_dim: int, target_rms: float) -> BridgeBase:
    kind = bcfg["type"]
    common = dict(in_dim=in_dim, out_dim=out_dim, target_rms=target_rms)
    extra = dict(num_prefix=bcfg.get("num_prefix", 0), gate_init=bcfg.get("gate_init", 1.0))
    if kind == "resampler":
        return ResamplerBridge(
            num_slots=bcfg["num_slots"], d_model=bcfg["d_model"], depth=bcfg["depth"], heads=bcfg["heads"], dropout=bcfg["dropout"],
            residual_base=bcfg.get("residual_base", False), **extra, **common
        )
    if kind == "per_token":
        return PerTokenBridge(d_model=bcfg["d_model"], dropout=bcfg["dropout"], **extra, **common)
    if kind == "prompt_tuning":
        return PromptTuningBridge(num_slots=bcfg["num_slots"], **common)
    raise ValueError(f"unknown bridge type {kind!r}")


def save_bridge(bridge: BridgeBase, cfg: dict, path) -> None:
    torch.save({"cfg": cfg, "in_dim": bridge.in_dim, "out_dim": bridge.out_dim, "state_dict": bridge.state_dict()}, path)


def load_bridge(path, map_location="cpu") -> tuple[BridgeBase, dict]:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ckpt["cfg"]
    bridge = build_bridge(cfg["bridge"], ckpt["in_dim"], ckpt["out_dim"], target_rms=1.0)
    missing, unexpected = bridge.load_state_dict(ckpt["state_dict"], strict=False)
    # checkpoints written before the gate existed behave as gate == 1, which is its default init here
    if unexpected or set(missing) - {"gate"}:
        raise RuntimeError(f"checkpoint mismatch: missing {missing}, unexpected {unexpected}")
    return bridge, cfg
