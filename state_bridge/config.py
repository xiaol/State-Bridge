"""YAML configuration with defaults and ``key.sub=value`` overrides."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import yaml

DEFAULTS: dict[str, Any] = {
    "run_name": "default",
    "runs_dir": "runs",
    "seed": 0,
    "hf_endpoint": None,  # e.g. https://hf-mirror.com when huggingface.co is unreachable
    "models": {
        "sender": {"path": "Qwen/Qwen3-1.7B", "device": "cuda:1", "dtype": "bfloat16"},
        # A ".pth" receiver path loads an official RWKV-7 checkpoint (state_bridge.rwkv7); then
        # "tokenizer" may point at rwkv_vocab_v20230424.txt and "chat_prefix" is appended after
        # "Assistant:" (e.g. " <think>\n</think>\n" to skip RWKV7-G1 reasoning).
        "receiver": {"path": "Qwen/Qwen3-0.6B", "device": "cuda:0", "dtype": "bfloat16", "tokenizer": None, "chat_prefix": ""},
        # Which sender layers cross the bridge (hidden_states index: 0 = embeddings,
        # i = output of block i, negative counts from the end).  Several layers are
        # concatenated on the feature axis.
        "sender_layers": [-8, -4],
        "max_prompt_tokens": 512,
    },
    "bridge": {
        # resampler | per_token | prompt_tuning (control: same slots, no sender input)
        "type": "resampler",
        # embed: slots enter the receiver's input-embedding sequence (soft tokens)
        # kv:    slots are projected to key/value prefixes in every receiver layer (transformer receiver)
        # state: slots are written into the initial recurrent state of every layer (RWKV-7 receiver)
        "injection": "kv",
        "kv_gate_init": 0.1,
        "num_slots": 64,
        "d_model": 1024,
        "depth": 2,
        "heads": 8,
        "dropout": 0.0,
        # Gated residual parametrisation: learned constant slots (prompt-tuning component) plus a
        # sender-dependent term scaled by a per-dim gate initialised at gate_init.
        "residual_base": True,   # resampler: add a learned constant per slot
        "num_prefix": 0,         # any bridge: prepend this many learned constant slots
        "gate_init": 0.1,
        # Where the slots enter the receiver's sequence: before the prompt (prefix)
        # or between the prompt and the answer (suffix).
        "position": "prefix",
    },
    "data": {
        "name": "gsm8k",  # gsm8k | synthetic | jsonl
        "path": None,  # for jsonl
        "train_limit": None,
        "eval_limit": None,
        "eval_split": "test",
        "val_size": 200,
        "max_target_tokens": 256,
        # Optional JSONL of training targets written by the models themselves (see
        # precompute.build_targets); avoids the style shift of training on terse gold rationales.
        "targets": None,
        # Optional JSONL of sender-generated solutions on the training split
        # (from `precompute`).  When set, training randomly hands off after
        # 0..handoff_max sender tokens so the bridge learns to read partial
        # reasoning states as well as pure prefill states.
        "sender_generations": None,
        "handoff_max": 256,
        "handoff_prob": 0.5,
    },
    "train": {
        "epochs": 2,
        "batch_size": 16,
        "lr": 2e-4,
        "weight_decay": 0.01,
        "warmup_steps": 50,
        "grad_clip": 1.0,
        "gate_lr_mult": 10.0,
        "log_every": 20,
        "eval_every": 200,
        "max_steps": None,
    },
    "eval": {
        "batch_size": 32,
        "max_new_tokens": 384,
        "sender_max_new_tokens": 512,
        # modes: receiver | sender | bridged | bridged_shuffled | bridged_ablated
        "modes": ["receiver", "bridged", "bridged_shuffled"],
        "checkpoint": None,
    },
    "handoff": {
        "ks": [0, 16, 32, 64, 128, 256],
        "limit": 300,
        "batch_size": 16,
    },
    "geometry": {
        "num_texts": 256,
        "layer_stride": 4,
        "ridge_alpha": 1.0,
    },
    "observe": {
        "limit": 500,
        "steer_alphas": [-2.0, -1.0, 0.0, 1.0, 2.0],
    },
}


def _deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _parse_scalar(s: str) -> Any:
    """YAML-style scalar parsing, plus plain floats like ``3e-4`` that YAML 1.1 treats as strings."""
    try:
        v = yaml.safe_load(s)
    except yaml.YAMLError:
        return s
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return v
    return v


def load_config(path: str | Path | None, overrides: list[str] | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path) as f:
            _deep_update(cfg, yaml.safe_load(f) or {})
    for ov in overrides or []:
        if "=" not in ov:
            raise ValueError(f"override must look like key.sub=value, got {ov!r}")
        key, val = ov.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _parse_scalar(val)
    return cfg


def run_dir(cfg: dict) -> Path:
    d = Path(cfg["runs_dir"]) / cfg["run_name"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def dump_config(cfg: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
