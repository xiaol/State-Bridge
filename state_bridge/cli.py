"""Command line entry point: ``python -m state_bridge <command> --config cfg.yaml [key=value ...]``."""

from __future__ import annotations

import argparse
import os
import sys

from .config import load_config


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="state-bridge", description=__doc__)
    p.add_argument("command", choices=["train", "eval", "handoff", "geometry", "observe", "compute", "precompute", "targets", "summarize"])
    p.add_argument("--config", "-c", default=None, help="YAML config (defaults are used when omitted)")
    p.add_argument("--modes", default=None, help="comma-separated eval modes, overrides eval.modes (e.g. receiver,bridged)")
    p.add_argument("--shard", default="0/1", help="precompute: i/n")
    p.add_argument("--device", default=None, help="precompute: device override")
    p.add_argument("--role", default="sender", choices=["sender", "receiver"], help="precompute: which model writes")
    p.add_argument("--subset", default=None, help="precompute: 'wrong' or 'wrong:<receiver file>' = only problems the receiver got wrong")
    p.add_argument("--tag", default=None, help="precompute: output file tag (default <i>of<n>)")
    p.add_argument("overrides", nargs="*", help="dotted overrides, e.g. train.lr=1e-4")
    a = p.parse_intermixed_args(argv)  # lets key=value overrides appear before or after flags
    cfg = load_config(a.config, a.overrides)
    if cfg.get("hf_endpoint"):
        os.environ.setdefault("HF_ENDPOINT", cfg["hf_endpoint"])
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    if a.command == "train":
        from .train import train
        train(cfg)
    elif a.command == "eval":
        from .evaluate import evaluate
        evaluate(cfg, [m.strip() for m in a.modes.split(",") if m.strip()] if a.modes else None)
    elif a.command == "summarize":
        from .config import run_dir
        from .evaluate import write_summary
        write_summary(run_dir(cfg))
    elif a.command == "handoff":
        from .handoff import run_handoff
        run_handoff(cfg)
    elif a.command == "geometry":
        from .geometry import run_geometry
        run_geometry(cfg)
    elif a.command == "observe":
        from .observe import run_observe
        run_observe(cfg)
    elif a.command == "precompute":
        from .precompute import run_precompute
        run_precompute(cfg, a.role, a.shard, a.device, subset=a.subset, tag=a.tag)
    elif a.command == "targets":
        from .precompute import build_targets
        build_targets(cfg)
    elif a.command == "compute":
        import json
        from .compute import ModelCost, report
        from .config import run_dir
        from .models import load_model
        out = run_dir(cfg)
        acc = {}
        for m in ("receiver", "sender", "bridged"):
            f = out / f"eval_{m}.json"
            if f.exists():
                acc[m] = json.load(open(f))["acc"]
        costs = []
        for role in ("sender", "receiver"):
            lm = load_model(cfg["models"][role]["path"], "cpu", "bfloat16", role)
            costs.append(ModelCost(role, lm.non_embedding_params, lm.num_layers, lm.hidden_size))
        text = report(costs[0], costs[1], prompt_tokens=int(cfg.get("compute", {}).get("prompt_tokens", 160)), gen_tokens=int(cfg.get("compute", {}).get("gen_tokens", 160)),
                      num_slots=cfg["bridge"]["num_slots"], acc=acc)
        with open(out / "compute.md", "w") as f:
            f.write(text)
        print(text)


if __name__ == "__main__":
    main(sys.argv[1:])
