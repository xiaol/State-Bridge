"""A simple compute-cost model for comparing deployment options.

FLOPs per processed token are approximated as 2 x (non-embedding parameters), plus an
attention term 4 x layers x hidden x context.  Prefill processes the prompt in one
parallel pass; decoding processes one token per step.  The model is deliberately
coarse: it is meant to order options, not to price them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ModelCost:
    name: str
    params: float  # non-embedding parameters
    layers: int
    hidden: int

    def flops(self, tokens: int, ctx: int) -> float:
        return 2.0 * self.params * tokens + 4.0 * self.layers * self.hidden * tokens * ctx


def scenario_flops(sender: ModelCost, receiver: ModelCost, prompt_tokens: int, gen_tokens: int, num_slots: int, handoff_tokens: int = 0) -> dict:
    """FLOPs for each way of getting an answer out of the pair.

    * receiver_alone / sender_alone: one model reads the prompt and writes the answer.
    * bridged: sender prefill only (+ ``handoff_tokens`` of sender decoding), receiver
      reads slots + prompt and writes the answer.
    * text_handoff: sender writes ``handoff_tokens`` tokens, receiver reads prompt + those
      tokens and writes the rest.
    """
    P, G, K, k = prompt_tokens, gen_tokens, num_slots, handoff_tokens
    ctx = P + G / 2
    return {
        "receiver_alone": receiver.flops(P, P / 2) + receiver.flops(G, ctx),
        "sender_alone": sender.flops(P, P / 2) + sender.flops(G, ctx),
        "bridged": sender.flops(P, P / 2) + sender.flops(k, P + k / 2) + receiver.flops(P + K, (P + K) / 2) + receiver.flops(G, ctx + K),
        "text_handoff": sender.flops(P, P / 2) + sender.flops(k, P + k / 2) + receiver.flops(P + k, (P + k) / 2) + receiver.flops(max(G - k, 0), ctx),
    }


def equivalent_model_params(acc_receiver: float, acc_sender: float, acc_bridged: float, params_receiver: float, params_sender: float) -> float:
    """Size of a single model that would score what the bridged pair scores, assuming accuracy
    is linear in log(params) between the two measured models.  A crude but explicit estimate."""
    if acc_sender <= acc_receiver:
        return float("nan")
    frac = (acc_bridged - acc_receiver) / (acc_sender - acc_receiver)
    frac = min(max(frac, 0.0), 1.0)
    return math.exp(math.log(params_receiver) + frac * (math.log(params_sender) - math.log(params_receiver)))


def report(sender: ModelCost, receiver: ModelCost, prompt_tokens: int, gen_tokens: int, num_slots: int, acc: dict | None = None, handoff_tokens: int = 0) -> str:
    f = scenario_flops(sender, receiver, prompt_tokens, gen_tokens, num_slots, handoff_tokens)
    lines = ["| scenario | GFLOPs / example | vs receiver alone |", "|---|---|---|"]
    for k, v in f.items():
        lines.append(f"| {k} | {v/1e9:,.1f} | x{v/f['receiver_alone']:.2f} |")
    if acc and all(m in acc for m in ("receiver", "sender", "bridged")):
        m = equivalent_model_params(acc["receiver"], acc["sender"], acc["bridged"], receiver.params, sender.params)
        if not math.isnan(m):
            mid = ModelCost("equivalent-single-model", m, receiver.layers, receiver.hidden)
            mid_flops = mid.flops(prompt_tokens, prompt_tokens / 2) + mid.flops(gen_tokens, prompt_tokens + gen_tokens / 2)
            ratio = mid_flops / f["bridged"]
            verdict = (f"the bridged pair uses **{ratio:.2f}x less compute** than that model." if ratio >= 1
                       else f"the bridged pair costs **{1/ratio:.1f}x more** than that model: at this accuracy the bridge does not pay for the sender's prefill.")
            lines += ["", f"A single model scoring what the bridged pair scores would need about {m/1e9:.2f}B non-embedding parameters",
                      f"(log-linear interpolation between receiver and sender).  Its cost: {mid_flops/1e9:,.1f} GFLOPs/example, i.e. {verdict}"]
    return "\n".join(lines) + "\n"
