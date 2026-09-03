"""How translated states enter the receiver.

Slots are inserted into the receiver's *input embedding* sequence, so the frozen
receiver treats them like tokens it cannot read but can attend to.  Two placements:

* ``prefix`` - slots come first, so the receiver's own reading of the prompt is
  already conditioned on the sender's state.
* ``suffix`` - slots sit between the prompt and the answer, so only generation
  attends to them and the receiver's prompt processing is untouched.

Training uses right padding (positions of real tokens are exact); generation uses
left padding (``generate`` derives position ids from the attention mask).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .models import LoadedModel

IGNORE = -100


@dataclass
class ReceiverBatch:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor | None
    lengths: list[int]


def build_receiver_batch(
    receiver: LoadedModel,
    prompt_ids: list[list[int]],
    slots: torch.Tensor | None,
    slot_mask: torch.Tensor | None,
    target_ids: list[list[int]] | None,
    position: str = "prefix",
    pad_left: bool = False,
) -> ReceiverBatch:
    emb = receiver.model.get_input_embeddings()
    device = receiver.device
    dtype = emb.weight.dtype
    B = len(prompt_ids)
    seqs, labs = [], []
    for i in range(B):
        p = emb(torch.tensor(prompt_ids[i], device=device))
        parts, lparts = [], []
        s = None
        if slots is not None:
            n_valid = int(slot_mask[i].sum().item())
            s = slots[i, :n_valid].to(device=device, dtype=dtype)
        if position == "prefix":
            if s is not None:
                parts.append(s); lparts.append(torch.full((s.shape[0],), IGNORE, device=device))
            parts.append(p); lparts.append(torch.full((p.shape[0],), IGNORE, device=device))
        elif position == "suffix":
            parts.append(p); lparts.append(torch.full((p.shape[0],), IGNORE, device=device))
            if s is not None:
                parts.append(s); lparts.append(torch.full((s.shape[0],), IGNORE, device=device))
        else:
            raise ValueError(f"unknown slot position {position!r}")
        if target_ids is not None:
            t = torch.tensor(target_ids[i], device=device)
            parts.append(emb(t)); lparts.append(t)
        seqs.append(torch.cat(parts, 0))
        labs.append(torch.cat(lparts, 0))
    lengths = [s.shape[0] for s in seqs]
    L = max(lengths)
    d = seqs[0].shape[1]
    inputs = torch.zeros(B, L, d, device=device, dtype=dtype)
    mask = torch.zeros(B, L, dtype=torch.long, device=device)
    labels = torch.full((B, L), IGNORE, dtype=torch.long, device=device)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        n = s.shape[0]
        if pad_left:
            inputs[i, L - n :] = s; mask[i, L - n :] = 1; labels[i, L - n :] = l
        else:
            inputs[i, :n] = s; mask[i, :n] = 1; labels[i, :n] = l
    return ReceiverBatch(inputs, mask, labels if target_ids is not None else None, lengths)


def encode_prompts(receiver: LoadedModel, texts: list[str], max_tokens: int | None = None) -> list[list[int]]:
    ids = receiver.tokenizer(texts, add_special_tokens=False)["input_ids"]
    if max_tokens is not None:
        ids = [x[:max_tokens] for x in ids]
    return ids


def encode_targets(receiver: LoadedModel, texts: list[str], max_tokens: int | None = None) -> list[list[int]]:
    eos = receiver.tokenizer.eos_token_id
    ids = receiver.tokenizer(texts, add_special_tokens=False)["input_ids"]
    out = []
    for x in ids:
        if max_tokens is not None:
            x = x[: max_tokens - 1]
        out.append(list(x) + [eos])
    return out


@torch.no_grad()
def generate(receiver: LoadedModel, batch: ReceiverBatch, max_new_tokens: int) -> list[str]:
    gen = receiver.model.generate(
        inputs_embeds=batch.inputs_embeds,
        attention_mask=batch.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=None,
        top_p=None,
        top_k=None,
        pad_token_id=receiver.tokenizer.pad_token_id,
    )
    return receiver.tokenizer.batch_decode(gen, skip_special_tokens=True)
