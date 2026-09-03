# Design

This document maps each claim in the source write-up (see `source-notes.md`) to the code that
tests it, and records the decisions taken where the write-up leaves details open.

## 1. The channel

**Claim.** One model hands its hidden states to another through a small trained bridge; the
receiver works with them directly; no text passes; neither model's weights change.

**Implementation.**

```
                 frozen sender (reads only)                    frozen receiver (writes)
 prompt ──► tokenizer_s ──► transformer ──► hidden states ──► BRIDGE ──► slots ─┐
                                              at layers L        (trained)        ├─► [slots | prompt_r] ──► generate
 prompt ──► tokenizer_r ──► input embeddings ────────────────────────────────────┘
```

- `models.SenderEncoder` runs the sender's prefill with `output_hidden_states=True`, takes the
  residual-stream states at the configured layers (default the 8th- and 4th-from-last blocks)
  and concatenates them on the feature axis. Right padding keeps positions exact. The sender
  never decodes in the basic setting.
- `bridge.ResamplerBridge` (default): LayerNorm and projection of the sender states, sinusoidal
  positions, then `num_slots` learned queries run through `depth` blocks of cross-attention
  over the sender tokens, self-attention among the slots, and an MLP. The output is projected
  to the receiver's hidden size and rescaled by `OutputScale` so that at initialisation the
  slots have the same RMS as the receiver's token embeddings. This keeps the frozen receiver
  in-distribution during the first steps. About 43M parameters for a 4096 -> 1024 pair.
- `bridge.PerTokenBridge`: one slot per sender token through an MLP, no compression. Useful
  when the receiver can afford a longer context.
- Gated residual parametrisation (both bridges): `slots = C + g * f(sender)`, where `C` are
  learned constant slots (`residual_base` for the resampler, `num_prefix` constant slots for
  any bridge) and `g` is a per-dimension gate initialised at `gate_init` (0.1). A first run
  without this (sender-dependent slots from scratch, gate 1.0) trained much more slowly than
  the prompt-tuning control and never caught up within two epochs: the frozen receiver is
  disturbed by slots it cannot yet interpret, and the gradient signal through it is weak.
  Starting from the prompt-tuning solution removes that obstacle; the sender-dependent term
  only has to explain what the constants cannot.
- `injection.build_receiver_batch` places the slots in the receiver's *input embedding*
  sequence, either before the prompt (`prefix`, default) or between prompt and answer
  (`suffix`). The receiver then sees a sequence of embeddings it cannot read but can attend
  to, exactly like soft prompts. Generation uses `generate(inputs_embeds=...)` with left
  padding; training uses right padding and a label mask that only scores the answer tokens.

**Why embedding-level injection.** The write-up says "prefill state" crosses. The most literal
reading is injecting into the receiver's KV cache at every layer. We chose embedding-level
injection because it (a) needs no model surgery, so any Hugging Face causal LM is a receiver,
(b) is length-invariant with the resampler, (c) already lets the receiver's own layers process
the slots in context with the prompt. Per-layer KV injection is a natural extension
(see section 8).

## 2. Both models frozen

`models.load_model` calls `requires_grad_(False)` and `eval()` on both models. The optimiser
sees only `bridge.parameters()`. Gradients flow *through* the frozen receiver into the slots,
which is what lets the bridge learn what the receiver can use.

## 3. Training objective

Next-token cross-entropy of the frozen receiver on a target solution, conditioned on
`[slots] + [receiver chat prompt]`. Only answer tokens are scored. AdamW, cosine schedule with
warm-up, gradient clipping, bf16 autocast for the receiver, fp32 bridge.

**Which targets.** The obvious choice, the gold GSM8K rationales (calculator annotations removed,
final line rewritten as `The final answer is \boxed{N}.`), turned out to be a trap: they are
terse, and a soft prompt trained on them pulls the receiver away from its own chain-of-thought
style. The prompt-tuning control trained on gold reached a *lower* LM loss than any bridge yet
scored 36% on GSM8K against 62% for the untouched receiver. Any "gap closed" computed against
receiver-alone would then be dominated by style, not information.

The default pipeline therefore builds targets from the models themselves
(`precompute` + `targets`): the receiver's own solution where it is correct (no style shift),
the sender's solution where the receiver is wrong and the sender is right (the large model's
knowledge, written out once at training time), and gold otherwise. With these targets the
prompt-tuning control should sit near receiver-alone, and anything above it is the channel.

Optionally (`data.sender_generations`), the sender's own solutions are precomputed and, with
probability `handoff_prob`, training hands off after a random number of sender-written tokens.
This makes the bridge read states over partially written reasoning as well as pure prefill
states, which the text-vs-latent hand-off sweep needs.

## 4. Controls that make the claim testable

The write-up's strict test is that the bridge exploits structure "already present". A learned
soft prompt with no sender input can also raise a small model's accuracy, so two controls
separate "information crossed the channel" from "soft prompting helped":

| control | what it does | expectation if the channel carries information |
|---|---|---|
| `bridge.type=prompt_tuning` | same number of slots, learned constants, sender ignored | bridged accuracy well above it |
| eval mode `bridged_shuffled` | slots from a *different* problem in the batch | accuracy drops toward / below receiver alone |
| eval mode `bridged_ablated` | slots replaced by their dataset mean | accuracy drops to roughly the prompt-tuning level |

## 5. Headline metrics

`evaluate.write_summary` reports, from `eval_<mode>.json` files:

- gap = sender accuracy - receiver accuracy; **gap closed** = (bridged - receiver) / gap;
- **relative uplift** = bridged / receiver - 1;
- the same per difficulty bucket, where difficulty is the number of reasoning lines in the gold
  GSM8K solution (<=3, 4-5, >=6). The write-up reports up to 2x uplift on hard subsets.

## 6. Text hand-off vs latent hand-off (`handoff.py`)

For each budget k, the sender writes k tokens of its own solution. Text channel: the receiver
continues the assistant turn from that text. Latent channel: the sender's hidden states over
prompt + k tokens cross the bridge and the receiver writes the whole answer. At k = 0 the text
channel is receiver-alone and the latent channel is the bridged system. Sender compute per k is
reported with `compute.scenario_flops`.

## 7. Compute model (`compute.py`)

FLOPs per token = 2 x non-embedding parameters plus an attention term. Prefill is one pass over
the prompt; decoding is one token per step. The "equivalent mid-sized model" is estimated by
interpolating accuracy linearly in log(parameters) between receiver and sender, which is a
stated assumption, not a measurement; the write-up's 2.5x figure presumably came from an actual
mid-sized model.

## 8. Geometry (`geometry.py`)

"We started with the geometry." For the same raw texts we mean-pool every layer of both models
and compute linear CKA between all sender/receiver layer pairs, and the cross-validated R^2 of a
ridge map from a sender layer to a receiver layer. R^2 is the quantity the write-up alludes to
with "how much work it takes to find a translation": high R^2 means a linear bridge would do,
low R^2 means the translation is non-linear or the information is not shared.

## 9. Observability (`observe.py`)

Records the three objects per example (sender state, slots, receiver behaviour), then:

- probes what survives translation: the same linear probes (answer magnitude, difficulty,
  receiver-correct) on sender state, slots, and the receiver's own prompt state;
- intervenes on the channel: adds a multiple of the probe direction for answer magnitude to
  every slot and measures whether the receiver's answers move in magnitude. A monotone response
  is evidence that the feature is *used*, the distinction the write-up draws.

## 10. What is not replicated, and extension points

- Scale: 753B -> 4B is replaced by 9B -> 0.6B (and 1.7B -> 0.6B). Same shape of experiment,
  much smaller gap in absolute knowledge.
- Per-layer KV injection: `injection.py` is the single place to add it; the bridge output
  would become per-layer key/value tensors (`num_layers x 2 x heads x head_dim`) appended to a
  `DynamicCache` before the prompt.
- Training-time channel (distillation, specialisation, several models training together):
  the bridge already carries gradients; unfreezing the receiver in `train.py` and adding the
  teacher's slots to its training inputs is the direct extension.
- Serving: no endpoint-to-endpoint transport of latent state. `SenderEncoder.encode` returns
  the tensors that would be serialised.

## 11. Practical notes

- Set `CUDA_DEVICE_ORDER=PCI_BUS_ID` when running several processes on a multi-GPU host so
  that `cuda:i` in the config is the same card `nvidia-smi` calls `i`. Pass device overrides
  as `models.sender.device=cuda:3`; the `--modes` flag takes a comma-separated list.
- Qwen3.5 (hybrid linear attention) runs on the PyTorch fallback without the `fla` kernels;
  prefill is fast enough for training, decoding is slow, which matches the write-up's premise
  that the large model should read, not write.
