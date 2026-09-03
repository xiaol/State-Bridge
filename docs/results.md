# Results

Pair: **Qwen3.5-9B (sender, reads only) -> Qwen3-0.6B (receiver, writes)**, both frozen.
Benchmark: GSM8K test (1319 problems), greedy decoding, thinking disabled, answer = last
`\boxed{}`. Difficulty buckets by number of reasoning lines in the gold solution:
easy <= 3 (n = 697), medium 4-5 (n = 472), hard >= 6 (n = 150). Hardware: 4x A100 40 GB.
Config: `configs/qwen3p5_9b_to_qwen3_0p6b.yaml`.

## Baselines

| system | accuracy | easy | medium | hard | tokens written per answer |
|---|---|---|---|---|---|
| receiver alone (Qwen3-0.6B) | **0.623** | 0.753 | 0.508 | 0.380 | 250 |
| sender alone (Qwen3.5-9B) | **0.842** | 0.923 | 0.814 | 0.560 | 371 |

Gap to close: 21.9 points overall; 17.0 easy, 30.6 medium, 18.0 hard.

## Phase 1: gold targets, soft-token injection (negative result)

Bridges trained with the receiver's cross-entropy on the *gold* GSM8K rationales, slots
injected as soft tokens at the receiver's input embeddings (`bridge.injection: embed`).

| system | trainable params | final val loss | accuracy | easy | medium | hard |
|---|---|---|---|---|---|---|
| prompt-tuning control (64 constant slots) | 0.07M | 0.641 | 0.384 | 0.509 | 0.286 | 0.113 |
| resampler bridge, no residual base, gate 1.0 (v1) | 43M | 0.798 | not evaluated | | | |
| resampler bridge, residual base + gate 0.1 (v2) | 43M | 0.615 | 0.406 | 0.524 | 0.314 | 0.153 |
| v2 with **shuffled** sender states (control) | | | 0.406 | 0.528 | 0.307 | 0.147 |
| v2 with **mean** slots (control) | | | 0.401 | 0.521 | 0.311 | 0.127 |

Two lessons, both of which changed the design:

1. **Gold targets destroy the receiver.** Every system trained on the terse gold rationales
   writes about 120 tokens per answer instead of 250 and loses 20+ points against the untouched
   receiver, control included. The LM loss on gold text is not the quantity that matters; any
   "gap closed" measured this way is dominated by style, not by information. Phase 2 therefore
   trains on targets the models write themselves (receiver's own correct solutions, sender's
   solutions where the receiver fails).
2. **Soft tokens are too weak a channel into a frozen 0.6B model.** The v2 bridge reached a
   lower validation loss than the control, so it did use sender information to predict gold
   text, yet its accuracy with the *wrong* problem's sender state (shuffled) equalled its
   accuracy with the right one. The trained gate on the sender-dependent part sat at 0.096
   (initialised at 0.1): the optimiser found nothing at the embedding layer worth amplifying.
   Phase 2 injects the transferred state as key/value prefixes in every receiver layer.

## Phase 2: model-written targets, deep (key/value) injection

_(filled in from `runs/qwen3p5_9b_to_qwen3_0p6b_kv/` when the run completes)_

## Geometry

How much structure do the two frozen models already share?  For 256 GSM8K questions (raw
text, each model's own tokenizer), every layer of both models was mean-pooled over tokens.
Linear CKA and the 5-fold cross-validated R^2 of a ridge map from a sender layer to a receiver
layer (`geometry.py`):

## Linear CKA

| sender \ receiver | L0 | L4 | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|---|---|
| L0 | 0.91 | 0.14 | 0.15 | 0.15 | 0.15 | 0.19 | 0.37 | 0.59 |
| L4 | 0.60 | 0.34 | 0.34 | 0.35 | 0.35 | 0.38 | 0.59 | 0.71 |
| L8 | 0.57 | 0.31 | 0.31 | 0.32 | 0.32 | 0.36 | 0.61 | 0.77 |
| L12 | 0.54 | 0.34 | 0.34 | 0.35 | 0.35 | 0.39 | 0.62 | 0.74 |
| L16 | 0.54 | 0.37 | 0.37 | 0.38 | 0.38 | 0.42 | 0.63 | 0.69 |
| L20 | 0.54 | 0.25 | 0.25 | 0.25 | 0.25 | 0.30 | 0.56 | 0.77 |
| L24 | 0.53 | 0.18 | 0.18 | 0.19 | 0.19 | 0.24 | 0.51 | 0.78 |
| L28 | 0.54 | 0.16 | 0.16 | 0.16 | 0.16 | 0.21 | 0.49 | 0.80 |
| L32 | 0.53 | 0.12 | 0.12 | 0.12 | 0.12 | 0.17 | 0.46 | 0.89 |

## Ridge R^2 (5-fold CV) sender layer -> receiver layer

| sender \ receiver | L0 | L4 | L8 | L12 | L16 | L20 | L24 | L28 |
|---|---|---|---|---|---|---|---|---|
| L0 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 | 0.07 | 0.05 | 0.05 |
| L4 | 0.35 | 0.83 | 0.83 | 0.83 | 0.81 | 0.73 | 0.61 | 0.51 |
| L8 | 0.30 | 0.89 | 0.88 | 0.88 | 0.87 | 0.78 | 0.66 | 0.54 |
| L12 | 0.24 | 0.88 | 0.88 | 0.87 | 0.86 | 0.77 | 0.65 | 0.51 |
| L16 | 0.21 | 0.89 | 0.88 | 0.88 | 0.87 | 0.78 | 0.62 | 0.47 |
| L20 | 0.20 | 0.84 | 0.84 | 0.84 | 0.83 | 0.74 | 0.64 | 0.51 |
| L24 | 0.25 | 0.80 | 0.80 | 0.80 | 0.79 | 0.71 | 0.66 | 0.55 |
| L28 | 0.27 | 0.77 | 0.77 | 0.76 | 0.75 | 0.68 | 0.65 | 0.57 |
| L32 | 0.22 | 0.70 | 0.70 | 0.70 | 0.69 | 0.61 | 0.61 | 0.58 |

Receiver self-baseline, R^2 from its own embeddings to each layer: L0=0.11, L4=0.09, L8=0.10, L12=0.10, L16=0.10, L20=0.09, L24=0.08, L28=0.07

Most linearly translatable pair: sender L8 -> receiver L4 (R^2 = 0.89).

Reading: the sender's middle layers (L4-L16 of 32) linearly predict the receiver's early and
middle layers (L4-L16 of 28) with R^2 about 0.88, against 0.10 from the receiver's own token
embeddings, so most of what the receiver computes about a problem is already linearly readable
from the sender's state. The receiver's last layers are harder to reach linearly (R^2 0.5-0.6),
and CKA is highest between the two models' final layers (0.89). In the write-up's terms: some
alignment is there, and the question is how much work the translation takes; here a linear
map does most of it for the middle of the network and a non-linear bridge has to do the rest.

## Text vs latent hand-off

_(from `runs/qwen3p5_9b_to_qwen3_0p6b_kv/handoff.md`)_

## Observability

_(from `runs/qwen3p5_9b_to_qwen3_0p6b_kv/observe.md`)_

## Compute

_(from `runs/qwen3p5_9b_to_qwen3_0p6b_kv/compute.md`)_
