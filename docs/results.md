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

Targets: 5014 receiver-written (its own correct solutions), 1553 sender-written (where the
receiver failed and the sender succeeded), 906 gold. Bridge: resampler, 64 slots, projected to
key/value prefixes in all 28 receiver layers (`bridge.injection: kv`), constant prefix
initialised from real activations, sender part gated; 2 epochs (910 steps, batch 16), about
25 minutes on two A100s. Hand-off-aware: half the examples with a sender-written solution hand
off after 1-256 sender tokens during training. Control: the same key/value prefix machinery with
constant slots and no sender (prefix tuning). Ablation: the same targets with soft-token
injection (`bridge.injection: embed`).

| system | trainable params | val loss | accuracy | easy | medium | hard | shuffled sender | mean slots |
|---|---|---|---|---|---|---|---|---|
| receiver alone | 0 | | 0.623 | 0.753 | 0.508 | 0.380 | | |
| prefix-tuning control (kv, no sender) | 59M | 0.213 | 0.604 | 0.733 | 0.492 | 0.360 | | |
| **bridge, kv injection** | 102M | 0.214 | **0.632** | 0.769 | 0.515 | 0.360 | 0.629 | 0.641 |
| bridge, soft-token injection | 43M | 0.275 | 0.612 | 0.729 | 0.517 | 0.367 | 0.609 | |
| sender alone | 0 | | 0.842 | 0.923 | 0.814 | 0.560 | | |

Where the channel would have to act (`scripts/compare_runs.py`): the *gap set* is the 340 test
problems the receiver gets wrong and the sender gets right; the *receiver-right set* is the 822
the receiver already solves.

| system | acc on gap set (n=340) | acc on receiver-right set (n=822) | answer equals sender's | answer equals receiver's |
|---|---|---|---|---|
| bridge, kv | 0.312 | 0.860 | 0.602 | 0.623 |
| bridge, kv, shuffled sender | 0.312 | 0.860 | 0.600 | 0.628 |
| bridge, kv, mean slots | 0.329 | 0.871 | 0.612 | 0.633 |
| prefix-tuning control | 0.247 | 0.848 | 0.575 | 0.627 |
| bridge, soft-token | 0.306 | 0.836 | 0.592 | 0.617 |
| bridge, soft-token, shuffled sender | 0.253 | 0.848 | 0.578 | 0.625 |
| receiver alone | 0.000 | 1.000 | 0.597 | 1.000 |

**Reading.** With model-written targets the receiver keeps its native style (252 tokens per
answer) and its accuracy, and the deep bridge edges the receiver by 0.9 points (4% of the gap,
+1.3% relative). But the controls take that away: the same bridge fed the *wrong* problem's
sender state scores 0.629, and fed the dataset-mean slots scores 0.641, higher than with the
real state. On the gap set the kv bridge and its shuffled control are identical (0.312). The
soft-token bridge is the only place with a hint of transfer: 0.306 vs 0.253 on the gap set
against its shuffled control (about 18 problems, roughly two standard errors), invisible in the
overall score. Answer agreement with the sender does not move (0.60 for every variant, 0.597
for the receiver alone).

So at this scale the headline claim does not reproduce: **no measurable information crosses the
channel in the read-only (k = 0) setting.** What the bridge does do is harmless: unlike the gold
targets of phase 1, it leaves the receiver's competence intact, and the prefix-tuning control
shows that even the constant prefix costs 2 points.

Plausible reasons, in the order we would test them next:

1. **What a 9B prefill holds about a GSM8K problem is mostly what the 0.6B already has.** The
   answer needs multi-step computation that happens during generation, not while reading. The
   write-up's sender is 753B and cites evidence that larger models hold more that the text never
   shows; the effect may simply need the gap to be in knowledge rather than in arithmetic. The
   hand-off sweep below (sender reasons for k tokens first) is the direct test.
2. **The objective gives the bridge little to explain.** 68% of targets are the receiver's own
   text, which it already predicts at 0.2 nats/token; the bridge's validation loss ends within
   0.001 of the control's. Targets written entirely by the sender (distillation), or a loss
   focused on the final-answer tokens, would put pressure on the channel.
3. **Budget.** 910 steps of a 100M-parameter bridge through a frozen receiver is small; the
   write-up does not state its budget.

## Compute

FLOPs per GSM8K example for this pair (prompt about 160 tokens, 160 generated, 64 slots), from
`compute.py`. Non-embedding parameters: sender 8.2B, receiver 0.44B.

| scenario | GFLOPs / example | vs receiver alone |
|---|---|---|
| receiver alone | 288 | x1.00 |
| sender alone | 4,455 | x15.5 |
| bridged (sender prefill only + receiver) | 2,568 | x8.9 |
| text hand-off (k = 0) | 2,509 | x8.7 |

The economics the write-up describes hold by construction: the bridged pair costs 1.7x less
than running the sender, because the sender only reads. The "2.5x cheaper than an equivalent
mid-sized model" claim needs a bridged accuracy well above the receiver's to be meaningful; with
a 0.9-point gain the interpolated equivalent model is barely larger than the receiver itself.

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

The sender reads the problem and writes k tokens of its own solution. *Text* hand-off: the
receiver continues from that text. *Latent* hand-off: the sender's hidden states over prompt +
k tokens cross the (phase-2 kv) bridge and the receiver writes the whole answer. 160 test
problems, greedy, `handoff.py`.

| sender tokens k | text hand-off | latent hand-off | delta |
|---|---|---|---|
| 0 | 0.675 | 0.656 | -1.9 pts |
| 32 | 0.644 | 0.525 | -11.9 pts |
| 128 | 0.744 | 0.494 | -25.0 pts |
| 256 | 0.844 | 0.525 | -31.9 pts |

The write-up reports the latent channel ahead at every budget, by up to 10 points, with the
largest margin at k = 0. Here the two channels tie at k = 0 (both are the receiver alone, give
or take) and diverge the other way as the sender reasons longer: text hand-off climbs toward
the sender's own accuracy (0.84), latent hand-off falls to about 0.5. The bridge was trained
almost entirely on prefill states (only about 16% of training examples handed off mid-solution),
so states over partial reasoning are out of distribution for it and it makes the receiver
worse than nothing. This is the experiment where a working channel would have to show itself,
because after 128-256 sender tokens the sender's state demonstrably contains the intermediate
results (the text channel proves it). A hand-off-heavy training schedule is the obvious next run.

## Observability

Three objects were recorded for 500 test problems: the sender's state (bridge input, mean
pooled), the translated slots (mean pooled), and the receiver's own last-layer state over the
prompt. The same linear probes (5-fold cross-validated) on each:

| representation | answer log-magnitude R^2 | difficulty probe acc (majority 0.78) | receiver-correct probe acc (majority 0.63) |
|---|---|---|---|
| sender state | 0.45 | 0.69 | 0.64 |
| translated slots | **0.00** | 0.72 | 0.67 |
| receiver's own prompt state | 0.20 | 0.71 | 0.63 |

Then an intervention: add a multiple of the probe direction for answer magnitude to every slot
and measure what the receiver writes (250 problems):

| alpha (x slot RMS) | mean log10 of predicted answer | accuracy |
|---|---|---|
| -2 | 1.804 | 0.632 |
| -1 | 1.788 | 0.632 |
| 0 | 1.816 | 0.648 |
| +1 | 1.777 | 0.672 |
| +2 | 1.783 | 0.648 |

This is the clearest diagnosis in the repository. The magnitude of the answer is linearly
readable from the sender's prefill state (R^2 0.45, more than twice what the receiver's own
reading of the prompt holds), and the bridge throws it away (R^2 0.00 in the slots). Steering
the slots along that direction moves nothing: the receiver's answers keep the same magnitude
and accuracy, i.e. the direction is neither present nor used. The write-up's proposal, that a
latent hand-off is a place to record, probe and intervene, works as a *method*: it told us in
one table that the channel is empty. The part of the proposal that needs a channel carrying
information could not be exercised.

## Cross-architecture pair: Qwen3.5-9B -> RWKV-7 G1 1.5B (RNN receiver)

Config `configs/qwen3p5_9b_to_rwkv7_1p5b.yaml`, `bridge.injection: state`: the sender's prefill
state is written into the RNN's initial recurrent state (design.md, section 1b). Same sender,
same GSM8K test prompts; the receiver uses the World chat format with a `<think>\n</think>`
prefix so it answers without a reasoning block.

| system | accuracy | easy | medium | hard | tokens per answer |
|---|---|---|---|---|---|
| receiver alone (RWKV-7 G1j 1.5B) | **0.530** | 0.637 | 0.441 | 0.313 | 184 |
| sender alone (Qwen3.5-9B) | **0.842** | 0.923 | 0.814 | 0.560 | 371 |

Gap to close: 31.2 points overall (28.6 easy, 37.3 medium, 24.7 hard), wider than for the
Qwen3-0.6B receiver.

Training targets, built the same way as for the transformer receiver: 4479 receiver-written
(its own correct solutions), 2075 sender-written (where the receiver failed and the sender
succeeded; the sender solves 87% of the problems the RNN gets wrong), 919 gold. Bridge:
resampler, 64 slots, `StateHead` into all 24 layers (`H=32`, `N=64` state per layer), constant
state initialised from a real prompt's state, sender part gated; 2 epochs (910 steps, batch 16),
32 minutes on one A100 for sender + receiver together (fused WKV-7 kernel). Validation loss
0.2148 after training. Control: the same constant-state machinery with no sender (state tuning),
validation loss 0.2150.

| system | trainable params | val loss | accuracy | easy | medium | hard | tokens per answer |
|---|---|---|---|---|---|---|---|
| receiver alone | 0 | | 0.530 | 0.637 | 0.441 | 0.313 | 184 |
| state-tuning control (constant initial state, no sender) | 3.4M | 0.2150 | 0.549 | 0.650 | 0.468 | 0.333 | 230 |
| **bridge, state injection** | 100M | 0.2148 | _(phase C)_ | | | | |
| sender alone | 0 | | 0.842 | 0.923 | 0.814 | 0.560 | 371 |

Unlike the transformer receiver, whose prefix-tuning control *lost* 2 points, the RNN's
constant learned initial state *gains* 1.9 points over the untouched model (state tuning is a
known effective adapter for RWKV). The bridge has to beat 0.549, not 0.530.

_(bridged evaluation with shuffled / mean-state controls, hand-off sweep and probes: filled in
from `runs/qwen3p5_9b_to_rwkv7_1p5b/` when phase C completes)_

**Geometry across architectures.** The same layer-wise analysis as above, transformer sender
against RNN receiver (256 questions, mean-pooled, 5-fold ridge R^2):

| sender \ receiver (RWKV-7, 24 layers) | L0 | L4 | L8 | L12 | L16 | L20 | L24 |
|---|---|---|---|---|---|---|---|
| L4 | 0.27 | 0.46 | 0.53 | 0.45 | 0.42 | 0.34 | 0.34 |
| L8 | 0.21 | 0.42 | **0.57** | 0.51 | 0.46 | 0.35 | 0.32 |
| L16 | 0.11 | 0.35 | 0.52 | 0.51 | 0.45 | 0.27 | 0.25 |
| L24 | 0.13 | 0.35 | 0.51 | 0.47 | 0.43 | 0.32 | 0.28 |
| L32 | 0.10 | 0.30 | 0.44 | 0.38 | 0.35 | 0.29 | 0.28 |

Receiver self-baseline (its own embeddings to each layer): 0.01-0.03. Linear CKA peaks at 0.88
between sender L12-L16 and receiver L12.

The two architectures share structure, but about a third less of it is linearly reachable than
between the two Qwen transformers (peak R^2 0.57 vs 0.89), and the best-aligned receiver depth
moves from the early layers to the middle (L8-L12 of 24). In the write-up's framing this is a
pair where "how much work the translation takes" is larger, so it is a fairer test of a
*trained* bridge than the same-family pair. Note that the RNN's mean-pooled residual stream is
not its state; the bridge writes into the WKV matrices, whose geometry this table does not
measure.

## Summary against the write-up's claims

| claim | outcome here (9B -> 0.6B, ~1 GPU-hour bridge) |
|---|---|
| bridged small model closes ~50% of the gap, +25% accuracy | not reproduced: +0.9 pts, controls equal or better |
| up to 2x uplift on hard subsets | not reproduced: hard bucket 0.360 vs 0.380 alone |
| latent hand-off beats text hand-off at every budget | reversed: latent worse by 12-32 pts for k > 0 |
| 2.5x cheaper than an equivalent mid-sized model | not applicable: no accuracy gain to price; bridged pair costs 8x the receiver |
| two models share exploitable structure ("we started with the geometry") | reproduced: linear R^2 0.88 mid-layer to mid-layer |
| a channel is a surface for observability | reproduced as a method: probes and interventions run and were decisive |

Honest reading: this repository reproduces the *machinery* and the *premise* of the write-up
and fails to reproduce its *result* at a scale roughly 80x smaller in sender size and with a
training budget the write-up does not state. The three levers most likely to change the
outcome are listed at the end of the phase-2 section.
