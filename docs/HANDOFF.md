# Handoff: State Bridge, status and plan (2026-09-06)

For the next agent. Everything below is reproducible from this repository; raw outputs of every
run are under `runs/` on the host (gitignored) and, for finished runs, copied to `docs/runs/`.

## 1. What this repository is

A replication of mostik.ai's write-up (https://mostik.ai/read-more): a small trained *bridge*
carries a frozen sender model's prefill hidden states into a frozen receiver model, which does
all the writing. The write-up claims the small model closes ~50% of the gap to the large one,
that latent hand-off beats text hand-off at every budget, and that the channel is a place to
probe and intervene. See `README.md`, `docs/design.md` (architecture, all decisions),
`docs/results.md` (all numbers), `docs/source-notes.md` (paraphrase of the write-up).

Pairs tested, all on GSM8K, all with the same sender (Qwen3.5-9B, 0.842 alone):

| receiver | injection | receiver alone | best bridge | control that matches it |
|---|---|---|---|---|
| Qwen3-0.6B (transformer) | key/value prefixes in every layer | 0.623 | 0.632 | mean-slot 0.641, shuffled 0.629 |
| RWKV-7 G1 1.5B (RNN) | initial recurrent state | 0.530 | 0.560 (0.571 after hand-off-heavy training) | mean-state 0.574, shuffled 0.556 / 0.564, state tuning 0.549 |

**Headline so far: no measurable instance-specific information crosses the channel in any
configuration.** Every accuracy gain is reproduced by a control with no sender (state/prefix
tuning) or with the wrong sender state (shuffled).

## 2. What the last three experiments established (this is the useful part)

### 2a. Capacity test: the sender reads the full solution (RNN receiver)

Sender reads question + worked solution, receiver sees only the question. Bridged 0.439,
shuffled 0.426, mean state 0.437 (reference bridge 0.560, receiver alone 0.530). The bridge
transmitted the *style* of the text the sender read (answers shrank from 218 to 93 tokens,
costing 12 points) and about 1.3 points of content. Corrected probes: answer magnitude is
linearly readable from the sender's state (R^2 0.67) and barely from the slots (R^2 0.06).

### 2b. Answer-only capacity test: can one number cross?

Receiver trained to write only `The final answer is \boxed{N}.`; sender reads the question and
that answer (gold solution at eval). Run dir `runs/qwen3p5_9b_to_rwkv7_1p5b_answer/`.

| system | accuracy | val loss |
|---|---|---|
| bridge (sender holds the answer) | 0.124 | 0.3455 |
| shuffled sender state | 0.116 | |
| mean state | 0.116 | |
| state-tuning control (no sender) | 0.127 | 0.3473 |

Probes on this run (300 problems, standardised ridge): answer magnitude R^2 = 0.70 in the sender
state, **0.74 in the translated slots**, 0.28 in the receiver's own prompt state.

So: **the bridge does put the answer into the slots when it is salient, and the receiver still
does not use it.** Validation loss is within 0.002 of the no-sender control on a target that is
essentially the number itself. This localises the failure to the last hop: `StateHead` ->
frozen RNN read-out. The resampler is not the bottleneck.

### 2c. Hand-off-heavy training (step 2 of the old plan)

Every training problem now has a sender-written solution (`gen_sender_train.*.jsonl`, 7,473
problems); 80% of training examples hand off after 1-256 sender tokens. Run dir
`runs/qwen3p5_9b_to_rwkv7_1p5b_handoff/`.

| sender tokens k | text hand-off | latent hand-off (old bridge) | latent hand-off (hand-off-heavy bridge) |
|---|---|---|---|
| 0 | 0.500 | 0.531 | 0.575 |
| 32 | 0.594 | 0.487 | 0.569 |
| 128 | 0.637 | 0.475 | 0.550 |
| 256 | 0.806 | 0.450 | 0.562 |

The collapse with k is gone (the bridge no longer breaks the receiver on partial-reasoning
states) but the latent curve is *flat*: the receiver does not benefit from the sender having
reasoned for 256 tokens, while text hand-off does (0.81). Same conclusion as 2b from a different
direction: the information is in the sender's state, the receiver is not reading it.

### 2d. Probe correction

The earlier "answer magnitude R^2 0.00 in the slots" lines used an unscaled ridge with a fixed
penalty; slots live at ~0.03 scale and were under-read. Probes now standardise features
(`observe.py`, test in `tests/test_data.py`) and save recorded features to
`observe_features.npz`. Re-probed numbers: standard RNN bridge slots R^2 0.03 (sender 0.44);
capacity bridge slots 0.06 (sender 0.67); answer-only bridge slots 0.74 (sender 0.70); transformer
kv bridge slots 0.11 (sender 0.44).

## 3. Why the receiver is not reading the state: two concrete hypotheses

1. **Injection point.** The state is written *before* the prompt. RWKV-7's per-channel decay
   is `exp(-exp(w))` with `w <= -0.5`, i.e. a factor of at most 0.55-1.0 per token; over a
   160-token prompt most of an initial-state perturbation is gone before generation starts.
   Transformer prefixes do not decay; RNN initial states do. The write-up's "prefill state"
   for an RNN should be added *after* the receiver has read the prompt, right before it writes.
2. **Magnitude.** The sender-dependent part of the state is RMS-normalised per layer/head and
   multiplied by a gate initialised at 0.1; trained gates end near 0.15 in every run. The
   receiver sees a ~15% perturbation of a state it then decays. The learned constant part
   (state tuning) is what carries the 2-point gain.

## 4. Recommended next steps, in order (all cheap, all on the RNN pair)

The answer-only test (2b) is the right harness for all of them: training takes ~15 min on one
A100 (targets are 10 tokens), evaluation ~5 min, and success is unambiguous: bridged far above
0.12 while shuffled stays at 0.12.

1. **Oracle upper bound, no bridge (30 min).** Let the *receiver itself* read question +
   solution, take its state at the end, then have it answer given only the question from that
   state (state added after the prompt). If even the receiver's own solution state does not
   raise answer-only accuracy above 0.12, post-prompt state injection cannot work with a frozen
   RWKV-7 and the plan changes (e.g. LoRA on the receiver's read-out, or a transformer
   receiver). Implementation: two-segment forward in `rwkv7.py` (prompt -> state; then generate
   from `state + delta`), ~40 lines; no training.
2. **Post-prompt state injection for the bridge (`injection: state_suffix`).** Run the receiver
   over the prompt from its base state, add the bridge's state delta, then decode / compute the
   loss on the target. Train on answer-only targets first, then full targets. Also try
   `kv_gate_init` 1.0 and no per-head RMS normalisation of the sender part (let the optimiser
   set the scale), since the current 0.1-gate perturbation is small by construction.
3. **Only if 2 works on answer-only:** repeat the full-solution capacity test, then the
   hand-off-heavy run with post-prompt injection. If latent hand-off then tracks text hand-off
   at k = 128-256, the write-up's central claim reproduces at this scale.
4. **If 1 fails:** the frozen RNN cannot exploit arbitrary state content; test the same oracle
   on the transformer receiver with a KV prefix built from the receiver's own cache after
   reading the solution (`Qwen3-0.6B`, `injection: kv`). That decides between "RNN-specific"
   and "frozen-receiver-general".

What not to do next: bigger senders, longer schedules of the current design, or thinking mode.
None of them addresses the last hop.

## 5. State of the host at handoff (2026-09-06 02:00 CST)

Nothing of this project is running. All runs are finished and folded into `docs/results.md`:

- Hand-off-heavy bridge at k = 0 on the full test set: bridged 0.571, shuffled 0.564; gap set
  0.352 vs 0.358. Best absolute RNN number, same verdict.
- Transformer (Qwen-kv) re-probe with standardised features: slots R^2 0.11 (sender 0.44,
  receiver's own prompt state 0.08).

Other sessions' jobs (`scripts/qwen35_27b_pipeline.sh` and RNN-StateTuning) share the GPUs;
check `nvidia-smi` before launching.

## 6. Operational notes (see also the memory file `env-network-and-gpus`)

- 4x A100 40 GB, shared with other sessions whose jobs take 20-40 GB per card and appear
  without warning. Use `scripts/wait_gpu.sh <min_free_mib>` and the `pick_pair` + OOM-retry
  pattern in `scripts/rwkv7_phase_c.sh`. Always `export CUDA_DEVICE_ORDER=PCI_BUS_ID
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
- huggingface.co and PyPI are unreachable. Models and data come from ModelScope via aria2c
  (`/tmp/mostik/ms_dl.sh`, and the URL pattern in the memory file). RWKV-7 G1 weights: repo
  `RWKV/rwkv7-g1`, local copies in `/root/x/models/rwkv7/`.
- Launch background jobs as `env -u CUDA_VISIBLE_DEVICES bash -c '(nohup ... &)'`; never
  `pkill -f` a pattern that appears in your own command line; never leave a `dl.py` on the
  import path (it shadows a module setuptools imports during CUDA extension builds).
- Git over SSH needs `core.sshCommand = env -u LD_LIBRARY_PATH /usr/bin/ssh` (already set).
- Tests: `PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES="" python -m pytest -q tests` (20 tests, ~3 min,
  CPU). Offline smoke: `bash scripts/smoke.sh` (transformer and RWKV-7 receivers).

## 7. Map of the code

- `state_bridge/bridge.py`: `ResamplerBridge` (sender states -> 64 slots), `KVHead` (slots ->
  per-layer key/value prefixes), `StateHead` (slots -> RWKV-7 initial state), `PromptTuningBridge`
  (control). `docs/design.md` section 1c has shapes and parameter counts.
- `state_bridge/rwkv7.py`: pure-PyTorch RWKV-7 with explicit state, World tokenizer, loads
  official `.pth`. `wkv7_kernel.py` + `cuda/`: fused kernel with gradients into the initial
  state (validated against the loop).
- `state_bridge/train.py`: `BridgeSystem` (sender + bridge + receiver, all injection modes),
  `train()`. `models.sender_sees_solution` is the capacity-test switch.
- `state_bridge/evaluate.py`, `handoff.py`, `observe.py`, `geometry.py`, `compute.py`,
  `precompute.py` (model-written targets), `scripts/compare_runs.py` (controls, gap set).
- Configs: `configs/qwen3p5_9b_to_rwkv7_1p5b*.yaml` (base, `_capacity`, `_answer`),
  `configs/qwen3p5_9b_to_qwen3_0p6b.yaml`. Chains: `scripts/rwkv7_*.sh`.
