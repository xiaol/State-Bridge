# State Bridge

A replication of mostik.ai's write-up *Bridging models' internal states*
(https://mostik.ai/read-more): a small trained **bridge** carries the hidden states a frozen
large model builds while *reading* a problem into the input space of a frozen small model,
which then does all of the *writing*. No text passes between the models and neither model's
weights change. The bridge is the only part that learns.

```
                 frozen sender (reads only)                       frozen receiver (writes)
 prompt ──► sender tokenizer ──► transformer ──► hidden states ──► BRIDGE ──► K slots ─┐
                                                 at layers L        (trained)             ├─► [slots | prompt] ──► answer
 prompt ──► receiver tokenizer ──► input embeddings ─────────────────────────────────────┘
```

The write-up's claims and where they are tested here:

| claim in the write-up | this repository |
|---|---|
| A trained bridge passes latent state between two frozen models | `state_bridge/bridge.py`, `injection.py`, `train.py` |
| Large model reads only, small model writes; small model closes ~50% of the gap, +25% accuracy, up to 2x on hard subsets | `evaluate.py` (`receiver`, `sender`, `bridged` modes, per-difficulty buckets, gap-closure summary) |
| Latent hand-off beats text hand-off at every level of large-model compute | `handoff.py` (sweep over k sender tokens, both channels) |
| Bridged pair costs 2.5x less than an equivalent mid-sized model | `compute.py` (FLOPs model, log-linear size interpolation) |
| "We started with the geometry": how much structure two models share | `geometry.py` (layer-wise CKA and ridge R^2) |
| The channel as an observability surface: record, probe, intervene | `observe.py` (probes on sender state / slots / receiver state, steering interventions) |
| Strict test: gains must come through the channel, not from adapting the models | controls: `prompt_tuning` bridge, `bridged_shuffled`, `bridged_ablated` |

Details of every decision are in [`docs/design.md`](docs/design.md); paraphrased notes on the
source are in [`docs/source-notes.md`](docs/source-notes.md); the numbers from the run in this
repository are in [`docs/results.md`](docs/results.md).

## Scale of this replication

The original pairs GLM-5.2 (753B) with Qwen-3.5 (4B). This repository ships two configs at a
scale one machine can run:

| config | sender (reads) | receiver (writes) | note |
|---|---|---|---|
| `configs/qwen3p5_9b_to_qwen3_0p6b.yaml` | Qwen3.5-9B | Qwen3-0.6B | different generations and tokenizers (248k vs 152k vocab); hybrid linear-attention sender |
| `configs/qwen3_1p7b_to_0p6b.yaml` | Qwen3-1.7B | Qwen3-0.6B | same family, both small downloads |
| `configs/qwen3p5_9b_to_rwkv7_1p5b.yaml` | Qwen3.5-9B | **RWKV-7 G1 1.5B** (RNN) | cross-architecture: the sender's state is written into the RNN's recurrent state (`bridge.injection: state`); pure-PyTorch RWKV-7 + fused CUDA kernel included |
| `configs/smoke.yaml`, `configs/smoke_rwkv7.yaml` | tiny random | tiny random (transformer / RWKV-7) | offline CPU smoke tests, ~2 minutes |

Benchmark: GSM8K (train split for the bridge, test split for evaluation), greedy decoding,
thinking disabled, answer = last `\boxed{}`.

## Quick start

Requirements: Python 3.10+, PyTorch 2.1+, transformers 4.45+, datasets, numpy, pyyaml. No
other dependencies (a CUDA toolkit with `nvcc` and `ninja` is optional: it enables the fused
RWKV-7 kernel). Install in editable mode or just set `PYTHONPATH`:

```bash
pip install -e .            # or: export PYTHONPATH=$PWD
bash scripts/smoke.sh       # offline end-to-end check on tiny random models
pytest -q                   # unit + pipeline tests (~40 s)
```

Full replication for one pair (four GPUs were used; one 40 GB GPU per model is enough):

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
bash scripts/run_all.sh configs/qwen3p5_9b_to_qwen3_0p6b.yaml
```

or step by step:

```bash
CFG=configs/qwen3p5_9b_to_qwen3_0p6b.yaml
python -m state_bridge precompute -c $CFG --role receiver                # receiver writes train solutions
python -m state_bridge precompute -c $CFG --role sender --subset wrong    # sender writes where receiver failed
python -m state_bridge targets    -c $CFG                                 # -> runs/<run>/targets_train.jsonl
python -m state_bridge train    -c $CFG data.targets=runs/<run>/targets_train.jsonl            # bridge
python -m state_bridge train    -c $CFG data.targets=runs/<run>/targets_train.jsonl run_name=ctrl bridge.type=prompt_tuning
python -m state_bridge eval     -c $CFG --modes receiver,sender,bridged,bridged_shuffled,bridged_ablated
python -m state_bridge eval     -c $CFG run_name=ctrl --modes bridged
python -m state_bridge handoff  -c $CFG                                   # text vs latent hand-off sweep
python -m state_bridge geometry -c $CFG                                   # CKA / ridge R^2 between layers
python -m state_bridge observe  -c $CFG                                   # probes + steering interventions
python -m state_bridge compute  -c $CFG                                   # FLOPs report
```

Any config key can be overridden on the command line as `section.key=value`
(`models.sender.device=cuda:3`, `train.lr=1e-4`, `data.eval_limit=200`). Outputs land in
`runs/<run_name>/`: `bridge.pt`, `train_log.jsonl`, `eval_<mode>.jsonl/json`, `summary.md`,
`handoff.md`, `geometry.md`, `observe.md`, `compute.md`.

If huggingface.co is unreachable, set `hf_endpoint` in the config (or download the models and
`main/*.parquet` of GSM8K by other means and point `models.*.path` / `data.path` at them).

## How it works

**Sender side.** `SenderEncoder` runs the frozen sender's prefill with hidden states enabled,
takes the residual stream at the configured layers (default: 8th- and 4th-from-last blocks)
and concatenates them. The sender never decodes in the basic setting.

**Bridge.** `ResamplerBridge` (default): LayerNorm + projection of sender states, sinusoidal
positions, then `num_slots` learned queries pass through `depth` blocks of cross-attention over
sender tokens, self-attention, and an MLP; output projected to the receiver's hidden size and
rescaled so that at initialisation the slots have the RMS of the receiver's token embeddings.
`PerTokenBridge` maps every sender token to one slot with an MLP. Both use a *gated residual*
parametrisation: `slots = learned_constants + gate * f(sender)` with the gate initialised small
(`bridge.gate_init`, `residual_base`, `num_prefix`). The system therefore starts as plain
prompt tuning and learns sender-dependent deviations on top, which trains far better through a
frozen receiver than sender-dependent slots from scratch (see `docs/results.md`).
`PromptTuningBridge` is the control: same slots, learned constants, sender ignored.

**Receiver side.** Three injection modes. `bridge.injection: kv` (default for transformers):
the slots are projected into key/value prefixes for every receiver layer and placed in its
cache before the prompt, so each attention layer can read the transferred state directly
(prefix tuning with a sender-computed prefix). `bridge.injection: state` (RWKV-7 receivers):
each slot becomes a key/value pair per layer and head and the receiver's *initial recurrent
state* is set to the sum of their outer products, exactly the object the RNN accumulates when
it reads tokens; the RNN then reads the prompt and writes the answer from that state.
`bridge.injection: embed`: the slots are inserted into the receiver's input-embedding sequence
as soft tokens. The embedding channel proved too weak for a frozen 0.6B receiver (see
`docs/results.md`).

**RWKV-7 receiver.** `state_bridge/rwkv7.py` is a pure-PyTorch RWKV-7 that loads official
`.pth` checkpoints (any `models.receiver.path` ending in `.pth`), exposes the recurrent state
as an input and output, and includes the World tokenizer (`assets/rwkv_vocab_v20230424.txt`).
`state_bridge/wkv7_kernel.py` compiles a fused CUDA kernel for the recurrence with gradients
into the initial state on first use (about 75x faster than the Python loop for training); it
falls back to the loop when no CUDA toolchain is present.

**Training.** Cross-entropy of the frozen receiver on a target solution, answer tokens only.
Gradients flow through the frozen receiver into the slots and the bridge. AdamW, cosine
schedule, bf16 autocast. Targets are built by the models themselves (`precompute` for each
role, then `targets`): the receiver's own correct solutions, the sender's solutions where the
receiver fails, gold as a fallback. Training on the terse gold rationales alone shifts the
receiver's style and costs it ~25 points of accuracy, which would swamp any channel effect
(see `docs/design.md`, section 3).

**Evaluation modes.** `receiver` (alone), `sender` (alone), `bridged`, `bridged_shuffled`
(slots from a different problem), `bridged_ablated` (slots replaced by their mean). The
summary reports gap closed, relative uplift, and both per difficulty bucket (number of
reasoning lines in the gold solution).

## Results (Qwen3.5-9B -> Qwen3-0.6B, GSM8K, 4x A100, about one GPU-hour per bridge)

Full tables and discussion in [`docs/results.md`](docs/results.md). In short:

| system | GSM8K accuracy |
|---|---|
| receiver alone (Qwen3-0.6B) | 0.623 |
| prefix-tuning control (no sender) | 0.604 |
| bridge, deep key/value injection | 0.632 |
| same bridge, sender state from the wrong problem | 0.629 |
| same bridge, mean slots | 0.641 |
| sender alone (Qwen3.5-9B) | 0.842 |

Cross-architecture pair, the sender's state written into the RNN's initial recurrent state
(`configs/qwen3p5_9b_to_rwkv7_1p5b.yaml`):

| system | GSM8K accuracy |
|---|---|
| receiver alone (RWKV-7 G1 1.5B) | 0.530 |
| state-tuning control (constant initial state, no sender) | 0.549 |
| bridge, state injection | 0.560 |
| same bridge, sender state from the wrong problem | 0.556 |
| same bridge, mean state | 0.574 |

- **The headline gain does not reproduce at this scale, with either receiver.** The bridged
  receiver matches (transformer) or slightly beats (RNN, +3 points) the untouched receiver, but
  the controls match it too: shuffling or ablating the sender's state changes nothing, overall
  and on the problems only the sender solves. The RNN's gain is the learned constant initial
  state (state tuning), not instance-specific information from the sender.
- **Two design traps are documented and fixed.** Training on the terse gold rationales costs the
  receiver 20+ points regardless of bridge (phase 1), so targets are written by the models
  themselves. Soft-token injection at the embedding layer never learned to use the sender
  (its gate stayed at initialisation); deep key/value injection trains cleanly and preserves
  the receiver, but still finds nothing to transfer.
- **Geometry reproduces the write-up's premise.** Mean-pooled mid-layer states of the sender
  linearly predict the receiver's mid layers with R^2 about 0.88 (0.10 from the receiver's own
  embeddings): the two frozen models share most of their structure, so translation is cheap
  to find. What is missing is content in the sender's prefill that the receiver lacks.
- **Hand-off sweep reverses the write-up's curve.** When the sender first reasons for k tokens,
  text hand-off climbs toward the sender's accuracy (0.675 at k=0 to 0.844 at k=256) while
  latent hand-off through this bridge falls (0.656 to 0.525): the bridge was trained almost
  only on prefill states.
- **Observability works as a method and is decisive.** A linear probe reads the answer's
  magnitude from the sender's state (R^2 0.44-0.67) but barely from the translated slots
  (R^2 0.03-0.06) in the standard runs, and steering the slots changes nothing downstream.
- **The failure is localised to the last hop.** In a capacity test where the sender reads the
  answer and the receiver writes only `\boxed{N}`, the bridge does put the answer into its
  slots (probe R^2 0.74), yet the frozen receiver's accuracy and loss match a control that never
  saw the sender. `docs/HANDOFF.md` has the plan that targets this (post-prompt state
  injection, oracle upper bound).

## Repository layout

```
state_bridge/
  config.py      defaults, YAML loading, key=value overrides
  models.py      frozen model loading, chat prompts, SenderEncoder (prefill -> hidden states)
  bridge.py      ResamplerBridge / PerTokenBridge / PromptTuningBridge; KVHead (kv) and StateHead (RWKV state) injection heads
  injection.py   slots into the receiver's embedding sequence or KV cache; padding; labels; generate
  rwkv7.py       pure-PyTorch RWKV-7 with explicit recurrent state + World tokenizer (loads official .pth)
  wkv7_kernel.py JIT-compiled CUDA WKV-7 recurrence with initial-state gradients (cuda/wkv7_state.cu)
  data.py        GSM8K / synthetic / JSONL, answer extraction, difficulty buckets
  train.py       BridgeSystem (sender + bridge + receiver) and the training loop
  evaluate.py    eval modes, controls, summary with gap closure
  handoff.py     text vs latent hand-off sweep over sender compute
  geometry.py    layer-wise CKA and ridge R^2 between the two models
  observe.py     record / probe / intervene on the channel
  compute.py     FLOPs cost model and equivalent-model estimate
  precompute.py  sender-written solutions for hand-off-aware training
  cli.py         python -m state_bridge <command>
configs/         smoke*.yaml, qwen3_1p7b_to_0p6b.yaml, qwen3p5_9b_to_qwen3_0p6b.yaml, qwen3p5_9b_to_rwkv7_1p5b.yaml
scripts/         smoke.sh, run_all.sh, rwkv7_phase_{a,b}.sh, wait_gpu.sh, compare_runs.py, make_tiny_models.py
assets/          rwkv_vocab_v20230424.txt (RWKV World tokenizer vocabulary)
tests/           unit and pipeline tests on tiny random models (transformer and RWKV-7 receivers)
docs/            source-notes.md, design.md, results.md, runs/ (raw evaluation outputs)
```

## Differences from the original and known limits

- Scale: 9B -> 0.6B instead of 753B -> 4B. Same experiment shape, smaller absolute gap.
- Injection is into the receiver's attention cache (or embeddings); the receiver's own layers
  still process its own prompt tokens.
- The "equivalent mid-sized model" cost ratio interpolates accuracy in log-parameters between
  the two measured models; the original presumably measured a real mid-sized model.
- Training-time use of the channel (distillation, specialisation, several models training
  together) is described as in progress in the write-up and is not implemented here; the
  bridge already carries gradients, so unfreezing the receiver in `train.py` is the entry
  point.

## References

- mostik.ai, *Bridging models' internal states*, https://mostik.ai/read-more
- WIRED, first external account of the company,
  https://www.wired.com/story/russian-startup-mostik-ai-models-communication/
- Hanna and Ameisen, ICLR 2026, arXiv:2604.12493 (planning ahead in Qwen-3)
- Lindsey et al., 2025; Gurnee et al., 2026 (Anthropic interpretability; rhyme planning,
  Jacobian lens / J-space)
- Korbak et al., 2025, arXiv:2507.11473 (limits of chain-of-thought monitoring)

License: MIT.
