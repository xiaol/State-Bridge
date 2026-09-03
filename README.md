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
| `configs/smoke.yaml` | tiny random | tiny random | offline CPU smoke test, ~1 minute |

Benchmark: GSM8K (train split for the bridge, test split for evaluation), greedy decoding,
thinking disabled, answer = last `\boxed{}`.

## Quick start

Requirements: Python 3.10+, PyTorch 2.1+, transformers 4.45+, datasets, numpy, pyyaml. No
other dependencies. Install in editable mode or just set `PYTHONPATH`:

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
python -m state_bridge train    -c $CFG                                   # bridge, both models frozen
python -m state_bridge train    -c $CFG run_name=ctrl bridge.type=prompt_tuning   # control
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
`PerTokenBridge` maps every sender token to one slot with an MLP. `PromptTuningBridge` is the
control: same slots, learned constants, sender ignored.

**Receiver side.** Slots are inserted into the receiver's input-embedding sequence
(`bridge.position: prefix` before the prompt, or `suffix` between prompt and answer). The
receiver attends to them like tokens it cannot read. Generation uses `generate(inputs_embeds=…)`.

**Training.** Cross-entropy of the frozen receiver on the gold solution, answer tokens only.
Gradients flow through the frozen receiver into the slots and the bridge. AdamW, cosine
schedule, bf16 autocast. Optionally the sender's own solutions are precomputed
(`python -m state_bridge precompute`) so training also hands off after a random number of
sender-written tokens.

**Evaluation modes.** `receiver` (alone), `sender` (alone), `bridged`, `bridged_shuffled`
(slots from a different problem), `bridged_ablated` (slots replaced by their mean). The
summary reports gap closed, relative uplift, and both per difficulty bucket (number of
reasoning lines in the gold solution).

## Results

See [`docs/results.md`](docs/results.md) for the tables produced by the run in this repository
and a comparison with the numbers claimed in the write-up.

## Repository layout

```
state_bridge/
  config.py      defaults, YAML loading, key=value overrides
  models.py      frozen model loading, chat prompts, SenderEncoder (prefill -> hidden states)
  bridge.py      ResamplerBridge / PerTokenBridge / PromptTuningBridge, save/load
  injection.py   slots into the receiver's embedding sequence; padding; labels; generate
  data.py        GSM8K / synthetic / JSONL, answer extraction, difficulty buckets
  train.py       BridgeSystem (sender + bridge + receiver) and the training loop
  evaluate.py    eval modes, controls, summary with gap closure
  handoff.py     text vs latent hand-off sweep over sender compute
  geometry.py    layer-wise CKA and ridge R^2 between the two models
  observe.py     record / probe / intervene on the channel
  compute.py     FLOPs cost model and equivalent-model estimate
  precompute.py  sender-written solutions for hand-off-aware training
  cli.py         python -m state_bridge <command>
configs/         smoke.yaml, qwen3_1p7b_to_0p6b.yaml, qwen3p5_9b_to_qwen3_0p6b.yaml
scripts/         smoke.sh, run_all.sh, make_tiny_models.py
tests/           unit and pipeline tests on tiny random models
docs/            source-notes.md, design.md, results.md
```

## Differences from the original and known limits

- Scale: 9B -> 0.6B instead of 753B -> 4B. Same experiment shape, smaller absolute gap.
- Injection is at the receiver's input-embedding layer; the write-up's "prefill state" could
  also mean per-layer KV injection. `injection.py` is the single place to add that.
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
