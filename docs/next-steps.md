# Next steps

State of play (see `results.md`): two receivers of different architectures (Qwen3-0.6B with
key/value injection, RWKV-7 1.5B with state injection), one sender (Qwen3.5-9B), full controls.
The bridged receiver never beats its shuffled-sender or mean-state control, the hand-off curve
reverses the write-up's, and the probes show the sender's state carries the answer's magnitude
(R^2 0.45) while the translated slots do not (R^2 0.00). Two causes are confounded: at k = 0
the sender's prefill may hold nothing the receiver lacks, and the objective gives the bridge no
reason to use what it does hold.

Planned order, cheapest and most diagnostic first.

## 1. Capacity test: the sender is shown the answer

Let the sender read the question *and the gold solution* while the receiver sees only the
question (a `sender_sees_solution` switch on the sender prompt). The sender's state then
provably contains the answer and the bridge's only job is translation.

- Success: bridged far above shuffled on the gap set; slot probes recover answer magnitude.
- Failure: the bridge or the objective is broken and scale will not fix it.
- Cost: about 2 GPU-hours on the RNN pair (no new data; gold rationales exist).

## 2. Hand-off-heavy training

Generate sender solutions for the remaining training problems (about 3,700 of 7,473 lack
one), train with `data.handoff_prob: 1.0`, rerun the hand-off sweep. Text hand-off reaches
0.81 at k = 256, so the information is demonstrably in the sender's state there; a bridge
trained on those states either matches it or gives a clean negative.

- Cost: about 1.5 GPU-hours of sender generation plus one training run.

## 3. Revisit k = 0 with pressure on the channel

Only after 1 and 2: sender-distillation targets for every problem (not just where the receiver
fails), or a loss weighted on the final-answer tokens. Today 68% of targets are the receiver's
own text at 0.2 nats per token, so the bridge's validation loss ends within 0.001 of the
control's.

## Not planned

A larger sender or thinking mode. Neither addresses the two causes above, and both are
expensive. The evaluation rule stays as it is: bridged minus shuffled on the gap set, plus the
slot probes; overall accuracy alone has misled twice.
