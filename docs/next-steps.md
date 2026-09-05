# Next steps

Superseded in part by `HANDOFF.md` (2026-09-06), which has the current state and the ordered
plan. Summary of what changed:

- Step 1 of the original plan (sender shown the solution) ran: about 1.3 points of content
  crossed; the rest was style. The sharper answer-only version showed the bridge *does* encode
  the answer in its slots (probe R^2 0.74) while the receiver ignores it.
- Step 2 (hand-off-heavy training) ran: the collapse with k is gone, the latent curve is flat.

So the bottleneck is the last hop, bridge output -> frozen receiver, and the next steps target
it directly (see `HANDOFF.md`, section 4):

1. Oracle upper bound with the receiver's own post-solution state added *after* the prompt.
2. Post-prompt state injection (`injection: state_suffix`), larger gate, no RMS normalisation of
   the sender part; validate on answer-only targets first.
3. Repeat the capacity and hand-off-heavy runs with the working injection.
4. If the oracle fails on the RNN, run the same oracle on the transformer receiver with a KV
   prefix from its own cache to separate "RNN-specific" from "frozen-receiver-general".
