# Source notes: mostik.ai, "Bridging models' internal states"

Source: https://mostik.ai/read-more (published 2 September 2026, retrieved 3 September 2026).
Company site: https://mostik.ai. First external account: WIRED,
https://www.wired.com/story/russian-startup-mostik-ai-models-communication/

These are paraphrased notes on the claims the repository sets out to replicate, not a copy of
the page.

## The argument

- A language model builds on the order of a hundred hidden vectors (about a million numbers,
  roughly two megabytes) to emit one token, then keeps only the token: about seventeen bits
  for a vocabulary of about 150,000. Every multi-model system (subagents, councils, routers)
  communicates through those seventeen bits.
- Interpretability evidence that the discarded state carries real information:
  - Hanna and Ameisen, ICLR 2026, arXiv:2604.12493: Qwen-3 represents an upcoming noun
    ("accountant") several tokens early, and that representation drives the article "an";
    the effect grows with model size.
  - Lindsey et al., 2025 (Anthropic): Claude 3.5 Haiku settles on a rhyme before writing the
    line.
  - Gurnee et al., 2026 (Anthropic): the "Jacobian lens" identifies concepts the model is
    poised to say ("J-space"); a model can silently hold a concept there while its output is
    unchanged.
- Scaling pre-training and single-agent test-time compute are both flattening; the next axis
  is models coordinating with each other, which needs efficient communication.

## What they built

- A small trained bridge hands one model's hidden states to another; the receiver works with
  them directly. No output text passes between the models; neither model's weights change.
  The bridge is the only new part. ("Mostik" means "little bridge".)
- They started with geometry: how much structure two models' representation spaces already
  share. Alignment does not increase on its own with capability, so a bridge has to be built;
  some alignment exists and varies predictably. The question is how much work a translation
  takes, not whether it exists.

## Headline result (large to small)

- Sender GLM-5.2 (753B), receiver Qwen-3.5 (4B). Chosen because deployments want a large
  model that knows things and a cheap model that writes.
- Economics: writing is token-by-token and expensive; reading is one parallel pass and cheap.
  So the large model only reads the problem, its hidden states cross the bridge, and it stops.
  The small model does all generation.
- With the bridge the small model closes 50% of the gap to the large model, lifting its own
  accuracy by 25%. On harder subsets, where the gap is larger, the uplift reaches 2x.
- A mid-sized model that would score what the bridged pair scores would cost 2.5x more compute
  to run.
- Compared against standard ways of combining models, the bridged system is a Pareto
  improvement.

## Text hand-off vs latent hand-off

- The large model works through part of a problem, then hands off to the small model. Text
  hand-off passes the text written so far; latent hand-off passes the internal state built up
  while reasoning.
- Latent hand-off wins at every level of large-model compute tested, by up to 10 percentage
  points. The advantage is largest when hand-off happens before the large model has written
  anything: text has nothing to pass, the latent state already holds what the model got from
  reading the problem. Both converge to large-model performance as its compute grows.

## Both models frozen

- The bridge is trained separately; both models are completely frozen. This is a strict test:
  if information passes without either model adapting to the channel, the bridge is exploiting
  structure that was already there.
- Practical consequence: an "advisor" pattern where a strong model reads the context and
  writes guidance for a small model pays twice (read, then write). The bridge removes the
  second payment: the advisor reads and stops; its guidance crosses as latent state.

## Observability

- Chain-of-thought monitoring only sees reasoning put into words (Korbak et al., 2025,
  arXiv:2507.11473 note some misbehaviour passes through undetected).
- A latent hand-off exposes three objects: the sender's hidden state, its translated form,
  and the receiver's behaviour. Each can be recorded; one can test what information survives
  translation and intervene on the transferred state to see what changes downstream.
  Intervention is the important part: if changing a feature in the channel predictably
  changes the receiver's behaviour, the feature was being used, not merely present.
- Not yet shown conclusively; framed as a surface on which it becomes possible.

## Where it goes (in progress)

- The bridge is trained, so gradients run through it, so it can be present while a model is
  still being trained.
- Distillation: teacher supervision arrives inside the student's generation rather than at
  the output layer; early signals suggest the student stays closer to the teacher for the
  same budget.
- Specialisation: small models pick up narrow skills more efficiently with the channel open
  and keep their general ability; a way of extending a large model by attaching modules.
- The mechanism is not particular to a pair: several models can train together.

## Deployment shapes described

- Low-latency serving: a stronger model contributes prefill state at selected points; the
  smaller model handles decoding.
- Batch-like latency: latent state transmitted between independently deployed endpoints, so
  the small model can run on older or underused GPUs.
- Integration point for councils, advisors, escalation paths, subagents: keep the workflow,
  replace the text passed between models with latent state.

## Team and backing (for context only)

15 people, 12 PhDs and a Fields Medalist, started four months before the post; backed by
General Catalyst, Foundation Capital and others.
