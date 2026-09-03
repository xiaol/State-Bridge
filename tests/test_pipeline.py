"""End-to-end checks on tiny random models: injection correctness, training, evaluation."""

import torch

from state_bridge.injection import IGNORE, build_receiver_batch, encode_prompts, encode_targets, generate
from state_bridge.models import load_model


def test_injection_labels_and_padding(tiny_cfg):
    r = load_model(tiny_cfg["models"]["receiver"]["path"], "cpu", "float32")
    prompts = encode_prompts(r, [r.chat_prompt("What is 2+2?"), r.chat_prompt("A much longer question about apples and coins?")])
    targets = encode_targets(r, ["4", "The final answer is \\boxed{7}."], max_tokens=8)
    slots = torch.randn(2, 3, r.hidden_size)
    mask = torch.ones(2, 3, dtype=torch.bool)
    for pos in ("prefix", "suffix"):
        rb = build_receiver_batch(r, prompts, slots, mask, targets, pos, pad_left=False)
        B, L, d = rb.inputs_embeds.shape
        assert B == 2 and d == r.hidden_size and L == max(len(p) + 3 + len(t) for p, t in zip(prompts, targets))
        for i in range(2):
            n_lab = int((rb.labels[i] != IGNORE).sum())
            assert n_lab == len(targets[i]) and rb.labels[i, rb.lengths[i] - 1] == r.tokenizer.eos_token_id
            assert rb.attention_mask[i].sum() == rb.lengths[i]
    # left padding for generation
    rb = build_receiver_batch(r, prompts, slots, mask, None, "prefix", pad_left=True)
    assert rb.attention_mask[0, 0] == 0 or rb.lengths[0] == rb.lengths[1]
    assert rb.attention_mask[:, -1].all()


def test_left_padded_generation_matches_unpadded(tiny_cfg):
    r = load_model(tiny_cfg["models"]["receiver"]["path"], "cpu", "float32")
    prompts = encode_prompts(r, [r.chat_prompt("short?"), r.chat_prompt("a considerably longer prompt with many more tokens in it?")])
    slots = torch.randn(2, 4, r.hidden_size) * r.embedding_rms
    mask = torch.ones(2, 4, dtype=torch.bool)
    batched = generate(r, build_receiver_batch(r, prompts, slots, mask, None, "prefix", pad_left=True), 8)
    single = generate(r, build_receiver_batch(r, prompts[:1], slots[:1], mask[:1], None, "prefix", pad_left=True), 8)
    assert batched[0] == single[0]


def test_train_then_eval_smoke(tiny_cfg):
    import copy, json

    from state_bridge.evaluate import evaluate
    from state_bridge.train import train

    cfg = copy.deepcopy(tiny_cfg)
    cfg["run_name"] = "pytest_smoke"
    cfg["train"]["max_steps"] = 6
    cfg["data"]["train_limit"] = 64
    cfg["data"]["val_size"] = 16
    cfg["data"]["eval_limit"] = 8
    path = train(cfg)
    assert path.exists()
    log = [json.loads(l) for l in open(path.parent / "train_log.jsonl")]
    assert all(torch.isfinite(torch.tensor(r.get("loss", 0.0))) for r in log)
    cfg["eval"]["modes"] = ["receiver", "bridged", "bridged_shuffled"]
    cfg["eval"]["max_new_tokens"] = 6
    sums = evaluate(cfg)
    assert set(sums) == {"receiver", "bridged", "bridged_shuffled"}
    assert (path.parent / "summary.md").exists()


def test_prompt_tuning_control_trains_without_sender(tiny_cfg):
    import copy

    from state_bridge.train import train

    cfg = copy.deepcopy(tiny_cfg)
    cfg["run_name"] = "pytest_pt"
    cfg["bridge"]["type"] = "prompt_tuning"
    cfg["train"]["max_steps"] = 2
    cfg["data"]["train_limit"] = 32
    cfg["data"]["val_size"] = 8
    assert train(cfg).exists()


def test_kv_prefix_generation_matches_unpadded_and_trains(tiny_cfg):
    """Deep injection: left-padded greedy decoding with a KV prefix equals the unpadded result,
    the loss is finite, and gradients reach the bridge through the frozen receiver."""
    import copy

    from state_bridge.bridge import build_bridge
    from state_bridge.injection import forward_with_prefix, greedy_generate_with_prefix

    r = load_model(tiny_cfg["models"]["receiver"]["path"], "cpu", "float32")
    rc = r.model.config
    kv_dims = (rc.num_hidden_layers, rc.num_key_value_heads, rc.head_dim)
    bcfg = dict(copy.deepcopy(tiny_cfg["bridge"]), injection="kv")
    bridge = build_bridge(bcfg, in_dim=12, out_dim=r.hidden_size, target_rms=r.embedding_rms, kv_dims=kv_dims)
    bridge.kv_head.calibrate(r, r.chat_prompt("2+2?"))
    assert bridge.kv_head.calib.min() > 0
    torch.manual_seed(0)
    slots, mask = bridge(torch.randn(2, 5, 12), torch.ones(2, 5))
    kv = bridge.kv_head(slots)
    assert kv.shape == (kv_dims[0], 2, 2, kv_dims[1], slots.shape[1], kv_dims[2])
    prompts = encode_prompts(r, [r.chat_prompt("short?"), r.chat_prompt("a considerably longer prompt with many more tokens in it?")])
    rb = build_receiver_batch(r, prompts, None, None, None, "prefix", pad_left=True)
    batched = greedy_generate_with_prefix(r, kv, mask, rb.inputs_embeds, rb.attention_mask, 8)
    rb1 = build_receiver_batch(r, prompts[:1], None, None, None, "prefix", pad_left=True)
    single = greedy_generate_with_prefix(r, kv[:, :, :1], mask[:1], rb1.inputs_embeds, rb1.attention_mask, 8)
    assert batched[0] == single[0]
    targets = encode_targets(r, ["4", "7"], 4)
    rb = build_receiver_batch(r, prompts, None, None, targets, "prefix", pad_left=False)
    out = forward_with_prefix(r, kv, mask, rb.inputs_embeds, rb.attention_mask, rb.labels)
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert bridge.kv_head.proj.weight.grad is not None and bridge.kv_head.proj.weight.grad.abs().sum() > 0
    assert bridge.in_proj.weight.grad is not None
