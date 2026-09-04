"""RWKV-7 receiver: tokenizer, padding-inert recurrence, state carry, and the state bridge head."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from state_bridge.bridge import StateHead, build_bridge
from state_bridge.rwkv7 import Rwkv7ForCausalLM, Rwkv7State, WorldTokenizer

ROOT = Path(__file__).resolve().parent.parent
VOCAB = ROOT / "assets" / "rwkv_vocab_v20230424.txt"


def tiny_rwkv(seed=0):
    torch.manual_seed(seed)
    cfg = SimpleNamespace(model_type="rwkv7", hidden_size=64, num_hidden_layers=2, vocab_size=300, head_size=16, num_heads=4, dim_ffn=128,
                          d_decay=16, d_aaa=16, d_mv=16, d_gate=32, num_attention_heads=4, num_key_value_heads=4, head_dim=16, tie_word_embeddings=False)
    m = Rwkv7ForCausalLM(cfg)
    for p in m.parameters():
        torch.nn.init.normal_(p, std=0.2)
    with torch.no_grad():
        for blk in m.blocks:
            blk.att.w0.fill_(-1.0)
    return m.eval()


@pytest.mark.skipif(not VOCAB.exists(), reason="World vocab not present")
def test_world_tokenizer_roundtrip_and_chat():
    tok = WorldTokenizer(str(VOCAB), chat_prefix=" <think>\n</think>\n")
    s = "Natalia sold 48 clips. 48/2 = 24.\n\nThe final answer is \\boxed{72}."
    ids = tok.encode(s)
    assert tok.decode(ids) == s and all(0 < i < len(tok) for i in ids)
    enc = tok([s, "hi"], padding=True)
    assert enc["input_ids"].shape[0] == 2 and enc["attention_mask"][1].sum() == len(tok.encode("hi"))
    assert tok.apply_chat_template([{"role": "user", "content": "Q?"}], add_generation_prompt=True) == "User: Q?\n\nAssistant: <think>\n</think>\n"


def test_right_padding_is_inert_and_state_carries():
    m = tiny_rwkv()
    a = torch.randint(1, 300, (1, 9))
    b = torch.randint(1, 300, (1, 5))
    ids = torch.zeros(2, 9, dtype=torch.long)
    ids[0] = a[0]; ids[1, :5] = b[0]
    mask = torch.tensor([[1] * 9, [1] * 5 + [0] * 4])
    with torch.no_grad():
        out = m(input_ids=ids, attention_mask=mask)
        single = m(input_ids=b)
        # logits of the short row at its real positions match the unpadded run
        assert torch.allclose(out.logits[1, :5], single.logits[0], atol=1e-4)
        # its final state equals the unpadded final state (pads did not touch it)
        assert torch.allclose(out.state.wkv[:, 1], single.state.wkv[:, 0], atol=1e-4)
        assert torch.allclose(out.state.att_x[:, 1], single.state.att_x[:, 0], atol=1e-4)
        # carrying the state is equivalent to processing the full sequence
        full = m(input_ids=a).logits[0, -1]
        st = m(input_ids=a[:, :6]).state
        cont = m(input_ids=a[:, 6:], state=st).logits[0, -1]
        assert torch.allclose(full, cont, atol=1e-4)


def test_state_head_shapes_and_gradients():
    m = tiny_rwkv()
    head = StateHead(d=12, num_layers=2, num_heads=4, head_size=16, hidden=64, num_slots=5, gate_init=0.1)
    tok = SimpleNamespace(__call__=None)
    receiver = SimpleNamespace(model=m, device=torch.device("cpu"), tokenizer=lambda text, return_tensors=None: {"input_ids": torch.randint(1, 300, (1, 7)), "attention_mask": torch.ones(1, 7, dtype=torch.long)})
    head.calibrate(receiver, "x")
    assert head.calib.min() > 0 and head.base.abs().sum() > 0
    slots = torch.randn(3, 5, 12, requires_grad=True)
    att, S, ffn = head(slots)
    assert att.shape == (2, 3, 64) and S.shape == (2, 3, 4, 16, 16) and ffn.shape == (2, 3, 64)
    ids = torch.randint(1, 300, (3, 6))
    labels = ids.clone(); labels[:, :2] = -100
    out = m(input_ids=ids, labels=labels, state=Rwkv7State(att, S, ffn))
    assert torch.isfinite(out.loss)
    out.loss.backward()
    assert slots.grad is not None and slots.grad.abs().sum() > 0 and head.proj.weight.grad.abs().sum() > 0


@pytest.mark.skipif(not VOCAB.exists(), reason="World vocab not present")
def test_rwkv7_receiver_end_to_end(tiny_cfg, tiny_dir):
    """Transformer sender -> state bridge -> RWKV-7 receiver: train a few steps, then evaluate
    receiver-alone, bridged and both controls on the CPU."""
    import copy

    from state_bridge.evaluate import evaluate
    from state_bridge.train import train

    cfg = copy.deepcopy(tiny_cfg)
    cfg["run_name"] = "pytest_rwkv7"
    cfg["models"]["receiver"] = {"path": str(tiny_dir / "receiver_rwkv7.pth"), "tokenizer": str(VOCAB), "chat_prefix": "", "device": "cpu", "dtype": "float32"}
    cfg["bridge"].update({"injection": "state", "num_slots": 4})
    cfg["train"]["max_steps"] = 3
    cfg["data"].update({"train_limit": 48, "val_size": 8, "eval_limit": 4})
    path = train(cfg)
    assert path.exists()
    cfg["eval"].update({"modes": ["receiver", "bridged", "bridged_shuffled", "bridged_ablated"], "max_new_tokens": 6, "batch_size": 4})
    sums = evaluate(cfg)
    assert set(sums) == {"receiver", "bridged", "bridged_shuffled", "bridged_ablated"}
    assert all(s["n"] == 4 for s in sums.values())


def test_build_bridge_state_injection():
    b = build_bridge({"type": "resampler", "injection": "state", "num_slots": 4, "d_model": 16, "depth": 1, "heads": 4, "dropout": 0.0, "residual_base": True},
                     in_dim=8, out_dim=12, target_rms=1.0, state_dims=(2, 4, 16, 64))
    assert b.state_head is not None and b.kv_head is None and b.state_head.K == 4
    ctrl = build_bridge({"type": "prompt_tuning", "injection": "state", "num_slots": 4}, in_dim=8, out_dim=12, target_rms=1.0, state_dims=(2, 4, 16, 64))
    slots, _ = ctrl(None, torch.ones(2, 1))
    att, S, ffn = ctrl.state_head(slots)
    assert S.shape == (2, 2, 4, 16, 16)
