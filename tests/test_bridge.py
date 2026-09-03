import torch

from state_bridge.bridge import PerTokenBridge, PromptTuningBridge, ResamplerBridge, build_bridge, load_bridge, save_bridge


def test_resampler_shapes_and_mask():
    b = ResamplerBridge(in_dim=32, out_dim=16, num_slots=5, d_model=24, depth=1, heads=4, target_rms=0.5)
    h = torch.randn(3, 7, 32)
    m = torch.tensor([[1] * 7, [1] * 4 + [0] * 3, [1] * 2 + [0] * 5])
    slots, sm = b(h, m)
    assert slots.shape == (3, 5, 16) and sm.shape == (3, 5) and sm.all()
    # padded sender positions must not influence the output
    h2 = h.clone(); h2[1, 4:] = 100.0
    slots2, _ = b(h2, m)
    assert torch.allclose(slots[1], slots2[1], atol=1e-5)


def test_output_scale_matches_target_rms_at_init():
    b = ResamplerBridge(in_dim=32, out_dim=64, num_slots=4, d_model=24, depth=1, heads=4, target_rms=0.02)
    slots, _ = b(torch.randn(2, 6, 32), torch.ones(2, 6))
    rms = slots.pow(2).mean(-1).sqrt().mean().item()
    assert abs(rms - 0.02) / 0.02 < 0.05


def test_per_token_and_prompt_tuning():
    m = torch.tensor([[1, 1, 0], [1, 1, 1]])
    pt = PerTokenBridge(in_dim=8, out_dim=6, d_model=12)
    slots, sm = pt(torch.randn(2, 3, 8), m)
    assert slots.shape == (2, 3, 6) and sm.tolist() == m.bool().tolist()
    ctrl = PromptTuningBridge(in_dim=8, out_dim=6, num_slots=3)
    assert not ctrl.uses_sender
    slots, sm = ctrl(None, torch.ones(4, 1))
    assert slots.shape == (4, 3, 6)


def test_save_and_load_roundtrip(tmp_path):
    cfg = {"bridge": {"type": "resampler", "num_slots": 3, "d_model": 16, "depth": 1, "heads": 4, "dropout": 0.0, "position": "prefix"}, "models": {"sender_layers": [-1]}}
    b = build_bridge(cfg["bridge"], in_dim=10, out_dim=12, target_rms=1.0)
    save_bridge(b, cfg, tmp_path / "b.pt")
    b2, cfg2 = load_bridge(tmp_path / "b.pt")
    assert cfg2 == cfg
    h, m = torch.randn(2, 5, 10), torch.ones(2, 5)
    assert torch.allclose(b(h, m)[0], b2(h, m)[0])
