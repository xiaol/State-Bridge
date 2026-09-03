from state_bridge.data import difficulty_bucket, extract_answer, is_correct, load_synthetic, summarize_accuracy


def test_extract_boxed_and_fallback():
    assert extract_answer("so \\boxed{1,234}.") == 1234.0
    assert extract_answer("Total = $18.") == 18.0
    assert extract_answer("\\boxed{\\frac{1}{2}}") == 2.0  # falls back to the last number inside
    assert extract_answer("no numbers here") is None
    assert extract_answer("first 3 then \\boxed{7} then 9") == 7.0


def test_is_correct_tolerance():
    assert is_correct(18.0, "18")
    assert is_correct(1000.00001, "1,000")
    assert not is_correct(17.0, "18")
    assert not is_correct(None, "18")


def test_synthetic_examples_are_consistent():
    ex = load_synthetic("train", 5)
    assert len(ex) == 5
    for e in ex:
        assert is_correct(extract_answer(e.solution), e.answer)
        assert "\\boxed{" in e.user_prompt


def test_summary_buckets():
    rows = [{"correct": True, "n_steps": 2}, {"correct": False, "n_steps": 6}]
    s = summarize_accuracy(rows)
    assert s["acc"] == 0.5 and s["n"] == 2
    assert difficulty_bucket(2) in s["buckets"] and difficulty_bucket(6) in s["buckets"]


def test_apply_targets_and_build_targets(tmp_path):
    import json

    from state_bridge.data import apply_targets
    from state_bridge.precompute import build_targets

    ex = load_synthetic("train", 4)
    # receiver right on 0, wrong on 1..3; sender right on 1, wrong on 2; nothing for 3
    (tmp_path / "gen_receiver_train.0of1.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"id": ex[0].id, "text": "r0 \\boxed{1}", "correct": True},
        {"id": ex[1].id, "text": "r1", "correct": False},
        {"id": ex[2].id, "text": "r2", "correct": False},
        {"id": ex[3].id, "text": "r3", "correct": False},
    ]) + "\n")
    (tmp_path / "gen_sender_train.0of1.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"id": ex[1].id, "text": "s1 \\boxed{2}", "correct": True},
        {"id": ex[2].id, "text": "s2", "correct": False},
    ]) + "\n")
    cfg = {"runs_dir": str(tmp_path.parent), "run_name": tmp_path.name, "seed": 0, "data": {"name": "synthetic", "train_limit": 4}}
    path = build_targets(cfg)
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(path)}
    assert rows[ex[0].id]["source"] == "receiver" and rows[ex[1].id]["source"] == "sender"
    assert rows[ex[2].id]["source"] == "gold" and rows[ex[3].id]["source"] == "gold"
    apply_targets(ex, str(path))
    assert ex[0].solution == "r0 \\boxed{1}" and ex[1].solution == "s1 \\boxed{2}" and "\\boxed" in ex[2].solution
