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
