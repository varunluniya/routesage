from routesage.core import (
    classify_zero_shot, classify_one_shot, classify_few_shot, approx_token_count,
    ZERO_SHOT_PROMPT, ONE_SHOT_PROMPT, FEW_SHOT_PROMPT,
)


def test_zero_shot_is_unparseable_prose():
    r = classify_zero_shot()
    assert r.parseable is False
    assert "Technical" in r.raw_text
    assert "Critical" in r.raw_text


def test_one_shot_fixes_format_but_category_drifts():
    r = classify_one_shot()
    assert r.category == "Billing"
    assert r.urgency == "High"


def test_few_shot_strict_json_shape_and_judgment():
    r = classify_few_shot()
    assert r.category == "Billing"
    assert r.urgency == "Critical"
    assert r.escalate_to_oncall is True


def test_few_shot_urgency_escalates_past_one_shot():
    one = classify_one_shot()
    few = classify_few_shot()
    severity = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
    assert severity[few.urgency] > severity[one.urgency]


def test_token_counts_grow_monotonically():
    z = approx_token_count(ZERO_SHOT_PROMPT)
    o = approx_token_count(ONE_SHOT_PROMPT)
    f = approx_token_count(FEW_SHOT_PROMPT)
    assert z < o < f


def test_few_shot_grows_substantially_more_than_one_shot_step():
    z = approx_token_count(ZERO_SHOT_PROMPT)
    o = approx_token_count(ONE_SHOT_PROMPT)
    f = approx_token_count(FEW_SHOT_PROMPT)
    zero_to_one_growth = (o - z) / z
    one_to_few_growth = (f - o) / o
    assert one_to_few_growth > zero_to_one_growth


def test_approx_token_count_basic():
    assert approx_token_count("hello, world!") == 4
