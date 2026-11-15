# RouteSage

A small, tested demo of the zero-shot -> one-shot -> few-shot escalation
pattern for support-ticket triage: classify a ticket by category and
urgency, with few-shot adding a strict `escalate_to_oncall` routing flag a
client's downstream system depends on.

**Disclosure:** no live LLM call is made anywhere in this package. Each
`classify_*` function is a deterministic stand-in, chosen to be
representative of what an instruction-tuned model plausibly produces for
that exact prompt shape. See `routesage/core.py` docstrings. Token counts
are a regex word/punctuation-level approximation, not a real tokenizer --
useful only for comparing relative growth across the three versions.

## Usage

```bash
pip install -e .
python -m routesage.cli
```

## What it demonstrates

Running the same deliberately ambiguous ticket through all three versions:

| Version | ~tokens | Format | Category | Urgency |
|---|---|---|---|---|
| Zero-shot | 73 | free prose, unparseable | Technical | Critical |
| One-shot | 83 | clean `Field: Value` | Billing | High |
| Few-shot | 251 | strict JSON | Billing | Critical |

One example fixes *format* (shape) but can't teach a *judgment* boundary
(Billing vs. Technical) from a single, unrelated demonstration. Three
contrasting examples can -- and the structurally-similar CSV-outage example
is what lets the model extend the urgency pattern one notch further
(High -> Critical) instead of just copying it.

## Tests

```bash
pytest -v
```

7/7 passing.

## License

MIT
