from .core import (
    TICKET, classify_zero_shot, classify_one_shot, classify_few_shot,
    approx_token_count, ZERO_SHOT_PROMPT, ONE_SHOT_PROMPT, FEW_SHOT_PROMPT,
)


def main():
    print(f"Ticket:\n  {TICKET}\n")
    z = classify_zero_shot()
    print(f"Zero-shot (~{approx_token_count(ZERO_SHOT_PROMPT)} tokens):\n  {z.raw_text}\n  parseable={z.parseable}\n")
    o = classify_one_shot()
    print(f"One-shot (~{approx_token_count(ONE_SHOT_PROMPT)} tokens):\n  category={o.category} urgency={o.urgency}\n")
    f = classify_few_shot()
    print(f"Few-shot (~{approx_token_count(FEW_SHOT_PROMPT)} tokens):\n"
          f"  category={f.category} urgency={f.urgency} escalate_to_oncall={f.escalate_to_oncall}\n")


if __name__ == "__main__":
    main()
