"""
RouteSage: a zero-shot -> one-shot -> few-shot escalation demo for support
ticket triage (category + urgency + an escalate_to_oncall routing flag).

DISCLOSURE: there is no live LLM call anywhere in this package. Each
classify_* function is a deterministic stand-in chosen to be representative
of what an instruction-tuned model plausibly produces for that exact prompt
shape -- not a captured API response. The approximate token counter is a
regex word/punctuation-level count, not a real BPE tokenizer, used only to
compare *relative* growth across the three prompt versions.
"""
import re
from dataclasses import dataclass

TICKET = (
    "We've been unable to process any payments through the API since this "
    "morning -- our checkout flow for all customers is completely down and "
    "we're losing revenue by the minute."
)

_TOKEN_RE = re.compile(r"\w+|[^\w\s]")


def approx_token_count(text: str) -> int:
    """Regex word/punctuation-level approximation of token count (not a real BPE tokenizer)."""
    return len(_TOKEN_RE.findall(text))


@dataclass(frozen=True)
class ZeroShotResult:
    raw_text: str
    parseable: bool


def classify_zero_shot(ticket: str = TICKET) -> ZeroShotResult:
    """Deterministic stand-in: a plain classification question with no output
    indicator, no format instruction. Produces free-form prose -- correct in
    substance, unparseable in shape (no output indicator was given)."""
    text = (
        "This ticket appears to be a Technical issue, since the payment API is "
        "failing to process transactions correctly. Given that it's affecting "
        "checkout for all customers and causing ongoing revenue loss, I'd "
        "classify the urgency as Critical and recommend escalating to the "
        "on-call engineering team immediately."
    )
    return ZeroShotResult(text, parseable=False)


@dataclass(frozen=True)
class OneShotResult:
    category: str
    urgency: str


def classify_one_shot(ticket: str = TICKET) -> OneShotResult:
    """Deterministic stand-in: one worked example fixes the output SHAPE
    (clean Field: Value lines) but not the category/urgency JUDGMENT -- the
    single example (an Access ticket) gives no signal on the Billing/Technical
    boundary, so both labels drift from the zero-shot version."""
    return OneShotResult(category="Billing", urgency="High")


@dataclass(frozen=True)
class FewShotResult:
    category: str
    urgency: str
    escalate_to_oncall: bool


def classify_few_shot(ticket: str = TICKET) -> FewShotResult:
    """Deterministic stand-in: three contrasting worked examples (including one
    structurally similar to the target ticket) teach both the output shape
    (strict JSON) AND the category/urgency/escalate judgment boundary that a
    single example cannot."""
    return FewShotResult(category="Billing", urgency="Critical", escalate_to_oncall=True)


ZERO_SHOT_PROMPT = (
    "Classify the following customer support ticket by category (Billing, Technical, "
    "Access, Feature Request, or Other) and urgency (Low, Medium, High, Critical).\n\n"
    f'Ticket: "{TICKET}"'
)

ONE_SHOT_PROMPT = (
    "Classify the ticket by category and urgency. Respond in the exact format shown.\n\n"
    'Ticket: "I forgot my password and the reset link isn\'t arriving in my email."\n'
    "Category: Access\nUrgency: Medium\n\n"
    f'Ticket: "{TICKET}"\nCategory:'
)

FEW_SHOT_PROMPT = (
    "Classify the ticket. Respond with ONLY a JSON object matching this schema, no other text:\n"
    '{"category": "Billing" | "Technical" | "Access" | "Feature Request" | "Other", '
    '"urgency": "Low" | "Medium" | "High" | "Critical", "escalate_to_oncall": true | false}\n\n'
    'Ticket: "I forgot my password and the reset link isn\'t arriving in my email."\n'
    '{"category": "Access", "urgency": "Medium", "escalate_to_oncall": false}\n\n'
    'Ticket: "The CSV export button on the Enterprise dashboard has been greyed out '
    'for every account since last night\'s deploy."\n'
    '{"category": "Technical", "urgency": "High", "escalate_to_oncall": true}\n\n'
    'Ticket: "Could you add a dark mode toggle to the settings page? Would love this '
    'for working late."\n'
    '{"category": "Feature Request", "urgency": "Low", "escalate_to_oncall": false}\n\n'
    f'Ticket: "{TICKET}"'
)
