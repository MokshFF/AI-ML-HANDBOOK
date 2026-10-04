"""
Safety, Guardrails & Alignment Core:
1. Prompt injection detection (direct & indirect with canary tokens).
2. PII detection and regex-based redaction engine.
3. Input/Output Guardrail pipeline with topic/toxicity filtering.
4. Statistical watermarking (green-list partitioning and z-score verification).
5. Constitutional AI critique-revision loop.
"""

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple


# --------------------------------------------------------------------------- #
# 1. Prompt Injection & Jailbreak Detection
# --------------------------------------------------------------------------- #
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions?",
    r"disregard\s+(all\s+)?(rules|guidelines|system\s+prompts?)",
    r"you\s+are\s+now\s+(in\s+)?(dan\s+mode|unrestricted|god\s+mode|jailbroken)",
    r"bypass\s+(safety|content)\s+filters?",
    r"reveal\s+(your\s+)?(system\s+prompt|initial\s+instructions)",
    r"new\s+system\s+prompt:",
    r"print\s+everything\s+above"
]


def detect_prompt_injection(text: str) -> Tuple[bool, List[str]]:
    """Heuristic detector scanning for known prompt override and jailbreak signatures."""
    lowered = text.lower()
    matches = []
    for pat in INJECTION_PATTERNS:
        if re.search(pat, lowered):
            matches.append(pat)
    return (len(matches) > 0, matches)


def verify_canary(text: str, canary_token: str) -> bool:
    """
    Indirect prompt injection defense:
    Verifies that an untrusted document did not cause the model to output or leak the secret canary.
    """
    return canary_token in text


def encapsulate_untrusted_input(untrusted_text: str, delimiter_tag: str = "user_data") -> str:
    """
    Defensive prompt engineering:
    Encapsulates untrusted input in explicit XML/markdown boundaries to mitigate instruction escape.
    """
    cleaned = untrusted_text.replace(f"<{delimiter_tag}>", "").replace(f"</{delimiter_tag}>", "")
    return f"<{delimiter_tag}>\n{cleaned}\n</{delimiter_tag}>"


# --------------------------------------------------------------------------- #
# 2. PII Detection & Data Leakage Scrubber
# --------------------------------------------------------------------------- #
PII_PATTERNS = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b",
    "PHONE": r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "IPV4": r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b",
    "API_KEY": r"\b(?:sk-[A-Za-z0-9]{20,}|ghp_[A-Za-z0-9]{30,}|Bearer\s+[A-Za-z0-9\-_]{20,})\b",
    "CREDIT_CARD": r"\b(?:\d{4}[-\s]?){3}\d{4}\b"
}


def redact_pii(text: str) -> Tuple[str, Dict[str, int]]:
    """Redacts sensitive PII entities (email, phone, ip, api keys, credit cards) from text."""
    redacted = text
    counts = {}
    for pii_type, pattern in PII_PATTERNS.items():
        found = re.findall(pattern, redacted, re.IGNORECASE)
        counts[pii_type] = len(found)
        if found:
            redacted = re.sub(pattern, f"[REDACTED_{pii_type}]", redacted, flags=re.IGNORECASE)
    return redacted, counts


# --------------------------------------------------------------------------- #
# 3. Input / Output Guardrail Pipeline
# --------------------------------------------------------------------------- #
BANNED_TOPICS = {
    "malware": ["keylogger", "write ransomware", "ransomware", "exploit buffer overflow", "ddos script", "malware"],
    "weapons": ["build a bomb", "synthesize nerve agent", "make explosives"],
    "hate": ["genocide was good", "superior race doctrine"]
}


@dataclass
class GuardrailResult:
    passed: bool
    filtered_text: str
    reasons: List[str] = field(default_factory=list)


class SafetyGuardrail:
    """Composite guardrail inspecting prompts and completions for safety compliance."""
    def __init__(self, sanitize_pii: bool = True, block_injections: bool = True):
        self.sanitize_pii = sanitize_pii
        self.block_injections = block_injections

    def check_input(self, user_prompt: str) -> GuardrailResult:
        reasons = []
        lowered = user_prompt.lower()

        # 1. Prompt Injection
        if self.block_injections:
            is_inj, matches = detect_prompt_injection(user_prompt)
            if is_inj:
                reasons.append("Prompt injection / jailbreak pattern detected")

        # 2. Banned high-risk topics
        for cat, phrases in BANNED_TOPICS.items():
            for p in phrases:
                if p in lowered:
                    reasons.append(f"Prohibited harm category: {cat}")
                    break

        if reasons:
            return GuardrailResult(passed=False, filtered_text="I cannot fulfill this request due to safety policies.", reasons=reasons)

        processed = user_prompt
        if self.sanitize_pii:
            processed, _ = redact_pii(processed)

        return GuardrailResult(passed=True, filtered_text=processed, reasons=[])

    def check_output(self, model_completion: str) -> GuardrailResult:
        reasons = []
        processed = model_completion

        # PII scrub
        if self.sanitize_pii:
            processed, counts = redact_pii(processed)
            if any(c > 0 for c in counts.values()):
                reasons.append("Output contained sensitive PII that was redacted")

        return GuardrailResult(passed=True, filtered_text=processed, reasons=reasons)


# --------------------------------------------------------------------------- #
# 4. Statistical Text Watermarking (Kirchenbauer et al.)
# --------------------------------------------------------------------------- #
class TextWatermarker:
    """
    Green-list partitioning watermarking:
    Pseudo-randomly partitions vocabulary into green-list and red-list based on previous token hash.
    During generation, logits of green tokens receive bonus delta.
    During detection, z-score of green tokens tests presence of watermark.
    """
    def __init__(self, vocab_size: int = 1000, gamma: float = 0.5, delta: float = 2.0, secret_key: str = "watermark_seed"):
        self.vocab_size = vocab_size
        self.gamma = gamma  # proportion of green tokens
        self.delta = delta  # logit bias
        self.secret_key = secret_key

    def _get_green_list(self, prev_token: int) -> Set[int]:
        # Hash previous token with secret key
        h = hashlib.sha256(f"{self.secret_key}:{prev_token}".encode("utf-8")).hexdigest()
        seed = int(h[:8], 16)
        import random
        rng = random.Random(seed)
        shuffled = list(range(self.vocab_size))
        rng.shuffle(shuffled)
        cutoff = int(self.gamma * self.vocab_size)
        return set(shuffled[:cutoff])

    def bias_logits(self, logits: List[float], prev_token: int) -> List[float]:
        """Adds delta bonus to green-list tokens."""
        green_set = self._get_green_list(prev_token)
        biased = list(logits)
        for token_id in green_set:
            if token_id < len(biased):
                biased[token_id] += self.delta
        return biased

    def detect_watermark(self, token_sequence: Sequence[int]) -> Dict[str, float]:
        """
        Calculates z-score:
        z = (|Green| - gamma * T) / sqrt(T * gamma * (1 - gamma))
        """
        if len(token_sequence) <= 1:
            return {"z_score": 0.0, "p_value": 1.0, "green_ratio": 0.0, "is_watermarked": False}
        t_total = len(token_sequence) - 1
        green_hits = 0
        for i in range(1, len(token_sequence)):
            prev = token_sequence[i - 1]
            curr = token_sequence[i]
            if curr in self._get_green_list(prev):
                green_hits += 1

        exp = self.gamma * t_total
        std = math.sqrt(t_total * self.gamma * (1.0 - self.gamma))
        z = (green_hits - exp) / std if std > 0 else 0.0
        # Standard normal approx p-value (one-tailed)
        p_val = 0.5 * math.erfc(z / math.sqrt(2))
        return {
            "z_score": z,
            "p_value": p_val,
            "green_ratio": green_hits / t_total,
            "is_watermarked": z > 4.0  # standard threshold in literature
        }


# --------------------------------------------------------------------------- #
# 5. Constitutional AI Critique & Revision Loop
# --------------------------------------------------------------------------- #
def constitutional_critique_and_revise(
    draft_response: str,
    principle: str,
    critique_fn: Callable[[str, str], str],
    revision_fn: Callable[[str, str, str], str]
) -> Dict[str, str]:
    """
    Simulates Anthropic's Constitutional AI step:
    1. Critique draft response against a constitutional principle.
    2. Revise response to fix critiques while maintaining helpfulness.
    """
    critique = critique_fn(draft_response, principle)
    revised = revision_fn(draft_response, principle, critique)
    return {
        "original_draft": draft_response,
        "principle": principle,
        "critique": critique,
        "revised_response": revised
    }
