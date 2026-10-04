"""
Generative AI Evaluation Suite: lexical metrics (Exact Match, Token F1, ROUGE-L, BLEU),
RAG groundedness & hallucination metrics, retrieval metrics (Recall@k, MRR, nDCG@k),
LLM-as-a-Judge prompt formatting and position-bias mitigation, and human evaluation
inter-annotator agreement (Cohen's Kappa & bootstrap confidence intervals).
"""

import math
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple


# --------------------------------------------------------------------------- #
# Tokenization & Text Normalization
# --------------------------------------------------------------------------- #
def normalize_text(text: str) -> str:
    """Lowercases, strips punctuation, and standardizes whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return " ".join(text.split())


def get_tokens(text: str) -> List[str]:
    return normalize_text(text).split()


# --------------------------------------------------------------------------- #
# Lexical & Overlap Metrics
# --------------------------------------------------------------------------- #
def exact_match(pred: str, target: str) -> float:
    """Exact string match after standard normalization."""
    return 1.0 if normalize_text(pred) == normalize_text(target) else 0.0


def token_f1(pred: str, target: str) -> float:
    """Token-level precision, recall, and harmonic F1 score."""
    p_tokens = get_tokens(pred)
    t_tokens = get_tokens(target)
    if not p_tokens or not t_tokens:
        return 1.0 if p_tokens == t_tokens else 0.0
    common: Dict[str, int] = {}
    p_counts: Dict[str, int] = {}
    for tok in p_tokens:
        p_counts[tok] = p_counts.get(tok, 0) + 1
    t_counts: Dict[str, int] = {}
    for tok in t_tokens:
        t_counts[tok] = t_counts.get(tok, 0) + 1
    shared = 0
    for tok, cnt in p_counts.items():
        if tok in t_counts:
            shared += min(cnt, t_counts[tok])
    if shared == 0:
        return 0.0
    prec = shared / len(p_tokens)
    rec = shared / len(t_tokens)
    return 2.0 * prec * rec / (prec + rec)


def lcs_length(x: Sequence[str], y: Sequence[str]) -> int:
    """Longest common subsequence length via dynamic programming."""
    m, n = len(x), len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def rouge_l(pred: str, target: str) -> Dict[str, float]:
    """ROUGE-L based on Longest Common Subsequence (precision, recall, f1)."""
    p_tokens = get_tokens(pred)
    t_tokens = get_tokens(target)
    if not p_tokens or not t_tokens:
        val = 1.0 if p_tokens == t_tokens else 0.0
        return {"precision": val, "recall": val, "f1": val}
    lcs = lcs_length(p_tokens, t_tokens)
    prec = lcs / len(p_tokens)
    rec = lcs / len(t_tokens)
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}


def bleu_score(pred: str, target: str, max_n: int = 4) -> float:
    """Sentence BLEU with modified n-gram precision and brevity penalty."""
    p_tokens = get_tokens(pred)
    t_tokens = get_tokens(target)
    p_len = len(p_tokens)
    t_len = len(t_tokens)
    if p_len == 0 or t_len == 0:
        return 1.0 if p_len == t_len else 0.0

    # Brevity penalty
    if p_len > t_len:
        bp = 1.0
    else:
        bp = math.exp(1.0 - t_len / p_len)

    log_sum = 0.0
    valid_n = 0
    for n in range(1, max_n + 1):
        if p_len < n:
            break
        # Build n-grams
        p_ngrams: Dict[Tuple[str, ...], int] = {}
        for i in range(p_len - n + 1):
            ng = tuple(p_tokens[i:i + n])
            p_ngrams[ng] = p_ngrams.get(ng, 0) + 1
        t_ngrams: Dict[Tuple[str, ...], int] = {}
        for i in range(t_len - n + 1):
            ng = tuple(t_tokens[i:i + n])
            t_ngrams[ng] = t_ngrams.get(ng, 0) + 1

        clipped = 0
        total_p = 0
        for ng, cnt in p_ngrams.items():
            total_p += cnt
            clipped += min(cnt, t_ngrams.get(ng, 0))

        if total_p == 0 or clipped == 0:
            p_n = 1e-9  # smoothing
        else:
            p_n = clipped / total_p
        log_sum += math.log(p_n)
        valid_n += 1

    if valid_n == 0:
        return 0.0
    geo_mean = math.exp(log_sum / valid_n)
    return bp * geo_mean


# --------------------------------------------------------------------------- #
# Groundedness, Hallucination & Relevance
# --------------------------------------------------------------------------- #
def extract_claims(text: str) -> List[str]:
    """Splits text into atomic claim-like sentences."""
    sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in sentences if len(s.strip().split()) >= 3]


def claim_groundedness(claim: str, context: str, threshold: float = 0.6) -> bool:
    """Checks if key non-stopword tokens in a claim are grounded in the source context."""
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "and", "or", "in", "on", "of", "to", "for", "with", "it", "this", "that", "from", "by", "at", "as"}
    claim_words = [w for w in get_tokens(claim) if w not in stop_words]
    if not claim_words:
        return True
    ctx_words = set(get_tokens(context))
    overlap = sum(1 for w in claim_words if w in ctx_words)
    return (overlap / len(claim_words)) >= threshold


def evaluate_hallucination(response: str, context: str) -> Dict[str, float]:
    """Computes groundedness ratio and hallucination rate across claims in response."""
    claims = extract_claims(response)
    if not claims:
        return {"groundedness": 1.0, "hallucination_rate": 0.0, "claim_count": 0.0}
    supported = sum(1 for c in claims if claim_groundedness(c, context))
    groundedness = supported / len(claims)
    return {
        "groundedness": groundedness,
        "hallucination_rate": 1.0 - groundedness,
        "claim_count": float(len(claims))
    }


def query_relevance(query: str, response: str) -> float:
    """Computes question-answering relevance based on query term coverage in response."""
    stop_words = {"what", "who", "where", "when", "why", "how", "is", "are", "the", "a", "an", "in", "of", "to"}
    q_words = [w for w in get_tokens(query) if w not in stop_words]
    if not q_words:
        return 1.0
    r_words = set(get_tokens(response))
    hit = sum(1 for w in q_words if w in r_words)
    return hit / len(q_words)


# --------------------------------------------------------------------------- #
# Retrieval Metrics
# --------------------------------------------------------------------------- #
def recall_at_k(retrieved_ids: Sequence[Any], ground_truth_ids: Set[Any], k: int) -> float:
    """Fraction of relevant documents retrieved within the top k positions."""
    if not ground_truth_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc in top_k if doc in ground_truth_ids)
    return hits / len(ground_truth_ids)


def mrr(retrieved_ids: Sequence[Any], ground_truth_ids: Set[Any]) -> float:
    """Mean Reciprocal Rank: reciprocal of the rank of the first relevant document."""
    for rank, doc in enumerate(retrieved_ids, start=1):
        if doc in ground_truth_ids:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved_ids: Sequence[Any], ground_truth_ids: Set[Any], k: int) -> float:
    """Normalized Discounted Cumulative Gain at position k for binary relevance."""
    if not ground_truth_ids or k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    dcg = 0.0
    for i, doc in enumerate(top_k):
        rel = 1.0 if doc in ground_truth_ids else 0.0
        dcg += rel / math.log2(i + 2)  # i+2 because rank=1 -> log2(2)=1
    # Ideal DCG
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(ground_truth_ids))))
    return dcg / idcg if idcg > 0 else 0.0


# --------------------------------------------------------------------------- #
# LLM-as-a-Judge Prompts & Position Bias Mitigation
# --------------------------------------------------------------------------- #
def format_rubric_judge_prompt(query: str, response: str, rubric: str) -> str:
    """Constructs a structured evaluation prompt for single-response score judging."""
    return f"""You are an impartial expert evaluator.
Evaluate the model response below according to the provided rubric.

[USER QUERY]
{query}

[MODEL RESPONSE]
{response}

[EVALUATION RUBRIC]
{rubric}

Provide your feedback in the following format:
Reasoning: <concise step-by-step assessment>
Score: <integer score between 1 and 5>"""


def format_pairwise_judge_prompt(query: str, resp_a: str, resp_b: str, criteria: str) -> str:
    """Constructs a pairwise comparison prompt between Response A and Response B."""
    return f"""You are an expert evaluator comparing two AI responses.
Criteria: {criteria}

[USER QUERY]
{query}

[RESPONSE A]
{resp_a}

[RESPONSE B]
{resp_b}

Which response is better? Output one of:
Winner: A
Winner: B
Winner: Tie"""


def parse_judge_score(judge_output: str) -> Optional[int]:
    """Extracts integer score (1-5) from judge response text."""
    match = re.search(r"score:\s*([1-5])", judge_output, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def parse_pairwise_winner(judge_output: str) -> str:
    """Parses winner label: 'A', 'B', or 'Tie'."""
    match = re.search(r"winner:\s*([ab]|tie)", judge_output, re.IGNORECASE)
    if match:
        w = match.group(1).upper()
        return "Tie" if w == "TIE" else w
    return "Tie"


def run_pairwise_swap_judge(
    judge_fn: Callable[[str, str, str], str],
    query: str,
    resp_a: str,
    resp_b: str,
    criteria: str = "Correctness and helpfulness"
) -> Dict[str, Any]:
    """
    Mitigates position bias by running two trials with swapped presentation order:
    Trial 1: Order (A, B)
    Trial 2: Order (B, A)
    If outcomes conflict due to order preference, returns Tie.
    """
    out1 = judge_fn(query, resp_a, resp_b)
    win1 = parse_pairwise_winner(out1)

    out2 = judge_fn(query, resp_b, resp_a)
    win2 = parse_pairwise_winner(out2)  # here 'A' corresponds to resp_b, 'B' to resp_a

    # Resolve win2 back to original labels
    if win2 == "A":
        resolved_win2 = "B"
    elif win2 == "B":
        resolved_win2 = "A"
    else:
        resolved_win2 = "Tie"

    if win1 == resolved_win2:
        final_winner = win1
        consistent = True
    else:
        final_winner = "Tie"  # Order effect / position bias neutralized
        consistent = False

    return {
        "final_winner": final_winner,
        "trial_1_winner": win1,
        "trial_2_winner": resolved_win2,
        "is_consistent": consistent
    }


# --------------------------------------------------------------------------- #
# Human Evaluation & Statistics
# --------------------------------------------------------------------------- #
def cohens_kappa(rater1: Sequence[int], rater2: Sequence[int]) -> float:
    """
    Cohen's Kappa for inter-annotator agreement between two categorical raters.
    kappa = (P_observed - P_expected) / (1 - P_expected)
    """
    if len(rater1) != len(rater2) or len(rater1) == 0:
        raise ValueError("Rater sequences must be non-empty and of equal length.")
    n = len(rater1)
    categories = sorted(set(rater1) | set(rater2))
    cat_to_idx = {c: i for i, c in enumerate(categories)}
    num_cats = len(categories)

    # Confusion matrix
    cm = [[0] * num_cats for _ in range(num_cats)]
    for r1, r2 in zip(rater1, rater2):
        cm[cat_to_idx[r1]][cat_to_idx[r2]] += 1

    # Observed agreement
    p_o = sum(cm[i][i] for i in range(num_cats)) / n

    # Expected chance agreement
    row_sums = [sum(cm[i][j] for j in range(num_cats)) for i in range(num_cats)]
    col_sums = [sum(cm[i][j] for i in range(num_cats)) for j in range(num_cats)]
    p_e = sum((row_sums[i] * col_sums[i]) for i in range(num_cats)) / (n * n)

    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def bootstrap_ci(scores: Sequence[float], n_boot: int = 1000, alpha: float = 0.05, seed: int = 42) -> Tuple[float, float]:
    """
    Non-parametric bootstrap confidence interval for mean benchmark performance.
    Returns (lower_bound, upper_bound) at (1 - alpha) confidence.
    """
    if not scores:
        return (0.0, 0.0)
    import random
    rng = random.Random(seed)
    n = len(scores)
    means = []
    for _ in range(n_boot):
        sample = [scores[rng.randint(0, n - 1)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    low_idx = int(math.floor((alpha / 2.0) * n_boot))
    high_idx = int(math.ceil((1.0 - alpha / 2.0) * n_boot)) - 1
    return (means[low_idx], means[high_idx])
