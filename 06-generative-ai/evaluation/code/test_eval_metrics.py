import pytest
from eval_metrics import (
    exact_match,
    token_f1,
    rouge_l,
    bleu_score,
    evaluate_hallucination,
    query_relevance,
    recall_at_k,
    mrr,
    ndcg_at_k,
    format_rubric_judge_prompt,
    format_pairwise_judge_prompt,
    parse_judge_score,
    parse_pairwise_winner,
    run_pairwise_swap_judge,
    cohens_kappa,
    bootstrap_ci
)


def test_exact_match_and_token_f1():
    assert exact_match("Paris, France!", "paris france") == 1.0
    assert exact_match("London", "Paris") == 0.0

    # Token F1
    pred = "the quick brown fox"
    target = "the fast brown fox"
    f1 = token_f1(pred, target)
    assert 0.70 < f1 < 0.80  # 3 shared tokens out of 4 (prec=0.75, rec=0.75, f1=0.75)


def test_rouge_l():
    pred = "deep learning models learn representations from data"
    target = "deep learning models extract high dimensional representations"
    res = rouge_l(pred, target)
    assert res["precision"] > 0.5
    assert res["recall"] > 0.5
    assert res["f1"] > 0.5


def test_bleu_score():
    pred = "the cat sat on the mat"
    target = "the cat sat on the mat"
    assert pytest.approx(bleu_score(pred, target), 1e-4) == 1.0

    pred2 = "a dog ran in the park"
    assert bleu_score(pred2, target) < 0.3


def test_groundedness_and_hallucination():
    context = "The Eiffel Tower was completed in 1889 in Paris, France for the World's Fair."
    grounded_resp = "The Eiffel Tower was built in Paris in 1889."
    hallucinated_resp = "The Eiffel Tower was erected in Tokyo Japan in 1958."

    g_eval = evaluate_hallucination(grounded_resp, context)
    assert g_eval["groundedness"] == 1.0
    assert g_eval["hallucination_rate"] == 0.0

    h_eval = evaluate_hallucination(hallucinated_resp, context)
    assert h_eval["groundedness"] < 0.5
    assert h_eval["hallucination_rate"] > 0.5


def test_query_relevance():
    query = "What is the capital of France?"
    good_resp = "Paris is the capital of France."
    bad_resp = "Photosynthesis is the process by which green plants make food."
    assert query_relevance(query, good_resp) > 0.6
    assert query_relevance(query, bad_resp) == 0.0


def test_retrieval_metrics():
    retrieved = ["doc_1", "doc_5", "doc_2", "doc_8", "doc_3"]
    ground_truth = {"doc_2", "doc_3"}

    assert recall_at_k(retrieved, ground_truth, k=2) == 0.0
    assert recall_at_k(retrieved, ground_truth, k=3) == 0.5
    assert recall_at_k(retrieved, ground_truth, k=5) == 1.0

    # First relevant doc is at rank 3 -> MRR = 1/3
    assert pytest.approx(mrr(retrieved, ground_truth), 1e-4) == 1.0 / 3.0

    # nDCG
    ndcg5 = ndcg_at_k(retrieved, ground_truth, k=5)
    assert 0.0 < ndcg5 <= 1.0


def test_judge_prompts_and_bias_mitigation():
    r_prompt = format_rubric_judge_prompt("Explain gravity", "Mass attracts mass", "5: accurate, 1: wrong")
    assert "[USER QUERY]" in r_prompt
    assert parse_judge_score("Reasoning: good. Score: 4") == 4

    p_prompt = format_pairwise_judge_prompt("Query", "A", "B", "helpful")
    assert "Winner: A" in p_prompt
    assert parse_pairwise_winner("After evaluation, Winner: A") == "A"

    # Test swap judge with consistent judge
    def consistent_judge(q, a, b):
        return "Winner: A" if "better" in a else "Winner: B"

    res = run_pairwise_swap_judge(consistent_judge, "Q", "better response", "inferior response")
    assert res["final_winner"] == "A"
    assert res["is_consistent"] is True

    # Test swap judge with position-biased judge (always picks first presented option A)
    def biased_judge(q, a, b):
        return "Winner: A"

    res_biased = run_pairwise_swap_judge(biased_judge, "Q", "response 1", "response 2")
    assert res_biased["final_winner"] == "Tie"  # neutralized
    assert res_biased["is_consistent"] is False


def test_human_eval_and_bootstrap():
    # Perfect agreement
    r1 = [1, 2, 3, 1, 2]
    r2 = [1, 2, 3, 1, 2]
    assert pytest.approx(cohens_kappa(r1, r2), 1e-4) == 1.0

    # Random / partial agreement
    r3 = [1, 2, 1, 2, 1]
    r4 = [2, 1, 2, 1, 2]
    assert cohens_kappa(r3, r4) < 0.0

    # Bootstrap CI
    scores = [0.8, 0.85, 0.78, 0.92, 0.88, 0.84, 0.90]
    low, high = bootstrap_ci(scores, n_boot=200, alpha=0.05)
    mean_val = sum(scores) / len(scores)
    assert low <= mean_val <= high
