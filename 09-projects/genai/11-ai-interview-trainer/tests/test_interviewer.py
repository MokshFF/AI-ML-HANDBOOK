"""Tests for AI Interview Trainer."""
from interviewer import InterviewTrainer

def test_interview_eval():
    trainer = InterviewTrainer()
    res = trainer.evaluate_response("q1", "L1 regularization causes sparsity driving weights to zero like Lasso.")
    assert res["score"] >= 4
    assert "sparsity" in res["matched_keywords"]
    
    summary = trainer.get_summary_report()
    assert summary["total_questions"] == 1
    assert summary["average_score"] >= 4.0
