"""Dynamic AI Interview Trainer and Multi-Dimensional Rubric Evaluator."""
from typing import Dict, Any, List

class InterviewTrainer:
    def __init__(self, role: str = "Machine Learning Engineer"):
        self.role = role
        self.questions = [
            {"id": "q1", "text": "Explain the difference between L1 and L2 regularization.", "keywords": ["lasso", "ridge", "sparsity", "penalty", "zero"]},
            {"id": "q2", "text": "How do you detect and handle data leakage in ML pipelines?", "keywords": ["train", "test", "target", "leakage", "split", "fit_transform"]},
            {"id": "q3", "text": "Explain the vanishing gradient problem in deep networks and how to mitigate it.", "keywords": ["sigmoid", "relu", "residual", "batchnorm", "skip"]}
        ]
        self.history = []

    def evaluate_response(self, question_id: str, candidate_answer: str) -> Dict[str, Any]:
        q = next((item for item in self.questions if item["id"] == question_id), None)
        if not q:
            raise ValueError(f"Unknown question ID: {question_id}")
            
        words = set(candidate_answer.lower().split())
        matched = [k for k in q["keywords"] if any(k in w for w in words)]
        match_ratio = len(matched) / len(q["keywords"])
        
        # Rubric scoring 1-5
        if match_ratio >= 0.7:
            score = 5
            feedback = "Outstanding depth and precise technical vocabulary."
        elif match_ratio >= 0.4:
            score = 4
            feedback = "Good conceptual grasp; consider elaborating on edge cases."
        elif match_ratio >= 0.2:
            score = 3
            feedback = "Acceptable high-level overview, but missed key technical mechanisms."
        else:
            score = 2
            feedback = "Incomplete response. Recommend reviewing foundational definitions."
            
        result = {
            "question_id": question_id,
            "score": score,
            "matched_keywords": matched,
            "feedback": feedback
        }
        self.history.append(result)
        return result

    def get_summary_report(self) -> Dict[str, Any]:
        if not self.history:
            return {"total_questions": 0, "average_score": 0.0}
        avg_score = sum(h["score"] for h in self.history) / len(self.history)
        return {
            "total_questions": len(self.history),
            "average_score": round(avg_score, 2),
            "readiness": "HIRE" if avg_score >= 4.0 else "NEEDS_PRACTICE"
        }

if __name__ == "__main__":
    trainer = InterviewTrainer()
    ans = "L1 produces sparsity by driving weights to zero (Lasso), while L2 uses squared penalties (Ridge)."
    res = trainer.evaluate_response("q1", ans)
    print("Evaluation Result:", res)
    print("Summary:", trainer.get_summary_report())
