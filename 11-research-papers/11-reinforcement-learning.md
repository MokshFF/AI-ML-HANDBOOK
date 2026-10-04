# Seminal Research Papers: Reinforcement Learning

Policy gradients, deep Q-networks, actor-critic architectures, and superhuman decision systems.

---

## 1. Proximal Policy Optimization Algorithms (PPO)
- **Title**: Proximal Policy Optimization Algorithms
- **Authors**: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov
- **Year**: 2017
- **Link**: https://arxiv.org/abs/1707.06347
- **Problem**: Standard policy gradient methods suffer from destructive policy updates and high sample complexity; Trust Region Policy Optimization (TRPO) is computationally heavy.
- **Main Idea**: Clip the probability ratio between the new and old policy within a small interval $[1-\epsilon, 1+\epsilon]$, preventing destructive policy collapse.
- **Key Contribution**: Formulated the clipped surrogate objective, providing TRPO-like monotonic improvement stability with first-order gradient simplicity.
- **Important Architecture/Math**:
  $$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \operatorname{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$
- **Why It Matters**: The default reinforcement learning algorithm across robotics, game AI (OpenAI Five), and RLHF for LLMs.
- **Prerequisites**: Policy gradient theorem, advantage function estimation (GAE), Markov decision processes.
- **Suggested Follow-up Papers**: *Trust Region Policy Optimization* (Schulman et al., 2015); *Direct Preference Optimization* (Rafailov et al., 2023).
