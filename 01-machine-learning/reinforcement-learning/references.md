# Reinforcement Learning - References & Curated Bibliography

A curated collection of foundational textbooks, landmark research papers, production toolkits, and benchmark suites for Reinforcement Learning.

---

## 1. Authoritative Textbooks & Monographs

- **Reinforcement Learning: An Introduction (2nd Edition)**
  - *Authors*: Richard S. Sutton and Andrew G. Barto (MIT Press, 2018).
  - *Significance*: The definitive canonical text covering Markov Decision Processes, Dynamic Programming, Monte Carlo methods, Temporal Difference learning, eligibility traces, and policy gradient theorems.
  - *URL*: [Complete Online Draft](http://incompleteideas.net/book/the-book-2nd.html)

- **Algorithms for Reinforcement Learning**
  - *Author*: Csaba Szepesvári (Morgan & Claypool Publishers, 2010).
  - *Significance*: Formal mathematical analysis of value functions, sample complexity, and contraction mappings.

- **Dynamic Programming and Optimal Control (Vols. I & II)**
  - *Author*: Dimitri P. Bertsekas (Athena Scientific).
  - *Significance*: Deep mathematical foundations of Hamilton-Jacobi-Bellman equations and stochastic dynamic programming.

---

## 2. Landmark Research Papers

### 2.1 Value-Based & Q-Learning Milestones
- **Learning from Delayed Rewards**
  - *Authors*: Christopher J. C. H. Watkins (Ph.D. Thesis, Cambridge University, 1989).
  - *Significance*: First formal introduction and convergence proof of Q-learning.
- **Human-level Control Through Deep Reinforcement Learning (DQN)**
  - *Authors*: Volodymyr Mnih, Koray Kavukcuoglu, David Silver, et al. (Nature, 2015).
  - *Significance*: Combined Q-learning with deep convolutional neural networks, experience replay, and target networks to solve Atari 2600 games.
- **Deep Reinforcement Learning with Double Q-learning**
  - *Authors*: Hado van Hasselt, Arthur Guez, David Silver (AAAI, 2016).
  - *Significance*: Addressed overestimation bias in deep Q-networks using decoupled network evaluations.

### 2.2 Policy Gradient & Actor-Critic Milestones
- **Policy Gradient Methods for Reinforcement Learning with Function Approximation**
  - *Authors*: Richard S. Sutton, David McAllester, Satinder Singh, Yishay Mansour (NeurIPS, 1999).
  - *Significance*: Established the analytical foundation for the Policy Gradient Theorem with arbitrary function approximation.
- **High-Dimensional Continuous Control Using Generalized Advantage Estimation (GAE)**
  - *Authors*: John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, Pieter Abbeel (ICLR, 2016).
  - *Significance*: Introduced GAE parameter $\lambda$ balancing bias and variance in advantage estimation.
- **Proximal Policy Optimization Algorithms (PPO)**
  - *Authors*: John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov (arXiv, 2017).
  - *Significance*: Standardized clipped surrogate objectives for stable, sample-efficient policy optimization.
- **Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor**
  - *Authors*: Tuomas Haarnoja, Aurick Zhou, Pieter Abbeel, Sergey Levine (ICML, 2018).
  - *Significance*: Unified off-policy actor-critic learning with maximum entropy exploration.

### 2.3 RL for Language Models & Alignment
- **Training Language Models to Follow Instructions with Human Feedback (InstructGPT)**
  - *Authors*: Long Ouyang, Jeff Wu, Xu Jiang, et al. (NeurIPS, 2022).
  - *Significance*: Established modern 3-stage alignment (SFT + Reward Modeling + PPO) for LLMs.
- **Direct Preference Optimization: Your Language Model is Secretly a Reward Model (DPO)**
  - *Authors*: Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D. Manning, Chelsea Finn (NeurIPS, 2023).
  - *Significance*: Proved exact analytical mapping between reward models and optimal policies, removing the need for explicit RL sampling.

---

## 3. Libraries, Toolkits & Production Frameworks

- **Gymnasium (Farama Foundation)**: Standard API for single-agent RL environments (successor to OpenAI Gym). [gymnasium.farama.org](https://gymnasium.farama.org/)
- **CleanRL**: High-quality single-file implementations of deep RL algorithms with clear benchmark logging. [docs.cleanrl.dev](https://docs.cleanrl.dev/)
- **Ray RLlib**: Production-grade distributed reinforcement learning supporting large-scale multi-GPU training. [docs.ray.io/en/latest/rllib/](https://docs.ray.io/en/latest/rllib/)
- **TRL (Transformer Reinforcement Learning - Hugging Face)**: Specialized library for training foundation models with PPO, DPO, and GRPO. [huggingface.co/docs/trl](https://huggingface.co/docs/trl)
