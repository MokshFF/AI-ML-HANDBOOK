"""
Reinforcement Learning for ML - Environments & Algorithms from Scratch
Implements GridWorld MDP, Dynamic Programming (Value Iteration),
Model-Free Temporal Difference Control (Q-Learning), and REINFORCE Policy Gradient.
"""

from __future__ import annotations
import numpy as np


class GridWorldMDP:
    """
    GridWorld Environment modeled as a Markov Decision Process (MDP).
    Grid size: height x width. Actions: 0=Up, 1=Down, 2=Left, 3=Right.
    """
    ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

    def __init__(self, height: int = 4, width: int = 4, goal: tuple[int, int] = (3, 3), traps: list[tuple[int, int]] = [(1, 1)]):
        self.height = height
        self.width = width
        self.goal = goal
        self.traps = traps
        self.n_states = height * width
        self.n_actions = 4
        self.state = (0, 0)

    def to_state_idx(self, coord: tuple[int, int]) -> int:
        return coord[0] * self.width + coord[1]

    def to_coord(self, state_idx: int) -> tuple[int, int]:
        return state_idx // self.width, state_idx % self.width

    def reset(self) -> int:
        self.state = (0, 0)
        return self.to_state_idx(self.state)

    def step(self, action: int) -> tuple[int, float, bool]:
        if self.state == self.goal or self.state in self.traps:
            return self.to_state_idx(self.state), 0.0, True

        delta = self.ACTIONS[action]
        next_r = min(max(self.state[0] + delta[0], 0), self.height - 1)
        next_c = min(max(self.state[1] + delta[1], 0), self.width - 1)
        next_coord = (next_r, next_c)

        self.state = next_coord
        state_idx = self.to_state_idx(next_coord)

        if next_coord == self.goal:
            return state_idx, 10.0, True
        elif next_coord in self.traps:
            return state_idx, -10.0, True
        else:
            return state_idx, -0.1, False  # Step cost to encourage shortest path


def value_iteration(env: GridWorldMDP, gamma: float = 0.9, theta: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """
    Solves Bellman Optimality Equation via Value Iteration:
    V_{k+1}(s) = max_a sum_{s'} P(s' | s, a) [R(s, a, s') + gamma * V_k(s')]
    """
    V = np.zeros(env.n_states, dtype=np.float64)

    while True:
        delta = 0.0
        for s in range(env.n_states):
            coord = env.to_coord(s)
            if coord == env.goal or coord in env.traps:
                continue

            v_old = V[s]
            q_values = []
            for a in range(env.n_actions):
                # Deterministic transitions in this environment
                dr, dc = env.ACTIONS[a]
                nr = min(max(coord[0] + dr, 0), env.height - 1)
                nc = min(max(coord[1] + dc, 0), env.width - 1)
                next_coord = (nr, nc)
                s_prime = env.to_state_idx(next_coord)

                if next_coord == env.goal:
                    reward = 10.0
                elif next_coord in env.traps:
                    reward = -10.0
                else:
                    reward = -0.1

                q_values.append(reward + gamma * V[s_prime])

            V[s] = max(q_values)
            delta = max(delta, abs(v_old - V[s]))

        if delta < theta:
            break

    # Extract optimal policy pi^*(s)
    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        coord = env.to_coord(s)
        q_vals = []
        for a in range(env.n_actions):
            dr, dc = env.ACTIONS[a]
            nr = min(max(coord[0] + dr, 0), env.height - 1)
            nc = min(max(coord[1] + dc, 0), env.width - 1)
            s_prime = env.to_state_idx((nr, nc))
            q_vals.append(V[s_prime])
        policy[s] = int(np.argmax(q_vals))

    return V, policy


class QLearningAgent:
    """
    Model-Free Temporal Difference (TD) Control via Q-Learning.
    Off-policy update: Q(s, a) <- Q(s, a) + alpha * [r + gamma * max_a' Q(s', a') - Q(s, a)]
    """
    def __init__(self, n_states: int, n_actions: int, lr: float = 0.1, gamma: float = 0.9, epsilon: float = 0.2):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = np.zeros((n_states, n_actions), dtype=np.float64)

    def select_action(self, state: int) -> int:
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        target = reward if done else reward + self.gamma * np.max(self.Q[next_state])
        td_error = target - self.Q[state, action]
        self.Q[state, action] += self.lr * td_error


class REINFORCEPolicyGradient:
    """
    Monte Carlo Policy Gradient (REINFORCE) with Softmax parameterized policy.
    pi_theta(a | s) = exp(theta_{s, a}) / sum_a' exp(theta_{s, a'})
    Update: theta <- theta + alpha * sum_t grad log pi(a_t | s_t) * G_t
    """
    def __init__(self, n_states: int, n_actions: int, lr: float = 0.05, gamma: float = 0.99):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = lr
        self.gamma = gamma
        self.theta = np.zeros((n_states, n_actions), dtype=np.float64)

    def get_action_probs(self, state: int) -> np.ndarray:
        logits = self.theta[state] - np.max(self.theta[state])
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    def select_action(self, state: int) -> int:
        probs = self.get_action_probs(state)
        return int(np.random.choice(self.n_actions, p=probs))

    def train_trajectory(self, trajectory: list[tuple[int, int, float]]) -> None:
        """trajectory: list of (state, action, reward) tuples."""
        T = len(trajectory)
        returns = np.zeros(T)
        G = 0.0
        for t in reversed(range(T)):
            G = trajectory[t][2] + self.gamma * G
            returns[t] = G

        # Update policy weights theta
        for t, (s, a, _) in enumerate(trajectory):
            probs = self.get_action_probs(s)
            grad_log_pi = -probs
            grad_log_pi[a] += 1.0  # d/d theta = 1 - p(a|s) for chosen action
            self.theta[s] += self.lr * grad_log_pi * returns[t]
