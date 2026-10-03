import pytest
import numpy as np
from rl_engine import GridWorldMDP, value_iteration, QLearningAgent, REINFORCEPolicyGradient


def test_gridworld_mdp():
    env = GridWorldMDP(height=3, width=3, goal=(2, 2), traps=[(1, 1)])
    s0 = env.reset()
    assert s0 == 0
    assert env.to_coord(s0) == (0, 0)
    assert env.to_state_idx((1, 2)) == 5

    # Move right: (0, 0) -> (0, 1)
    s_next, r, done = env.step(3)
    assert s_next == 1
    assert r == -0.1
    assert not done

    # Move down to trap: (0, 1) -> (1, 1)
    s_next, r, done = env.step(1)
    assert s_next == 4
    assert r == -10.0
    assert done


def test_value_iteration():
    env = GridWorldMDP(height=3, width=3, goal=(2, 2), traps=[(1, 1)])
    V, policy = value_iteration(env, gamma=0.9, theta=1e-4)

    assert len(V) == 9
    assert len(policy) == 9
    # The state next to the goal should have higher value than initial state (0,0)
    s_goal_neighbor = env.to_state_idx((2, 1))
    s_start = env.to_state_idx((0, 0))
    assert V[s_goal_neighbor] > V[s_start]
    # Optimal policy should be valid actions
    assert np.all((policy >= 0) & (policy < 4))


def test_q_learning_agent():
    np.random.seed(42)
    env = GridWorldMDP(height=3, width=3, goal=(2, 2), traps=[(1, 1)])
    agent = QLearningAgent(n_states=env.n_states, n_actions=env.n_actions, lr=0.1, gamma=0.9, epsilon=0.2)

    # Initial Q-values are zero
    assert np.all(agent.Q == 0.0)

    # Test greedy selection
    agent.epsilon = 0.0
    agent.Q[0, 3] = 5.0
    assert agent.select_action(0) == 3

    # Test TD update
    agent.update(state=0, action=3, reward=1.0, next_state=1, done=False)
    # Target = 1.0 + 0.9 * max(Q[1]) = 1.0; td_error = 1.0 - 5.0 = -4.0
    # Q[0, 3] = 5.0 + 0.1 * (-4.0) = 4.6
    assert np.isclose(agent.Q[0, 3], 4.6)

    # Mini training loop
    for _ in range(200):
        s = env.reset()
        done = False
        steps = 0
        while not done and steps < 20:
            a = agent.select_action(s)
            next_s, r, done = env.step(a)
            agent.update(s, a, r, next_s, done)
            s = next_s
            steps += 1

    assert np.any(agent.Q != 0.0)


def test_reinforce_policy_gradient():
    np.random.seed(42)
    agent = REINFORCEPolicyGradient(n_states=4, n_actions=2, lr=0.1, gamma=0.99)
    probs = agent.get_action_probs(0)
    assert np.isclose(np.sum(probs), 1.0)
    assert np.allclose(probs, [0.5, 0.5])

    # Provide a positive trajectory for action 1 in state 0
    trajectory = [(0, 1, 10.0)]
    agent.train_trajectory(trajectory)

    new_probs = agent.get_action_probs(0)
    # Action 1 should have higher probability after positive reward
    assert new_probs[1] > new_probs[0]
