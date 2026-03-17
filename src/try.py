import numpy as np

from homework2 import Hw2Env

N_ACTIONS = 8
env = Hw2Env(n_actions=N_ACTIONS, render_mode="gui")
for episode in range(10):
    env.reset()
    done = False
    cumulative_reward = 0.0
    episode_steps = 0
    while not done:
        action = np.random.randint(N_ACTIONS)
        state, reward, is_terminal, is_truncated = env.step(action)
        done = is_terminal or is_truncated
        cumulative_reward += reward
        episode_steps += 1
    print(f"Episode={episode}, reward={cumulative_reward}, RPS={cumulative_reward/episode_steps}")