import os
import random
from collections import deque

# Initialize CUDA before MuJoCo/EGL claims the GPU
import torch
if torch.cuda.is_available():
    torch.zeros(1).cuda()

import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from homework2 import Hw2Env

# ─── Hyperparameters ───
N_ACTIONS = 8
GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 0.999
EPSILON_DECAY_ITER = 10       # decay epsilon every 10 optimization steps
MIN_EPSILON = 0.1
LEARNING_RATE = 0.0003
BATCH_SIZE = 128
UPDATE_FREQ = 2               # optimize every 2 env steps
TARGET_NETWORK_UPDATE_FREQ = 500   # update target net every 500 optimization steps
BUFFER_LENGTH = 50000
REPLAY_WARMUP = 2000
PROGRESS_REWARD_SCALE = 8.0
TERMINAL_SUCCESS_BONUS = 8.0
TRUNCATION_PENALTY = 1.0
STUCK_PENALTY = 0.1
NUM_EPISODES = 5000
USE_HIGH_LEVEL_STATE = True


# ─── Replay Buffer ───
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ─── CNN Q-Network (for pixel input) ───
class ConvQNetwork(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1), nn.ReLU(),
            nn.Conv2d(256, 512, 4, 2, 1), nn.ReLU(),
        )
        self.head = nn.Linear(512, n_actions)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=[2, 3])
        return self.head(x)


# ─── MLP Q-Network (for high_level_state: 6-dim) ───
class MLPQNetwork(nn.Module):
    def __init__(self, state_dim, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


# ─── DQN Agent ───
class DQNAgent:
    def __init__(self, n_actions, device, use_high_level=True):
        self.n_actions = n_actions
        self.device = device
        self.epsilon = EPSILON
        self.update_count = 0

        state_dim = 6 if use_high_level else None
        if use_high_level:
            self.q_net = MLPQNetwork(state_dim, n_actions).to(device)
            self.target_net = MLPQNetwork(state_dim, n_actions).to(device)
        else:
            self.q_net = ConvQNetwork(n_actions).to(device)
            self.target_net = ConvQNetwork(n_actions).to(device)

        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(BUFFER_LENGTH)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            state_t = state.unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t)
            return q_values.argmax(dim=1).item()

    def update(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(BATCH_SIZE)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Double DQN: policy net picks action, target net evaluates
            next_actions = self.q_net(next_states).argmax(dim=1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q_values = rewards + GAMMA * next_q_values * (1 - dones)

        loss = nn.functional.smooth_l1_loss(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.q_net.parameters(), 100)
        self.optimizer.step()

        self.update_count += 1

        # Target network update (per optimization step, like sample1)
        if self.update_count % TARGET_NETWORK_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        # Epsilon decay (per optimization step)
        if self.update_count % EPSILON_DECAY_ITER == 0:
            self.epsilon = max(MIN_EPSILON, self.epsilon * EPSILON_DECAY)

        return loss.item()


# ─── Helpers ───
def get_obs(env):
    if USE_HIGH_LEVEL_STATE:
        return torch.tensor(env.high_level_state(), dtype=torch.float32)
    else:
        return env.state()


# ─── Training ───
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    env = Hw2Env(n_actions=N_ACTIONS, render_mode="offscreen")
    agent = DQNAgent(N_ACTIONS, device, use_high_level=USE_HIGH_LEVEL_STATE)

    episode_rewards = []
    episode_rps = []
    global_step = 0
    success_count = 0

    for episode in range(NUM_EPISODES):
        env.reset()
        obs = get_obs(env)
        prev_hs = env.high_level_state()
        prev_obj_goal_dist = np.linalg.norm(prev_hs[2:4] - prev_hs[4:6])
        done = False
        cumulative_reward = 0.0
        episode_steps = 0
        was_success = False

        while not done:
            action = agent.select_action(obs)

            _, reward, is_terminal, is_truncated = env.step(action)
            done = is_terminal or is_truncated

            next_obs = get_obs(env)
            curr_hs = env.high_level_state()
            curr_obj_goal_dist = np.linalg.norm(curr_hs[2:4] - curr_hs[4:6])

            # Reward shaping aimed at success, while still retaining the original reward.
            progress_bonus = (prev_obj_goal_dist - curr_obj_goal_dist) * PROGRESS_REWARD_SCALE
            train_reward = reward + progress_bonus
            if is_terminal:
                train_reward += TERMINAL_SUCCESS_BONUS
                was_success = True
            elif is_truncated:
                train_reward -= TRUNCATION_PENALTY

            # Penalize no-motion behavior to reduce local stuck loops near boundaries.
            ee_move = np.linalg.norm(curr_hs[0:2] - prev_hs[0:2])
            if ee_move < 1e-3:
                train_reward -= STUCK_PENALTY

            # Only mask future value on true terminal (goal reached), not truncation.
            agent.replay_buffer.push(obs, action, train_reward, next_obs, float(is_terminal))

            obs = next_obs
            prev_hs = curr_hs
            prev_obj_goal_dist = curr_obj_goal_dist
            cumulative_reward += reward
            episode_steps += 1
            global_step += 1

            if len(agent.replay_buffer) >= REPLAY_WARMUP and global_step % UPDATE_FREQ == 0:
                agent.update()

        success_count += int(was_success)
        rps = cumulative_reward / max(episode_steps, 1)
        episode_rewards.append(cumulative_reward)
        episode_rps.append(rps)

        print(f"Episode {episode:4d} | Reward: {cumulative_reward:8.3f} | "
              f"RPS: {rps:6.3f} | Eps: {agent.epsilon:.4f} | Steps: {episode_steps}")

        if (episode + 1) % 100 == 0:
            success_rate = success_count / 100.0
            print(f"Episodes {episode-98:4d}-{episode+1:4d} | Success rate: {success_rate:.2%}")
            success_count = 0

    # Save model
    torch.save(agent.q_net.state_dict(), "dqn_model.pt")
    print("Model saved to dqn_model.pt")

    # ─── Plotting ───
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    window = 100

    axes[0].plot(episode_rewards, alpha=0.3, label="Raw")
    if len(episode_rewards) >= window:
        smoothed = np.convolve(episode_rewards, np.ones(window)/window, mode="valid")
        axes[0].plot(range(window-1, len(episode_rewards)), smoothed, label=f"Moving Avg ({window})")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Cumulative Reward")
    axes[0].set_title("Reward per Episode")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(episode_rps, alpha=0.3, label="Raw")
    if len(episode_rps) >= window:
        smoothed = np.convolve(episode_rps, np.ones(window)/window, mode="valid")
        axes[1].plot(range(window-1, len(episode_rps)), smoothed, label=f"Moving Avg ({window})")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Reward per Step")
    axes[1].set_title("RPS per Episode")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("dqn_training_plots.png", dpi=150)
    print("Plots saved to dqn_training_plots.png")


# ─── Test / Evaluate learned policy ───
def test(model_path="dqn_model.pt", num_episodes=5, render_mode="gui"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = Hw2Env(n_actions=N_ACTIONS, render_mode=render_mode)

    if USE_HIGH_LEVEL_STATE:
        q_net = MLPQNetwork(6, N_ACTIONS).to(device)
    else:
        q_net = ConvQNetwork(N_ACTIONS).to(device)

    q_net.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    q_net.eval()

    successes = 0
    for episode in range(num_episodes):
        env.reset()
        obs = get_obs(env)
        done = False
        cumulative_reward = 0.0
        episode_steps = 0
        terminal = False

        while not done:
            with torch.no_grad():
                q_values = q_net(obs.unsqueeze(0).to(device))
                action = q_values.argmax(dim=1).item()

            _, reward, is_terminal, is_truncated = env.step(action)
            done = is_terminal or is_truncated
            terminal = is_terminal
            obs = get_obs(env)
            cumulative_reward += reward
            episode_steps += 1

        successes += int(terminal)
        rps = cumulative_reward / max(episode_steps, 1)
        print(f"Test Episode {episode} | Reward: {cumulative_reward:.3f} | "
              f"RPS: {rps:.3f} | Steps: {episode_steps}")
    print(f"Test success rate: {successes}/{num_episodes} = {successes/max(1, num_episodes):.2%}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        model = sys.argv[2] if len(sys.argv) > 2 else "dqn_model.pt"
        test(model_path=model)
    else:
        train()
