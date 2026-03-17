# Homework 2 — DQN for Robotic Object Pushing

## How to Run

```bash
cd src
# Train
python dqn.py

# Test (GUI visualization)
python dqn.py test dqn_model_baseline.pt
```

## Approach & Hyperparameter Tuning

### Configuration

| Parameter | Value | Rationale |
|---|---|---|
| **Reward shaping** | **Two-phase**: EE→obj (scale 10) + obj→goal (scale 15) | Gives the robot a clear gradient to first approach the object, then push it toward the goal |
| N_ACTIONS | 16 | Finer directional control (22.5° apart), enabling more precise pushing angles |
| Network hidden size | 128 | Sufficient capacity to learn the two-phase reward signal |
| STUCK_PENALTY | 0.5 | Strongly discourages no-motion actions (stuck at boundaries) |
| TERMINAL_SUCCESS_BONUS | 20 | Amplifies positive signal from successful episodes |
| TRUNCATION_PENALTY | 2 | Stronger negative signal for failing to reach the goal |
| LEARNING_RATE | 0.0005 | Faster learning given the clear reward landscape |
| EPSILON_DECAY | 0.998 | Moderately fast transition from exploration to exploitation |
| MIN_EPSILON | 0.05 | Allows strong exploitation of the learned policy |
| UPDATE_FREQ | 1 | Learn from every environment step (dense updates) |
| TARGET_NETWORK_UPDATE_FREQ | 300 | Target network tracks online network at a stable pace |
| REPLAY_WARMUP | 1000 | Start learning sooner since early shaped rewards are already informative |
| BATCH_SIZE | 128 | Standard batch size for stable gradient estimates |
| GAMMA | 0.99 | Standard discount factor for long-horizon tasks |

### Discussion

- **Finer action space (16 vs 8):** With 8 actions, the closest push direction might be 22.5° off from the ideal push angle toward the goal. Doubling to 16 actions halves this alignment error, leading to more efficient pushes per step.
- **Larger stuck penalty (0.5 vs 0.1):** At 0.1 the penalty was dwarfed by noise in the shaped reward. Increasing to 0.5 makes "doing nothing" clearly the worst-case action, breaking oscillation loops.
- **Faster updates (UPDATE_FREQ=1, TARGET_NETWORK_UPDATE_FREQ=300):** With a clear reward signal, more frequent optimization steps help the agent converge faster without instability.

## Training Performance

![Training Plots](src/dqn_training_baseline.png)

The left plot shows cumulative reward per episode (with 100-episode moving average), and the right plot shows reward per step (RPS). Expected behavior: an initial exploration phase with low rewards, followed by a steady climb as the agent learns to approach and push the object.
