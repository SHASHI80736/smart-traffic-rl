import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

from traffic_env import TrafficEnv


# ─────────────────────────────────────────
# DQN Neural Network
# ─────────────────────────────────────────
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────
# Replay Buffer
# ─────────────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions),
                np.array(rewards), np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)


# ─────────────────────────────────────────
# DQN Agent
# ─────────────────────────────────────────
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.action_dim = action_dim
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995
        self.lr = 1e-3
        self.batch_size = 64

        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.buffer = ReplayBuffer()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.policy_net(state_t).argmax().item()

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        states_t      = torch.FloatTensor(states)
        actions_t     = torch.LongTensor(actions).unsqueeze(1)
        rewards_t     = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states)
        dones_t       = torch.FloatTensor(dones).unsqueeze(1)

        q_values = self.policy_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(1, keepdim=True)[0]
            target_q   = rewards_t + self.gamma * max_next_q * (1 - dones_t)

        loss = nn.MSELoss()(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path="model.pth"):
        torch.save(self.policy_net.state_dict(), path)
        print(f"Model saved to {path}")


# ─────────────────────────────────────────
# Training Loop
# ─────────────────────────────────────────
def train():
    env = TrafficEnv()
    obs, _ = env.reset()

    agent = DQNAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )

    episodes = 300
    rewards_history = []
    target_update_freq = 10

    print("Training Smart Traffic RL Agent...\n")

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0

        while True:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.buffer.push(obs, action, reward, next_obs, done)
            agent.train_step()

            obs = next_obs
            total_reward += reward

            if done:
                break

        if ep % target_update_freq == 0:
            agent.update_target()

        rewards_history.append(total_reward)

        if ep % 50 == 0:
            avg = np.mean(rewards_history[-50:])
            print(f"Episode {ep:>3} | Reward: {total_reward:.2f} | Avg(50): {avg:.2f} | Epsilon: {agent.epsilon:.3f}")

    print("\nTraining Complete!")
    agent.save("model.pth")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(rewards_history, alpha=0.4, color='steelblue', label="Episode Reward")
    window = 20
    smoothed = np.convolve(rewards_history, np.ones(window) / window, mode='valid')
    plt.plot(range(window - 1, len(rewards_history)), smoothed, color='red', label=f"Smoothed ({window}-ep)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Smart Traffic Controller - DQN Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_results.png")
    plt.show()
    print("Plot saved as training_results.png")

    env.close()


if __name__ == "__main__":
    train()
