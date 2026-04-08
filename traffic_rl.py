import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

class TrafficEnv:
    def __init__(self):
        self.n_actions = 4
        self.n_states = 8
        self.max_queue = 20
        self.step_count = 0
        self.max_steps = 200

    def reset(self):
        self.queues = np.random.randint(0, 5, size=4).astype(np.float32)
        self.step_count = 0
        return self._get_state()

    def _get_state(self):
        return np.concatenate([self.queues / self.max_queue,
                                self.queues / (self.queues.sum() + 1e-5)])

    def step(self, action):
        self.step_count += 1
        cleared = np.zeros(4)
        cleared[action] = np.random.randint(3, 6)
        for i in range(4):
            if i != action:
                cleared[i] = np.random.randint(0, 2)
        arrivals = np.random.poisson(lam=2.0, size=4)
        self.queues = np.clip(self.queues - cleared + arrivals, 0, self.max_queue)
        total_wait = self.queues.sum()
        reward = -total_wait / (4 * self.max_queue)
        done = self.step_count >= self.max_steps
        return self._get_state(), reward, done

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

def train():
    env = TrafficEnv()
    agent = DQNAgent(state_dim=env.n_states, action_dim=env.n_actions)
    episodes = 300
    rewards_history = []
    target_update_freq = 10
    print("Training Smart Traffic RL Agent...\n")
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        while True:
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            agent.buffer.push(state, action, reward, next_state, done)
            agent.train_step()
            state = next_state
            total_reward += reward
            if done:
                break
        if ep % target_update_freq == 0:
            agent.update_target()
        rewards_history.append(total_reward)
        if ep % 50 == 0:
            avg = np.mean(rewards_history[-50:])
            print(f"Episode {ep:>3} | Total Reward: {total_reward:.2f} | Avg(50): {avg:.2f} | Epsilon: {agent.epsilon:.3f}")
    print("\nTraining Complete!")
    plt.figure(figsize=(10, 4))
    plt.plot(rewards_history, alpha=0.4, label="Episode Reward")
    window = 20
    smoothed = np.convolve(rewards_history, np.ones(window)/window, mode='valid')
    plt.plot(range(window-1, len(rewards_history)), smoothed, color='red', label=f"Smoothed ({window}-ep)")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.title("Smart Traffic Controller - DQN Training")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_results.png")
    plt.show()
    print("Plot saved as training_results.png")

if __name__ == "__main__":
    train()
