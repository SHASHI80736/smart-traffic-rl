"""
demo.py — Run the trained Smart Traffic RL Agent
=================================================
This script loads the saved model and runs it for 5 episodes,
printing step-by-step signal decisions and queue states.
"""

import torch
import numpy as np
from traffic_env import TrafficEnv
from train import DQN


def run_demo(model_path="model.pth", episodes=5):
    env = TrafficEnv(render_mode="human")
    obs, _ = env.reset()

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Load trained model
    model = DQN(state_dim, action_dim)
    try:
        model.load_state_dict(torch.load(model_path, weights_only=True))
        print(f"Loaded trained model from {model_path}\n")
    except FileNotFoundError:
        print("No saved model found. Run train.py first!\n")
        return

    model.eval()
    lane_names = ["North", "South", "East", "West"]

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        print(f"\n{'='*50}")
        print(f"Episode {ep + 1}")
        print(f"{'='*50}")

        while True:
            state_t = torch.FloatTensor(obs).unsqueeze(0)
            with torch.no_grad():
                action = model(state_t).argmax().item()

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            print(f"  Green: {lane_names[action]:<6} | Queues: {info['queues'].astype(int)} | Total wait: {int(info['total_queue'])}")

            if terminated or truncated:
                break

        print(f"\nEpisode {ep + 1} Total Reward: {total_reward:.2f}")

    env.close()
    print("\nDemo complete!")


if __name__ == "__main__":
    run_demo()
