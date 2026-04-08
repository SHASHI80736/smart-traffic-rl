# Smart Traffic Signal Controller using Deep RL

A reinforcement learning agent that learns to control 
traffic signals at a 4-way intersection using DQN (Deep Q-Network).

## Problem
Traffic congestion causes delays and pollution. 
Fixed signal timers waste time on empty lanes.

## Solution
A DQN agent that observes queue lengths on all 4 lanes 
and decides which lane gets the green signal — 
minimizing total waiting time.

## How it Works
- Environment: 4-lane intersection simulation
- Agent: Deep Q-Network (DQN) with PyTorch
- Reward: Negative of total waiting cars (less waiting = better)
- Training: 300 episodes, each with 200 steps

## Results
The agent improved from -160 reward to -118 reward over 300 episodes,
showing clear learning progress.

![Training Results](training_results.png)

## Tech Stack
- Python 3.10+
- PyTorch
- NumPy
- Matplotlib

## Run Locally
pip install torch numpy matplotlib
python traffic_rl.py

## Team
- Shashidhar S Hugar — Team Lead
- Sharan Kumar K S
- Santosh Ryaka
