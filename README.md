# Smart Traffic Signal Controller — Deep RL

> Meta Scalar School of Technology Hackathon | Round 1 Submission  
> Team: Meta Morphosis | Track: Reinforcement Learning

---

## Problem Statement

Urban traffic congestion is a growing challenge in smart cities.  
Fixed-timer traffic signals waste time on empty lanes while congested lanes keep waiting.  
This project builds an RL agent that **learns to optimally control traffic signals** at a 4-way intersection to minimize total vehicle waiting time.

---

## Solution

A **Deep Q-Network (DQN)** agent trained on a custom Gymnasium environment that simulates a 4-lane intersection.  
The agent observes queue lengths and learns which lane to prioritize for the green signal.

---

## Environment Details

| Property | Value |
|---|---|
| Framework | Gymnasium (OpenEnv compatible) |
| Observation Space | Box(8,) — normalized queues + proportions |
| Action Space | Discrete(4) — one green phase per lane |
| Reward | Negative normalized total waiting cars |
| Max Steps | 200 per episode |

### Reward Logic
```
reward = -(total waiting cars) / (4 lanes × max queue size)
```
- Range: [-1.0, 0.0]
- Closer to 0 = better (fewer cars waiting)

---

## Results

The agent improved from **-160 reward → -118 reward** over 300 episodes, demonstrating clear learning.

![Training Results](training_results.png)

---

## Project Structure

```
smart-traffic-rl/
├── traffic_env.py       # Gymnasium environment
├── train.py             # DQN agent + training loop
├── demo.py              # Run trained agent
├── requirements.txt     # Dependencies
└── training_results.png # Training graph
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the agent
```bash
python train.py
```

### 3. Run the demo
```bash
python demo.py
```

---

## Tech Stack

- Python 3.10+
- PyTorch — DQN neural network
- Gymnasium — RL environment standard
- NumPy — numerical computation
- Matplotlib — training visualization

---

## Team Meta Morphosis

| Name |
|---|
| Shashidhar S Hugar |
| Sharan Kumar K S |
| Santosh Ryaka |

Institution: VVCE, Mysuru
