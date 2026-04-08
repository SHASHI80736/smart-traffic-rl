import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt


class TrafficEnv(gym.Env):
    """
    Smart Traffic Signal Controller - Mini RL Environment
    ======================================================
    A 4-way intersection where an RL agent controls signal phases
    to minimize total vehicle waiting time.

    Observation: 8-dim vector (normalized queue lengths + proportions)
    Action: Discrete(4) — which lane gets the green signal
    Reward: Negative of normalized total waiting cars
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None):
        super().__init__()

        self.n_lanes = 4
        self.max_queue = 20
        self.max_steps = 200
        self.render_mode = render_mode

        # Action space: 4 signal phases (one per lane)
        self.action_space = spaces.Discrete(self.n_lanes)

        # Observation space: normalized queues + proportions (8 values)
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(8,),
            dtype=np.float32
        )

        self.queues = None
        self.step_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.queues = self.np_random.integers(0, 5, size=self.n_lanes).astype(np.float32)
        self.step_count = 0
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        normalized = self.queues / self.max_queue
        proportions = self.queues / (self.queues.sum() + 1e-5)
        return np.concatenate([normalized, proportions]).astype(np.float32)

    def step(self, action):
        assert self.action_space.contains(action), f"Invalid action: {action}"
        self.step_count += 1

        # Green lane clears more cars
        cleared = np.zeros(self.n_lanes)
        cleared[action] = self.np_random.integers(3, 6)
        for i in range(self.n_lanes):
            if i != action:
                cleared[i] = self.np_random.integers(0, 2)

        # Random car arrivals
        arrivals = self.np_random.poisson(lam=2.0, size=self.n_lanes)
        self.queues = np.clip(self.queues - cleared + arrivals, 0, self.max_queue)

        total_wait = self.queues.sum()
        reward = float(-total_wait / (self.n_lanes * self.max_queue))

        terminated = False
        truncated = self.step_count >= self.max_steps

        obs = self._get_obs()
        info = {"total_queue": float(total_wait), "queues": self.queues.copy()}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            print(f"Step {self.step_count:>3} | Queues: {self.queues.astype(int)} | Total: {int(self.queues.sum())}")

    def close(self):
        pass
