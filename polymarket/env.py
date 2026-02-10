from __future__ import annotations

from typing import Tuple

import gymnasium as gym
import numpy as np

from polymarket.config import DataConfig
from polymarket.dataset import DailyDataset, compute_reward


class PolymarketDailyEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, dataset: DailyDataset, config: DataConfig, initial_balance: float = 1.0):
        super().__init__()
        self.dataset = dataset
        self.config = config
        self.index = 0
        self.initial_balance = initial_balance
        self.balance = initial_balance

        # Observation: price, days_to_expiry, volume, n_outcomes, rank, slippage, risk, balance
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, 0, 2, 1, 0, 0, 0], dtype=np.float32),
            high=np.array([1, 30, 1e9, 50, 10, 0.2, 1, 1000], dtype=np.float32),
        )

        # Action: 0 = skip, 1 = buy
        self.action_space = gym.spaces.Discrete(2)

    def _get_obs(self) -> np.ndarray:
        sample = self.dataset[self.index]
        return np.array(
            [
                sample.price,
                float(sample.days_to_expiry),
                sample.volume_num,
                float(sample.n_outcomes),
                float(sample.rank_by_price),
                float(sample.slippage),
                float(sample.risk_score),
                float(self.balance),
            ],
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options=None) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.index = 0
        self.balance = self.initial_balance
        if len(self.dataset) == 0:
            return np.zeros(self.observation_space.shape, dtype=np.float32), {}
        return self._get_obs(), {}

    def step(self, action: int):
        if len(self.dataset) == 0:
            obs = np.zeros(self.observation_space.shape, dtype=np.float32)
            return obs, 0.0, True, False, {}

        sample = self.dataset[self.index]
        reward = 0.0
        if action == 1:
            daily_reward = compute_reward(sample, self.config)
            self.balance *= max(1.0 + daily_reward, 0.0)
            reward = daily_reward

        self.index += 1
        terminated = self.index >= len(self.dataset)
        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, reward, terminated, False, {}
