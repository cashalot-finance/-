from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from polymarket.config import DataConfig, TrainConfig
from polymarket.dataset import load_daily_dataset
from polymarket.env import PolymarketDailyEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Polymarket daily PPO agent.")
    parser.add_argument("--csv", default=DataConfig.csv_path, help="Path to labeled timeseries CSV.")
    parser.add_argument("--total-timesteps", type=int, default=TrainConfig.total_timesteps)
    parser.add_argument("--model-path", default=TrainConfig.model_path)
    parser.add_argument("--seed", type=int, default=TrainConfig.seed)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = DataConfig(csv_path=args.csv)
    train_cfg = TrainConfig(total_timesteps=args.total_timesteps, seed=args.seed, model_path=args.model_path)

    dataset, _ = load_daily_dataset(data_cfg)
    if len(dataset) == 0:
        raise SystemExit("Dataset is empty. Run parser.py first or adjust filters.")

    env = DummyVecEnv([lambda: PolymarketDailyEnv(dataset, data_cfg)])

    np.random.seed(train_cfg.seed)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=train_cfg.learning_rate,
        gamma=train_cfg.gamma,
        batch_size=train_cfg.batch_size,
        n_steps=train_cfg.n_steps,
        seed=train_cfg.seed,
    )

    model.learn(total_timesteps=train_cfg.total_timesteps)

    model_path = Path(train_cfg.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path.as_posix())
    print(f"[INFO] Model saved to {model_path}")


if __name__ == "__main__":
    main()
