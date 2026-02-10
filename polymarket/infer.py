from __future__ import annotations

import argparse

import numpy as np
from stable_baselines3 import PPO

from polymarket.config import DataConfig
from polymarket.dataset import load_daily_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank daily opportunities with trained PPO model.")
    parser.add_argument("--csv", default=DataConfig.csv_path)
    parser.add_argument("--model", default="models/polymarket_daily_ppo")
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = DataConfig(csv_path=args.csv)
    dataset, obs = load_daily_dataset(data_cfg)

    if len(dataset) == 0:
        raise SystemExit("Dataset is empty. Run parser.py first or adjust filters.")

    model = PPO.load(args.model)

    balance_feature = np.full((obs.shape[0], 1), 1.0, dtype=np.float32)
    obs_with_balance = np.hstack([obs, balance_feature]) if obs.size else obs

    actions, _ = model.predict(obs_with_balance, deterministic=True)
    scores = []
    for sample, action in zip(dataset.samples, actions):
        if int(action) == 1:
            scores.append(
                {
                    "market_id": sample.market_id,
                    "token_id": sample.token_id,
                    "question": sample.question,
                    "price": sample.price,
                    "days_to_expiry": sample.days_to_expiry,
                    "volume_num": sample.volume_num,
                }
            )

    scores = sorted(scores, key=lambda x: (x["price"], -x["volume_num"]), reverse=True)

    print("[INFO] Selected opportunities:")
    for row in scores[: args.top]:
        print(
            f"{row['question']} | price={row['price']:.4f} | days={row['days_to_expiry']} | volume={row['volume_num']:.0f}"
        )


if __name__ == "__main__":
    main()
