from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd

from polymarket.calculations import estimate_slippage, risk_score
from polymarket.config import DataConfig


@dataclass
class DailySample:
    market_id: str
    token_id: str
    question: str
    day_index: int
    days_to_expiry: int
    price: float
    next_price: float
    volume_num: float
    n_outcomes: int
    is_winner_token: int
    rank_by_price: int
    slippage: float
    risk_score: float


class DailyDataset:
    def __init__(self, samples: List[DailySample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DailySample:
        return self.samples[idx]


def _slippage_from_volume(volume_num: float, cap: float) -> float:
    if volume_num <= 0:
        return cap
    return min(cap, 1.0 / (1.0 + np.log1p(volume_num)))


def load_daily_dataset(config: DataConfig) -> Tuple[DailyDataset, np.ndarray]:
    df = pd.read_csv(config.csv_path)

    df["t"] = pd.to_datetime(df["t"], utc=True, errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], utc=True, errors="coerce")
    df = df.dropna(subset=["t", "end_date", "p"])\
        .sort_values(["market_id", "token_id", "t"])

    df = df[df["volume_num"] >= config.min_volume_num]
    df = df[(df["p"] >= config.min_price) & (df["p"] <= config.max_price)]

    df["hours_left"] = (df["end_date"] - df["t"]).dt.total_seconds() / 3600.0
    df = df[df["hours_left"] > 0]

    df["day_index"] = (df["hours_left"] // 24).astype(int)
    df["days_to_expiry"] = df["day_index"].clip(lower=0)

    df = df[
        (df["days_to_expiry"] >= config.min_days_to_expiry)
        & (df["days_to_expiry"] <= config.max_days_to_expiry)
    ]

    if df.empty:
        return DailyDataset([]), np.zeros((0, 7), dtype=np.float32)

    daily = (
        df.groupby(["market_id", "token_id", "day_index"])\
        .agg(
            question=("question", "last"),
            price=("p", "last"),
            end_date=("end_date", "last"),
            volume_num=("volume_num", "last"),
            n_outcomes=("n_outcomes", "last"),
            is_winner_token=("is_winner_token", "last"),
        )
        .reset_index()
    )

    daily["days_to_expiry"] = daily["day_index"]

    daily = daily.sort_values(["market_id", "token_id", "day_index"])
    daily["next_price"] = daily.groupby(["market_id", "token_id"])["price"].shift(-1)
    daily = daily.dropna(subset=["next_price"])

    daily["rank_by_price"] = (
        daily.groupby(["market_id", "day_index"])["price"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    daily = daily[daily["rank_by_price"] <= config.top_k_by_price]
    daily["slippage"] = daily["volume_num"].apply(lambda v: estimate_slippage(v, config.slippage_cap))
    daily["risk_score"] = daily.apply(
        lambda r: risk_score(float(r["price"]), int(r["days_to_expiry"])), axis=1
    )

    samples: List[DailySample] = []
    obs_rows: List[np.ndarray] = []

    for _, row in daily.iterrows():
        samples.append(
            DailySample(
                market_id=str(row["market_id"]),
                token_id=str(row["token_id"]),
                question=str(row["question"]),
                day_index=int(row["day_index"]),
                days_to_expiry=int(row["days_to_expiry"]),
                price=float(row["price"]),
                next_price=float(row["next_price"]),
                volume_num=float(row["volume_num"]),
                n_outcomes=int(row["n_outcomes"]),
                is_winner_token=int(row["is_winner_token"]),
                rank_by_price=int(row["rank_by_price"]),
                slippage=float(row["slippage"]),
                risk_score=float(row["risk_score"]),
            )
        )
        obs_rows.append(
            np.array(
                [
                    float(row["price"]),
                    float(row["days_to_expiry"]),
                    float(row["volume_num"]),
                    float(row["n_outcomes"]),
                    float(row["rank_by_price"]),
                    float(row["slippage"]),
                    float(row["risk_score"]),
                ],
                dtype=np.float32,
            )
        )

    return DailyDataset(samples), np.vstack(obs_rows)


def compute_reward(sample: DailySample, config: DataConfig) -> float:
    slippage = estimate_slippage(sample.volume_num, config.slippage_cap)
    effective_buy = sample.price * (1.0 + slippage + config.fee_rate)
    effective_sell = sample.next_price * (1.0 - slippage - config.fee_rate)
    raw_return = (effective_sell - effective_buy) / max(effective_buy, 1e-8)

    risk = risk_score(sample.price, sample.days_to_expiry)
    return float(np.log1p(np.clip(raw_return, -0.99, 10.0)) - config.risk_weight * risk)
