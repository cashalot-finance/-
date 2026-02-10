from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

import numpy as np

from polymarket.calculations import daily_return, position_fraction, reward_from_return, risk_score
from polymarket.config import DataConfig
from polymarket.dataset import build_daily_frame


@dataclass
class BacktestResult:
    total_days: int
    trades: int
    win_rate: float
    total_return: float
    annualized_return: float
    max_drawdown: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rule-based backtest for daily low-risk strategy.")
    parser.add_argument("--csv", default=DataConfig.csv_path)
    parser.add_argument("--min-expected-return", type=float, default=DataConfig.min_expected_return)
    parser.add_argument("--max-risk-score", type=float, default=DataConfig.max_risk_score)
    parser.add_argument("--max-positions", type=int, default=DataConfig.max_positions_per_day)
    return parser.parse_args()


def backtest(config: DataConfig) -> BacktestResult:
    df = build_daily_frame(config)
    if df.empty:
        return BacktestResult(0, 0, 0.0, 0.0, 0.0, 0.0)

    df["slippage"] = df["slippage"].fillna(config.slippage_cap)
    df["risk_score"] = df.apply(
        lambda r: risk_score(float(r["price"]), int(r["days_to_expiry"])), axis=1
    )
    df["expected_return"] = df.apply(
        lambda r: daily_return(
            float(r["price"]),
            float(r["next_price"]),
            config.fee_rate,
            float(r["slippage"]),
        ),
        axis=1,
    )
    if "liquidity_score" not in df.columns:
        df["liquidity_score"] = 0.0
    if "volatility" not in df.columns:
        df["volatility"] = 0.0

    df = df[
        (df["expected_return"] >= config.min_expected_return)
        & (df["risk_score"] <= config.max_risk_score)
    ]

    if df.empty:
        return BacktestResult(0, 0, 0.0, 0.0, 0.0, 0.0)

    df = df.sort_values(["day_index", "expected_return"], ascending=[True, False])

    balance = 1.0
    balances: List[float] = [balance]
    trades = 0
    wins = 0

    for day_idx, day_df in df.groupby("day_index"):
        picks = day_df.head(config.max_positions_per_day)
        if picks.empty:
            balances.append(balance)
            continue

        daily_returns = []
        for _, row in picks.iterrows():
            ret = float(row["expected_return"])
            risk = float(row["risk_score"])
            liquidity = float(row.get("liquidity_score", 0.0))
            position_frac = position_fraction(liquidity, config.max_position_fraction)
            reward = reward_from_return(ret, risk, config.risk_weight) * position_frac
            daily_returns.append(reward)
            trades += 1
            if ret > 0:
                wins += 1

        avg_reward = float(np.mean(daily_returns)) if daily_returns else 0.0
        balance *= max(1.0 + avg_reward, 0.0)
        balances.append(balance)

    balances_arr = np.array(balances)
    max_balance = np.maximum.accumulate(balances_arr)
    drawdowns = (balances_arr - max_balance) / max_balance

    total_days = len(balances_arr) - 1
    total_return = balance - 1.0
    annualized = (1.0 + total_return) ** (365.0 / max(total_days, 1)) - 1.0
    max_drawdown = float(drawdowns.min()) if drawdowns.size else 0.0
    win_rate = wins / trades if trades else 0.0

    return BacktestResult(
        total_days=total_days,
        trades=trades,
        win_rate=win_rate,
        total_return=total_return,
        annualized_return=annualized,
        max_drawdown=max_drawdown,
    )


def main() -> None:
    args = parse_args()
    config = DataConfig(
        csv_path=args.csv,
        min_expected_return=args.min_expected_return,
        max_risk_score=args.max_risk_score,
        max_positions_per_day=args.max_positions,
    )
    result = backtest(config)
    print(f"days={result.total_days}")
    print(f"trades={result.trades}")
    print(f"win_rate={result.win_rate:.3f}")
    print(f"total_return={result.total_return:.4f}")
    print(f"annualized_return={result.annualized_return:.4f}")
    print(f"max_drawdown={result.max_drawdown:.4f}")


if __name__ == "__main__":
    main()
