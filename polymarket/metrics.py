from __future__ import annotations

import argparse

import pandas as pd

from polymarket.config import DataConfig
from polymarket.dataset import build_daily_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dataset quality report for Polymarket daily strategy.")
    parser.add_argument("--csv", default=DataConfig.csv_path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DataConfig(csv_path=args.csv)
    df = build_daily_frame(config)
    if df.empty:
        print("dataset_empty=1")
        return

    report = {
        "rows": len(df),
        "markets": df["market_id"].nunique(),
        "tokens": df["token_id"].nunique(),
        "avg_price": df["price"].mean(),
        "avg_volume": df["volume_num"].mean(),
        "avg_risk": df["risk_score"].mean(),
        "avg_liquidity": df["liquidity_score"].mean(),
        "avg_volatility": df.get("volatility", pd.Series([0.0])).mean(),
        "avg_expected_return": df["expected_return"].mean(),
    }

    for k, v in report.items():
        if isinstance(v, float):
            print(f"{k}={v:.6f}")
        else:
            print(f"{k}={v}")


if __name__ == "__main__":
    main()
