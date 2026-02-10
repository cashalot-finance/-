from __future__ import annotations

import argparse

from polymarket.calculations import daily_return, estimate_slippage, reward_from_return, risk_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket daily return calculator.")
    parser.add_argument("--price-now", type=float, required=True)
    parser.add_argument("--price-next", type=float, required=True)
    parser.add_argument("--volume", type=float, default=10000)
    parser.add_argument("--days", type=int, default=1)
    parser.add_argument("--fee", type=float, default=0.0015)
    parser.add_argument("--slippage-cap", type=float, default=0.01)
    parser.add_argument("--risk-weight", type=float, default=0.6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    slippage = estimate_slippage(args.volume, args.slippage_cap)
    ret = daily_return(args.price_now, args.price_next, args.fee, slippage)
    risk = risk_score(args.price_now, args.days)
    reward = reward_from_return(ret, risk, args.risk_weight)

    print(f"slippage={slippage:.6f}")
    print(f"return={ret:.6f}")
    print(f"risk={risk:.6f}")
    print(f"reward={reward:.6f}")


if __name__ == "__main__":
    main()
