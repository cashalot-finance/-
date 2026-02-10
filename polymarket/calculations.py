from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class PriceLadder:
    price_now: float
    price_next: float
    fee_rate: float
    slippage: float


def estimate_slippage(volume_num: float, cap: float) -> float:
    if volume_num <= 0:
        return cap
    return min(cap, 1.0 / (1.0 + np.log1p(volume_num)))


def effective_prices(price_now: float, price_next: float, fee_rate: float, slippage: float) -> PriceLadder:
    buy = price_now * (1.0 + slippage + fee_rate)
    sell = price_next * (1.0 - slippage - fee_rate)
    return PriceLadder(price_now=buy, price_next=sell, fee_rate=fee_rate, slippage=slippage)


def daily_return(price_now: float, price_next: float, fee_rate: float, slippage: float) -> float:
    ladder = effective_prices(price_now, price_next, fee_rate, slippage)
    return (ladder.price_next - ladder.price_now) / max(ladder.price_now, 1e-8)


def risk_score(price_now: float, days_to_expiry: int) -> float:
    """
    Heuristic risk score:
    - higher price -> lower risk
    - shorter horizon -> lower risk
    """
    price_risk = 1.0 - price_now
    horizon_risk = min(max(days_to_expiry / 7.0, 0.0), 1.0)
    return 0.7 * price_risk + 0.3 * horizon_risk


def reward_from_return(ret: float, risk: float, risk_weight: float) -> float:
    safe_ret = np.clip(ret, -0.99, 10.0)
    return float(np.log1p(safe_ret) - risk_weight * risk)


def liquidity_score(volume_num: float) -> float:
    if volume_num <= 0:
        return 0.0
    return float(min(1.0, np.log1p(volume_num) / 12.0))


def position_fraction(liquidity: float, max_fraction: float) -> float:
    return float(min(max_fraction, max(0.0, liquidity)))


def target_sell_price(price_now: float, desired_return: float) -> float:
    return float(price_now * (1.0 + desired_return))


def price_bands(price_now: float) -> Tuple[float, float]:
    return float(min(0.999, price_now + 0.02)), float(max(0.01, price_now - 0.05))
