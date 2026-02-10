from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple


@dataclass(frozen=True)
class OrderLevel:
    price: float
    size: float


@dataclass(frozen=True)
class FillResult:
    filled_size: float
    avg_price: float
    remaining: float


def vwap_fill(levels: Iterable[OrderLevel], target_size: float) -> FillResult:
    remaining = target_size
    total_cost = 0.0
    filled = 0.0

    for level in levels:
        if remaining <= 0:
            break
        take = min(level.size, remaining)
        total_cost += take * level.price
        filled += take
        remaining -= take

    avg_price = total_cost / filled if filled > 0 else 0.0
    return FillResult(filled_size=filled, avg_price=avg_price, remaining=remaining)


def levels_from_book(raw_levels: Iterable[Tuple[float, float]]) -> List[OrderLevel]:
    return [OrderLevel(price=float(p), size=float(s)) for p, s in raw_levels]
