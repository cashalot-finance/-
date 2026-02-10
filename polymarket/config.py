from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    csv_path: str = "polymarket_labeled_timeseries.csv"
    min_volume_num: float = 0.0
    min_days_to_expiry: int = 1
    max_days_to_expiry: int = 7
    top_k_by_price: int = 3
    slippage_cap: float = 0.01


@dataclass(frozen=True)
class TrainConfig:
    total_timesteps: int = 200_000
    learning_rate: float = 3e-4
    gamma: float = 0.995
    batch_size: int = 512
    n_steps: int = 2048
    seed: int = 42
    model_path: str = "models/polymarket_daily_ppo"
