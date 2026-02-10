from dataclasses import dataclass


@dataclass(frozen=True)
class DataConfig:
    csv_path: str = "polymarket_labeled_timeseries.csv"
    min_volume_num: float = 0.0
    min_days_to_expiry: int = 1
    max_days_to_expiry: int = 7
    top_k_by_price: int = 3
    slippage_cap: float = 0.01
    min_price: float = 0.9
    max_price: float = 0.999
    fee_rate: float = 0.0015
    risk_weight: float = 0.6
    min_expected_return: float = 0.002
    max_risk_score: float = 0.25
    max_positions_per_day: int = 3
    skip_penalty: float = 0.0
    overtrade_penalty: float = -0.01


@dataclass(frozen=True)
class TrainConfig:
    total_timesteps: int = 200_000
    learning_rate: float = 3e-4
    gamma: float = 0.995
    batch_size: int = 512
    n_steps: int = 2048
    seed: int = 42
    model_path: str = "models/polymarket_daily_ppo"
    initial_balance: float = 1.0
