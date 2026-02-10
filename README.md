# Polymarket Daily RL

Небольшой проект для сбора исторических данных Polymarket и обучения RL-модели, ориентированной на **ежедневные низкорисковые сделки**.

## Состав
- `parser.py` — сборщик данных (Gamma markets + CLOB price history) с метками исходов.
- `polymarket/dataset.py` — подготовка дневных сэмплов, фильтры по риску/ликвидности.
- `polymarket/env.py` — минимальная Gym-среда: решение «покупать/пропускать».
- `polymarket/train.py` — обучение PPO.
- `polymarket/infer.py` — ранжирование возможностей по обученной модели.
- `polymarket/calculator.py` — калькулятор доходности/риска для быстрой проверки гипотез.
- `polymarket/backtest.py` — простой бэктест стратегии (фильтр по доходности/риску).
- `polymarket/execution.py` — утилиты для расчёта средней цены исполнения по стакану.

## Установка
```bash
pip install -r requirements.txt
```

## Сбор данных
```bash
python parser.py --output polymarket_labeled_timeseries.csv --history-days 365 --min-volume 0
```

## Обучение
```bash
python -m polymarket.train --csv polymarket_labeled_timeseries.csv --total-timesteps 200000
```

## Инференс
```bash
python -m polymarket.infer --csv polymarket_labeled_timeseries.csv --model models/polymarket_daily_ppo
```

## Калькулятор
```bash
python -m polymarket.calculator --price-now 0.97 --price-next 0.99 --volume 50000 --days 1
```

## Бэктест
```bash
python -m polymarket.backtest --csv polymarket_labeled_timeseries.csv --min-expected-return 0.002 --max-risk-score 0.25
```

## Логика отбора
- Сэмплы строятся на дневных шагах до экспирации.
- Для каждого дня берутся **топ-N по цене** исходы (низкий риск).
- Дополнительно используются фильтры по цене (по умолчанию 0.9–0.999), объёму, плюс риск‑скор по сроку и цене.
- Агент решает: купить или пропустить. Вознаграждение — лог‑доходность с учётом slippage, комиссии и риска.
  Компаундинг моделируется через баланс внутри среды.

> Важно: модель и среда — минимальный рабочий каркас. Для продакшена нужны полноценные стаканы, риск-менеджмент и проверка ликвидности.
