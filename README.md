# Polymarket Daily RL

Небольшой проект для сбора исторических данных Polymarket и обучения RL-модели, ориентированной на **ежедневные низкорисковые сделки**.

## Состав
- `parser.py` — сборщик данных (Gamma markets + CLOB price history) с метками исходов.
- `polymarket/dataset.py` — подготовка дневных сэмплов, фильтры по риску/ликвидности.
- `polymarket/env.py` — минимальная Gym-среда: решение «покупать/пропускать».
- `polymarket/train.py` — обучение PPO.
- `polymarket/infer.py` — ранжирование возможностей по обученной модели.

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

## Логика отбора
- Сэмплы строятся на дневных шагах до экспирации.
- Для каждого дня берутся **топ-N по цене** исходы (низкий риск).
- Дополнительно используются фильтры по цене (по умолчанию 0.9–0.999) и объёму.
- Агент решает: купить или пропустить. Вознаграждение — дневная доходность с учётом слippage, штрафа за риск и компаундинга.

> Важно: модель и среда — минимальный рабочий каркас. Для продакшена нужны полноценные стаканы, риск-менеджмент и проверка ликвидности.
