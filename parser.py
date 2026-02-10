#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Polymarket resolved markets -> labeled price timeseries CSV (multi-outcome supported)

Key points:
- Gamma /markets provides outcomes, outcomePrices, umaResolutionStatus, clobTokenIds, endDate, volumeNum, etc.
- CLOB /prices-history provides history points {t, p} for a token_id.
- CLOB may reject too long (startTs,endTs) ranges with 400: "interval is too long".
  We handle that by chunking and adaptive splitting.

Docs:
- /prices-history params and mutual exclusivity of interval vs startTs/endTs:
  https://docs.polymarket.com/api-reference/pricing/get-price-history-for-a-traded-token
"""

import argparse
import csv
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ---------------- CONFIG ----------------
OUTPUT_FILE = "polymarket_labeled_timeseries.csv"
ERROR_LOG = "errors.log"

GAMMA_URL = "https://gamma-api.polymarket.com/markets"
CLOB_PRICES_URL = "https://clob.polymarket.com/prices-history"

SCAN_LIMIT_MARKETS = 2000
PAGE_SIZE = 100

MIN_VOLUME_NUM = 0

# Need 365 days of history before endDate
HISTORY_DAYS = 365

# We will request in chunks; CLOB rejects too-long ranges.
# Start with 30d chunks; if still too long, split further automatically.
INITIAL_CHUNK_DAYS = 30
MIN_CHUNK_DAYS = 1

# Fidelity = resolution in minutes (try denser -> sparser)
FIDELITIES = [60, 240, 1440]  # 1h, 4h, 1d

# Progress / pacing
PRINT_EVERY_N_REQUESTS = 50
SLEEP_BETWEEN_REQUESTS_SEC = 0.08
SLEEP_EVERY_N_MARKETS = 25
SLEEP_LONG_SEC = 0.6

# Hard timeouts: (connect_timeout, read_timeout)
HTTP_TIMEOUT = (10, 35)

# ---------------- HELPERS ----------------


def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (compatible; PolyLabeledParser/1.3)",
            "Accept": "application/json",
        }
    )
    retry = Retry(
        total=6,
        backoff_factor=0.7,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    s.mount("https://", HTTPAdapter(max_retries=retry))
    return s


def log_error(msg: str) -> None:
    ts = pd.Timestamp.utcnow().isoformat()
    line = f"[{ts}] {msg}\n"
    try:
        with open(ERROR_LOG, "a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass


def _json_load_maybe(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (list, dict)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except Exception:
            return x
    return x


def _to_str_list(xs: Any) -> Optional[List[str]]:
    xs = _json_load_maybe(xs)
    if not isinstance(xs, list):
        return None
    return [str(v) for v in xs]


def _to_float_list(xs: Any) -> Optional[List[float]]:
    xs = _json_load_maybe(xs)
    if not isinstance(xs, list):
        return None
    out: List[float] = []
    for v in xs:
        try:
            out.append(float(v))
        except Exception:
            return None
    return out


@dataclass
class MarketRecord:
    market_id: str
    question: str
    end_date: pd.Timestamp
    volume_num: float
    outcomes: List[str]
    token_ids: List[str]
    outcome_prices: List[float]
    winner_index: int
    winner_outcome: str
    winner_token_id: str


def extract_label_from_market(m: Dict[str, Any]) -> Optional[MarketRecord]:
    status = (m.get("umaResolutionStatus") or "").lower()
    if status != "resolved":
        return None

    market_id = str(m.get("id") or "").strip()
    question = str(m.get("question") or "").strip()
    end_date = pd.to_datetime(m.get("endDate"), utc=True, errors="coerce")

    if not market_id or not question or pd.isna(end_date):
        return None

    outcomes = _to_str_list(m.get("outcomes"))
    token_ids = _to_str_list(m.get("clobTokenIds") or m.get("clob_token_ids"))
    outcome_prices = _to_float_list(m.get("outcomePrices"))

    if not outcomes or not token_ids or not outcome_prices:
        return None
    if not (len(outcomes) == len(token_ids) == len(outcome_prices)) or len(outcomes) < 2:
        return None

    vol_num = m.get("volumeNum")
    if vol_num is None:
        try:
            vol_num = float(m.get("volume") or 0)
        except Exception:
            vol_num = 0.0

    vol_num = float(vol_num)
    if vol_num < float(MIN_VOLUME_NUM):
        return None

    winner_index = max(range(len(outcome_prices)), key=lambda i: outcome_prices[i])

    return MarketRecord(
        market_id=market_id,
        question=question,
        end_date=end_date,
        volume_num=vol_num,
        outcomes=outcomes,
        token_ids=token_ids,
        outcome_prices=outcome_prices,
        winner_index=int(winner_index),
        winner_outcome=outcomes[winner_index],
        winner_token_id=token_ids[winner_index],
    )


def fetch_resolved_markets(session: requests.Session) -> List[MarketRecord]:
    results: List[MarketRecord] = []
    offset = 0

    print("[INFO] Fetching resolved markets from Gamma /markets ...")
    while offset < SCAN_LIMIT_MARKETS:
        params = {
            "limit": PAGE_SIZE,
            "offset": offset,
            "closed": True,
            "order": "volume",
            "ascending": False,
        }

        try:
            r = session.get(GAMMA_URL, params=params, timeout=HTTP_TIMEOUT)
        except Exception as e:
            log_error(f"Gamma request failed offset={offset}: {repr(e)}")
            break

        if r.status_code != 200:
            log_error(f"Gamma status={r.status_code} offset={offset}: {r.text[:250]}")
            break

        data = r.json()
        if not data:
            break

        for m in data:
            rec = extract_label_from_market(m)
            if rec is not None:
                results.append(rec)

        offset += PAGE_SIZE
        time.sleep(0.2)

    print(f"[INFO] Resolved markets collected: {len(results)}")
    return results


def _fetch_prices_history_window(
    session: requests.Session,
    token_id: str,
    start_ts: int,
    end_ts: int,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Try to fetch one (startTs,endTs) window using different fidelities.
    Returns: (df or None, error_string_if_any)
    """
    for fid in FIDELITIES:
        params = {
            "market": token_id,
            "startTs": start_ts,
            "endTs": end_ts,
            "fidelity": fid,
        }
        try:
            r = session.get(CLOB_PRICES_URL, params=params, timeout=HTTP_TIMEOUT)
        except Exception as e:
            return None, f"request failed token={token_id} fid={fid}: {repr(e)}"

        if r.status_code != 200:
            # keep the error body to detect "interval is too long"
            return None, f"status={r.status_code} token={token_id} fid={fid}: {r.text[:200]}"

        payload = r.json()
        hist = payload.get("history") or []
        if not hist:
            # empty is not necessarily error; treat as success-but-empty
            return pd.DataFrame(columns=["t", "p", "fidelity_min"]), None

        df = pd.DataFrame(hist)
        if "t" not in df.columns or "p" not in df.columns:
            return None, f"bad response shape token={token_id} fid={fid}"

        df["t"] = pd.to_datetime(df["t"], unit="s", utc=True, errors="coerce")
        df["p"] = pd.to_numeric(df["p"], errors="coerce")
        df = df.dropna(subset=["t", "p"])

        df["fidelity_min"] = fid
        return df[["t", "p", "fidelity_min"]], None

    return None, f"no fidelity worked token={token_id}"


def fetch_prices_history_365d_chunked(
    session: requests.Session,
    token_id: str,
    end_date: pd.Timestamp,
) -> Optional[pd.DataFrame]:
    """
    Fetch last HISTORY_DAYS days before end_date using chunking and adaptive splitting.
    CLOB rejects too-long (startTs,endTs) ranges, so we:
      - walk from end_date backwards in chunks
      - if a chunk fails with "interval is too long", split chunk in half, etc.
    """
    if end_date.tzinfo is None:
        end_date = end_date.tz_localize("UTC")
    else:
        end_date = end_date.tz_convert("UTC")

    start_date_global = end_date - pd.Timedelta(days=HISTORY_DAYS)

    # We'll move a cursor backwards from end_date to start_date_global
    cursor_end = end_date

    dfs: List[pd.DataFrame] = []

    while cursor_end > start_date_global:
        # initial chunk size
        chunk_days = INITIAL_CHUNK_DAYS

        # define tentative chunk start
        while True:
            cursor_start = cursor_end - pd.Timedelta(days=chunk_days)
            if cursor_start < start_date_global:
                cursor_start = start_date_global

            start_ts = int(cursor_start.timestamp())
            end_ts = int(cursor_end.timestamp())

            df, err = _fetch_prices_history_window(session, token_id, start_ts, end_ts)

            if err is None:
                # success (df may be empty)
                if df is not None and not df.empty:
                    dfs.append(df)
                # move cursor back
                cursor_end = cursor_start
                break

            # if interval too long -> split chunk
            if "interval is too long" in err:
                # reduce chunk
                new_chunk_days = max(MIN_CHUNK_DAYS, chunk_days // 2)
                if new_chunk_days == chunk_days:
                    # can't reduce further
                    log_error(f"CLOB still says too long at MIN_CHUNK_DAYS token={token_id}: {err}")
                    return None
                chunk_days = new_chunk_days
                continue

            # other errors -> log and give up this token
            log_error(f"CLOB error token={token_id}: {err}")
            return None

        time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)

    if not dfs:
        return None

    out = pd.concat(dfs, ignore_index=True)
    out.drop_duplicates(subset=["t"], inplace=True)
    out.sort_values("t", inplace=True)
    return out


def init_csv(path: str, append: bool) -> None:
    if append:
        return
    header = [
        "market_id",
        "question",
        "end_date",
        "token_id",
        "outcome_name",
        "t",
        "p",
        "hours_left",
        "winner_index",
        "winner_outcome",
        "winner_token_id",
        "is_winner_token",
        "n_outcomes",
        "volume_num",
        "fidelity_min",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(header)


def append_rows_csv(path: str, rows: List[List[Any]]) -> None:
    if not rows:
        return
    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Polymarket price history with labels.")
    parser.add_argument("--output", default=OUTPUT_FILE, help="CSV output file.")
    parser.add_argument("--errors", default=ERROR_LOG, help="Error log file.")
    parser.add_argument("--history-days", type=int, default=HISTORY_DAYS)
    parser.add_argument("--min-volume", type=float, default=MIN_VOLUME_NUM)
    parser.add_argument("--scan-limit", type=int, default=SCAN_LIMIT_MARKETS)
    parser.add_argument("--page-size", type=int, default=PAGE_SIZE)
    parser.add_argument("--resume", action="store_true", help="Append to existing CSV if it exists.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global OUTPUT_FILE, ERROR_LOG, HISTORY_DAYS, MIN_VOLUME_NUM, SCAN_LIMIT_MARKETS, PAGE_SIZE
    OUTPUT_FILE = args.output
    ERROR_LOG = args.errors
    HISTORY_DAYS = args.history_days
    MIN_VOLUME_NUM = args.min_volume
    SCAN_LIMIT_MARKETS = args.scan_limit
    PAGE_SIZE = args.page_size

    session = make_session()

    markets = fetch_resolved_markets(session)
    if not markets:
        print("[ERROR] No resolved markets found.")
        return

    total_tokens = sum(len(m.token_ids) for m in markets)
    print(f"[INFO] Total token_ids to fetch: {total_tokens}")
    print(f"[INFO] History window: last {HISTORY_DAYS} days before each market endDate")

    init_csv(OUTPUT_FILE, append=args.resume)
    print("[INFO] Downloading prices-history for tokens ...")

    req_i = 0
    token_series_ok = 0
    rows_buffer: List[List[Any]] = []

    for mi, m in enumerate(markets, 1):
        token_to_outcome = {tid: m.outcomes[idx] for idx, tid in enumerate(m.token_ids)}

        for token_id in m.token_ids:
            req_i += 1

            df = fetch_prices_history_365d_chunked(session, token_id, m.end_date)
            if df is None:
                # no data or error already logged
                if req_i % PRINT_EVERY_N_REQUESTS == 0:
                    print(f"[INFO] progress: {req_i}/{total_tokens} token histories | ok={token_series_ok}")
                continue

            # compute hours_left and keep points before end_date
            df["hours_left"] = (m.end_date - df["t"]).dt.total_seconds() / 3600.0
            df = df[df["hours_left"] > 0]
            if df.empty:
                if req_i % PRINT_EVERY_N_REQUESTS == 0:
                    print(f"[INFO] progress: {req_i}/{total_tokens} token histories | ok={token_series_ok}")
                continue

            token_series_ok += 1

            for _, r in df.iterrows():
                rows_buffer.append(
                    [
                        m.market_id,
                        m.question,
                        m.end_date.isoformat(),
                        token_id,
                        token_to_outcome.get(token_id, ""),
                        r["t"].isoformat(),
                        float(r["p"]),
                        float(r["hours_left"]),
                        m.winner_index,
                        m.winner_outcome,
                        m.winner_token_id,
                        int(token_id == m.winner_token_id),
                        len(m.outcomes),
                        m.volume_num,
                        int(r["fidelity_min"]),
                    ]
                )

            if len(rows_buffer) >= 5000:
                append_rows_csv(OUTPUT_FILE, rows_buffer)
                rows_buffer.clear()

            if req_i % PRINT_EVERY_N_REQUESTS == 0:
                print(
                    f"[INFO] progress: {req_i}/{total_tokens} token histories | "
                    f"series_ok={token_series_ok} | markets_done={mi}/{len(markets)}"
                )

        if mi % SLEEP_EVERY_N_MARKETS == 0:
            time.sleep(SLEEP_LONG_SEC)

    if rows_buffer:
        append_rows_csv(OUTPUT_FILE, rows_buffer)
        rows_buffer.clear()

    print("[INFO] DONE")
    print(f"[INFO] token series fetched: {token_series_ok}")
    print(f"[INFO] CSV saved: {OUTPUT_FILE}")
    print(f"[INFO] Errors (if any): {ERROR_LOG}")


if __name__ == "__main__":
    main()
