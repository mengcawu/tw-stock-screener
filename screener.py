"""
Taiwan Stock Screener

This script fetches up to 60 days of price data from the FinMind API,
filters stocks with closing prices within a given range, identifies
stocks with short-term uptrends (5-day MA above 20-day MA and price above
MA5), calculates momentum and RSI, optionally retrieves PER and PBR, and
generates a markdown report.

"""

from __future__ import annotations

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import requests  # type: ignore


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def fetch_dataset(
    dataset: str,
    token: str,
    params: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    url = "https://api.finmindtrade.com/api/v4/data"
    headers: Dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    query: Dict[str, str] = {"dataset": dataset}
    if params:
        query.update(params)
    try:
        resp = requests.get(url, headers=headers, params=query, timeout=30)
        resp.raise_for_status()
    except Exception as exc:
        logging.error(f"Failed to fetch {dataset}: {exc}")
        return pd.DataFrame()
    data = resp.json().get("data", [])
    return pd.DataFrame(data)


def compute_moving_averages(df: pd.DataFrame, window: int, value_col: str = "close") -> pd.Series:
    return df[value_col].rolling(window=window, min_periods=window).mean()


def compute_rsi(df: pd.DataFrame, periods: int = 14) -> pd.Series:
    delta = df["close"].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(window=periods, min_periods=periods).mean()
    roll_down = down.rolling(window=periods, min_periods=periods).mean()
    rs = roll_up / roll_down
    return 100 - (100 / (1 + rs))


def analyse_stock(
    price_history: pd.DataFrame,
    per_history: Optional[pd.DataFrame] = None,
    price_range: Tuple[float, float] = (10.0, 40.0),
) -> Optional[Dict[str, object]]:
    if price_history.empty:
        return None
    price_history = price_history.sort_values("date")
    latest_close = price_history["close"].iloc[-1]
    low, high = price_range
    if not (low <= latest_close <= high):
        return None
    ma5 = compute_moving_averages(price_history, 5)
    ma20 = compute_moving_averages(price_history, 20)
    if len(ma20.dropna()) == 0:
        return None
    if not (ma5.iloc[-1] > ma20.iloc[-1] and latest_close > ma5.iloc[-1]):
        return None
    returns = price_history["close"].pct_change()
    momentum = returns.tail(3).sum()
    rsi_series = compute_rsi(price_history, periods=14)
    current_rsi = rsi_series.iloc[-1] if not rsi_series.empty else np.nan
    per = np.nan
    pbr = np.nan
    dividend_yield = np.nan
    if per_history is not None and not per_history.empty:
        per_history = per_history.sort_values("date")
        latest_per_row = per_history.iloc[-1]
        per = latest_per_row.get("PER", np.nan)
        pbr = latest_per_row.get("PBR", np.nan)
        dividend_yield = latest_per_row.get("dividend_yield", np.nan)
    return {
        "stock_id": price_history["stock_id"].iloc[-1],
        "latest_close": latest_close,
        "ma5": ma5.iloc[-1],
        "ma20": ma20.iloc[-1],
        "momentum3": momentum,
        "rsi14": current_rsi,
        "PER": per,
        "PBR": pbr,
        "dividend_yield": dividend_yield,
    }


def generate_report(candidates: List[Dict[str, object]], report_path: str) -> None:
    candidates_sorted = sorted(candidates, key=lambda x: x.get("momentum3", 0), reverse=True)
    lines: List[str] = []
    date_str = datetime.now().strftime("%Y-%m-%d")
    lines.append(f"# Daily Taiwan Stock Screening Report ({date_str})\n")
    lines.append(
        "The following table lists stocks trading between the specified price range "
        "that have exhibited a short-term upward trend (5-day MA above 20-day MA and price above MA5) "
        "along with momentum and basic fundamental metrics.\n"
    )
    header = [
        "Stock ID",
        "Close",
        "MA5",
        "MA20",
        "Momentum (3d)",
        "RSI14",
        "PER",
        "PBR",
        "Dividend Yield",
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for c in candidates_sorted:
        row = [
            str(c.get("stock_id")),
            f"{c.get('latest_close'):.2f}",
            f"{c.get('ma5'):.2f}",
            f"{c.get('ma20'):.2f}",
            f"{c.get('momentum3'):.4f}",
            f"{c.get('rsi14'):.2f}" if pd.notna(c.get("rsi14")) else "",
            f"{c.get('PER'):.2f}" if pd.notna(c.get("PER")) else "",
            f"{c.get('PBR'):.2f}" if pd.notna(c.get("PBR")) else "",
            f"{c.get('dividend_yield'):.2f}%" if pd.notna(c.get("dividend_yield")) else "",
        ]
        lines.append("| " + " | ".join(row) + " |")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    setup_logging()
    token = os.getenv("FINMIND_TOKEN", "").strip()
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=60)
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    logging.info(f"Fetching price data from {start_date_str} to {end_date_str}")
    price_df = fetch_dataset(
        "TaiwanStockPrice",
        token,
        params={"start_date": start_date_str, "end_date": end_date_str},
    )
    if price_df.empty:
        logging.error("No price data retrieved; aborting.")
        return
    price_df["date"] = pd.to_datetime(price_df["date"])
    price_df["close"] = pd.to_numeric(price_df["close"], errors="coerce")
    candidates: List[Dict[str, object]] = []
    unique_stocks = price_df["stock_id"].unique()
    logging.info(f"Analysing {len(unique_stocks)} stocks...")
    for stock_id in unique_stocks:
        stock_prices = price_df[price_df["stock_id"] == stock_id].copy()
        metrics = analyse_stock(stock_prices)
        if metrics is None:
            continue
        per_df = fetch_dataset(
            "TaiwanStockPER",
            token,
            params={"data_id": stock_id, "start_date": start_date_str, "end_date": end_date_str},
        )
        metrics_with_per = analyse_stock(stock_prices, per_history=per_df)
        if metrics_with_per is not None:
            candidates.append(metrics_with_per)
        else:
            candidates.append(metrics)
    report_date_str = end_date.strftime("%Y-%m-%d")
    report_path = os.path.join(os.path.dirname(__file__), f"daily_report_{report_date_str}.md")
    generate_report(candidates, report_path)


if __name__ == "__main__":
    main()
