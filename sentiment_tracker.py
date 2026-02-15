import csv
import datetime as dt
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd
import requests
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "sentiment_data.csv"
HEALTH_FILE = BASE_DIR / "feed_health.csv"

STOCK_TICKERS = ("SPY", "QQQ")
STOCK_WEIGHTS = {"SPY": 0.6, "QQQ": 0.4}
STOCK_PERIOD = "9mo"
STOCK_MA_WINDOW = 50
RSI_WINDOW = 14
CRYPTO_URL = "https://api.alternative.me/fng/?limit=1&format=json"
VIX_SYMBOL = "^VIX"
VIX_MIN_DEFAULT = 10.0
VIX_MAX_DEFAULT = 80.0
REQUEST_TIMEOUT = 10
REQUEST_RETRIES = 3
RETRY_BACKOFF_SECONDS = 2.0

FIELDNAMES = [
    "timestamp",
    "stock_fear_greed",
    "crypto_fear_greed",
    "vix",
    "vix_normalized",
    "mood_score",
    "mood_label",
]

HEALTH_FIELDNAMES = [
    "timestamp",
    "cnn_status",
    "crypto_status",
    "vix_status",
    "cnn_latency_ms",
    "crypto_latency_ms",
    "vix_latency_ms",
    "fallback_used",
    "error_notes",
]


def clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, value))


def normalize(value: float, min_value: float, max_value: float) -> Optional[float]:
    if max_value <= min_value:
        return None
    return clamp((value - min_value) / (max_value - min_value) * 100.0)


def request_json(url: str, headers: Optional[Dict[str, str]] = None) -> Dict:
    last_error: Optional[Exception] = None
    for attempt in range(1, REQUEST_RETRIES + 1):
        try:
            response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except Exception as exc:
            last_error = exc
            if attempt < REQUEST_RETRIES:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
    if last_error is None:
        raise RuntimeError("Failed to fetch JSON without an exception")
    raise last_error


def fetch_stock_history(period: str = STOCK_PERIOD):
    try:
        return yf.download(
            tickers=list(STOCK_TICKERS),
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        return None


def extract_stock_prices(data) -> pd.DataFrame:
    if data is None or data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        levels = set(data.columns.get_level_values(0))
        if "Adj Close" in levels:
            prices = data["Adj Close"].copy()
        elif "Close" in levels:
            prices = data["Close"].copy()
        else:
            return pd.DataFrame()
    else:
        close_col = "Adj Close" if "Adj Close" in data.columns else "Close" if "Close" in data.columns else None
        if close_col is None:
            return pd.DataFrame()
        prices = data[[close_col]].copy()
        prices.columns = [STOCK_TICKERS[0]]

    if isinstance(prices, pd.Series):
        prices = prices.to_frame(name=STOCK_TICKERS[0])

    return prices.sort_index().ffill()


def compute_rsi(series: pd.Series, window: int = RSI_WINDOW) -> Optional[float]:
    clean = series.dropna()
    if len(clean) < window + 1:
        return None

    delta = clean.diff()
    gain = delta.clip(lower=0).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.dropna()
    if rsi.empty:
        return None
    return float(rsi.iloc[-1])


def score_stock_series(series: pd.Series) -> Optional[float]:
    clean = series.dropna()
    if len(clean) < max(STOCK_MA_WINDOW, RSI_WINDOW) + 1:
        return None

    latest = float(clean.iloc[-1])
    ma = clean.rolling(STOCK_MA_WINDOW).mean().iloc[-1]
    if ma is None or pd.isna(ma) or ma == 0:
        return None

    deviation = (latest - float(ma)) / float(ma)
    deviation = max(-0.12, min(0.12, deviation))
    score = 50.0 + deviation * 300.0

    rsi = compute_rsi(clean, RSI_WINDOW)
    if rsi is not None:
        if rsi > 70.0:
            score -= min(12.0, (rsi - 70.0) * 0.7)
        elif rsi < 30.0:
            score += min(12.0, (30.0 - rsi) * 0.7)

    return clamp(score, 0.0, 100.0)


def fetch_stock_sentiment() -> float:
    data = fetch_stock_history()
    prices = extract_stock_prices(data)
    if prices.empty:
        raise ValueError("Stock price data unavailable")

    weighted_scores: list[tuple[float, float]] = []
    for ticker, weight in STOCK_WEIGHTS.items():
        if ticker not in prices.columns:
            continue
        score = score_stock_series(prices[ticker])
        if score is None:
            continue
        weighted_scores.append((score, float(weight)))

    if not weighted_scores:
        raise ValueError("Failed to compute stock sentiment score")

    total_weight = sum(weight for _, weight in weighted_scores)
    if total_weight <= 0:
        raise ValueError("Invalid stock weights")
    return sum(score * weight for score, weight in weighted_scores) / total_weight


def fetch_crypto_fear_greed() -> float:
    payload = request_json(CRYPTO_URL)
    data = payload.get("data") or []
    score = data[0].get("value") if data else None
    if score is None:
        score = payload.get("value")
    if score is None:
        raise ValueError("Crypto response missing value")
    return float(score)


def fetch_vix_history(period: str):
    try:
        return yf.download(
            tickers=VIX_SYMBOL,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        return None


def extract_close_series(data) -> Optional[object]:
    if data is None or data.empty:
        return None
    if "Close" in data.columns:
        series = data["Close"]
        if getattr(series, "ndim", 1) > 1:
            series = series.iloc[:, 0]
    else:
        series = data.iloc[:, 0]
        if getattr(series, "ndim", 1) > 1:
            series = series.iloc[:, 0]
    series = series.dropna()
    if series.empty:
        return None
    return series


def fetch_vix() -> Tuple[Optional[float], Optional[float]]:
    hist_recent = fetch_vix_history("5d")
    close_recent = extract_close_series(hist_recent)
    if close_recent is None:
        return None, None

    last_close = float(close_recent.iloc[-1])

    hist_year = fetch_vix_history("1y")
    min_val = None
    max_val = None
    close_year = extract_close_series(hist_year)
    if close_year is not None:
        min_val = float(close_year.min())
        max_val = float(close_year.max())

    if min_val is None or max_val is None or max_val <= min_val:
        min_val = VIX_MIN_DEFAULT
        max_val = VIX_MAX_DEFAULT

    normalized = normalize(last_close, min_val, max_val)
    return last_close, normalized


def mood_label(score: float) -> str:
    if score <= 20:
        return "STORMY"
    if score <= 80:
        return "CLOUDY"
    return "SHINY"


def load_latest_row(path: Path) -> Dict[str, Optional[float]]:
    if not path.exists():
        return {}

    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        if not rows:
            return {}
        return rows[-1]


def ensure_data_file(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()


def append_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def ensure_health_file(path: Path) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEALTH_FIELDNAMES)
        writer.writeheader()


def append_health_row(path: Path, row: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEALTH_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def coerce_float(value: Optional[object]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def calculate_weighted_mood_score(stock: float, crypto: float, vix_normalized: float) -> float:
    """
    Compute the Cloudy&Shiny Index composite score.

    Formula:
        (stock * 0.4) + (crypto * 0.3) + ((100 - vix_normalized) * 0.3)
    """
    if stock is None or crypto is None or vix_normalized is None:
        raise ValueError("All inputs (stock, crypto, vix_normalized) must be provided")

    stock = clamp(stock, 0.0, 100.0)
    crypto = clamp(crypto, 0.0, 100.0)
    vix_normalized = clamp(vix_normalized, 0.0, 100.0)

    score = (stock * 0.4) + (crypto * 0.3) + ((100.0 - vix_normalized) * 0.3)
    return clamp(score, 0.0, 100.0)


def main() -> int:
    ensure_data_file(DATA_FILE)
    ensure_health_file(HEALTH_FILE)
    latest = load_latest_row(DATA_FILE)

    stock = None
    crypto = None
    vix_value = None
    vix_norm = None
    source_status = {"stock": "missing", "crypto": "missing", "vix": "missing"}
    source_latency_ms: Dict[str, Optional[float]] = {"stock": None, "crypto": None, "vix": None}
    errors: list[str] = []

    start = time.perf_counter()
    try:
        stock = fetch_stock_sentiment()
        source_status["stock"] = "live"
    except Exception as exc:
        print(f"Warning: failed to fetch Stock Sentiment ({exc})")
        errors.append(f"stock:{exc}")
        source_status["stock"] = "failed"
    source_latency_ms["stock"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    try:
        crypto = fetch_crypto_fear_greed()
        source_status["crypto"] = "live"
    except Exception as exc:
        print(f"Warning: failed to fetch Crypto Fear & Greed ({exc})")
        errors.append(f"crypto:{exc}")
        source_status["crypto"] = "failed"
    source_latency_ms["crypto"] = (time.perf_counter() - start) * 1000.0

    start = time.perf_counter()
    try:
        vix_value, vix_norm = fetch_vix()
        source_status["vix"] = "live"
    except Exception as exc:
        print(f"Warning: failed to fetch VIX ({exc})")
        errors.append(f"vix:{exc}")
        source_status["vix"] = "failed"
    source_latency_ms["vix"] = (time.perf_counter() - start) * 1000.0

    fallback_used = False
    if stock is None:
        stock = coerce_float(latest.get("stock_fear_greed"))
        if stock is not None:
            source_status["stock"] = "fallback"
            fallback_used = True
        else:
            source_status["stock"] = "missing"
    if crypto is None:
        crypto = coerce_float(latest.get("crypto_fear_greed"))
        if crypto is not None:
            source_status["crypto"] = "fallback"
            fallback_used = True
        else:
            source_status["crypto"] = "missing"
    if vix_value is None:
        vix_value = coerce_float(latest.get("vix"))
    if vix_norm is None:
        vix_norm = coerce_float(latest.get("vix_normalized"))
    if source_status["vix"] != "live":
        if vix_norm is not None:
            source_status["vix"] = "fallback"
            fallback_used = True
        else:
            source_status["vix"] = "missing"

    health_timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    health_row = {
        "timestamp": health_timestamp,
        "cnn_status": source_status["stock"],
        "crypto_status": source_status["crypto"],
        "vix_status": source_status["vix"],
        "cnn_latency_ms": round(source_latency_ms["stock"], 1)
        if source_latency_ms["stock"] is not None
        else "",
        "crypto_latency_ms": round(source_latency_ms["crypto"], 1)
        if source_latency_ms["crypto"] is not None
        else "",
        "vix_latency_ms": round(source_latency_ms["vix"], 1)
        if source_latency_ms["vix"] is not None
        else "",
        "fallback_used": "yes" if fallback_used else "no",
        "error_notes": "; ".join(errors)[:500],
    }
    append_health_row(HEALTH_FILE, health_row)

    if stock is None or crypto is None or vix_norm is None:
        print("Error: missing inputs; run again when data sources are available.")
        return 1

    mood_score = calculate_weighted_mood_score(stock, crypto, vix_norm)
    label = mood_label(mood_score)

    timestamp = health_timestamp
    row = {
        "timestamp": timestamp,
        "stock_fear_greed": round(stock, 2),
        "crypto_fear_greed": round(crypto, 2),
        "vix": round(vix_value, 2) if vix_value is not None else "",
        "vix_normalized": round(vix_norm, 2),
        "mood_score": round(mood_score, 2),
        "mood_label": label,
    }

    append_row(DATA_FILE, row)
    print(
        "Saved:", timestamp, "| Stock:", row["stock_fear_greed"], "| Crypto:",
        row["crypto_fear_greed"], "| VIX:", row["vix"], "| Mood:",
        row["mood_score"], label
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
