import math
from pathlib import Path

import pandas as pd
import yfinance as yf

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "sentiment_data.csv"
TEMPLATE_FILE = BASE_DIR / "template.html"
OUTPUT_FILE = BASE_DIR / "index.html"
HEALTH_FILE = BASE_DIR / "feed_health.csv"

STATUS_COLORS = {
    "STORMY": "var(--accent-red)",
    "CLOUDY": "var(--accent-yellow)",
    "SHINY": "var(--accent-green)",
}

STATUS_BADGES = {
    "STORMY": "fear",
    "CLOUDY": "neutral",
    "SHINY": "greed",
}

STATUS_TAGLINES = {
    "STORMY": "Extreme Fear",
    "CLOUDY": "Neutral",
    "SHINY": "Extreme Greed",
}

ASSETS = [
    {"ticker": "SPY", "name": "S&P 500", "is_fear": False},
    {"ticker": "QQQ", "name": "Nasdaq 100", "is_fear": False},
    {"ticker": "^GDAXI", "name": "DAX", "is_fear": False},
    {"ticker": "^FCHI", "name": "CAC 40", "is_fear": False},
    {"ticker": "^N225", "name": "Nikkei 225", "is_fear": False},
    {"ticker": "000001.SS", "name": "Shanghai Comp", "is_fear": False},
    {"ticker": "^HSI", "name": "Hang Seng", "is_fear": False},
    {"ticker": "XU100.IS", "name": "BIST 100", "is_fear": False},
    {"ticker": "^VIX", "name": "VIX Fear Index", "is_fear": True},
    {"ticker": "TLT", "name": "US 20Y Treasury", "is_fear": True},
    {"ticker": "GLD", "name": "Gold", "is_fear": True},
    {"ticker": "DX-Y.NYB", "name": "US Dollar Index", "is_fear": True},
]

GDP_BY_COUNTRY = {
    "US": 26.9,
    "Germany": 4.4,
    "France": 3.0,
    "Japan": 4.2,
    "China": 17.5,
    "HongKong": 0.36,
    "Turkey": 1.15,
}

TICKER_COUNTRY = {
    "SPY": "US",
    "QQQ": "US",
    "^GDAXI": "Germany",
    "^FCHI": "France",
    "^N225": "Japan",
    "000001.SS": "China",
    "^HSI": "HongKong",
    "XU100.IS": "Turkey",
}

RISK_WEIGHT_SHARE = 0.7
FEAR_WEIGHT_SHARE = 0.3

US_TICKERS = ["SPY", "QQQ"]
ASIA_TICKERS = ["^N225", "000001.SS", "^HSI"]
EU_TICKERS = ["^GDAXI", "^FCHI", "XU100.IS"]

NUMERIC_COLUMNS = [
    "stock_fear_greed",
    "crypto_fear_greed",
    "vix",
    "vix_normalized",
    "mood_score",
]


def score_label(score: float) -> str:
    if score <= 20:
        return "STORMY"
    if score <= 80:
        return "CLOUDY"
    return "SHINY"


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()

    data = pd.read_csv(path)
    if data.empty:
        return data

    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    for col in NUMERIC_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.dropna(subset=["timestamp"]).sort_values("timestamp")


def load_health_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path)
    if data.empty:
        return data
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    for col in ["cnn_latency_ms", "crypto_latency_ms", "vix_latency_ms"]:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in ["cnn_status", "crypto_status", "vix_status", "fallback_used"]:
        if col in data.columns:
            data[col] = data[col].astype(str).str.strip().str.lower()
    return data.dropna(subset=["timestamp"]).sort_values("timestamp")


def prepare_recent(data: pd.DataFrame, days: int = 7) -> pd.DataFrame:
    if data.empty:
        return data
    indexed = data.set_index("timestamp").sort_index()
    daily = indexed.resample("1D").last().dropna(how="all")
    if daily.empty:
        return data.tail(days)
    return daily.tail(days).reset_index()


def prepare_recent_hours(data: pd.DataFrame, hours: int = 48) -> pd.DataFrame:
    if data.empty:
        return data
    end_time = data["timestamp"].max()
    start_time = end_time - pd.Timedelta(hours=hours)
    recent = data[(data["timestamp"] >= start_time) & (data["timestamp"] <= end_time)].copy()
    if recent.empty:
        return recent

    hourly = recent.set_index("timestamp").sort_index().resample("1h").mean(numeric_only=True)
    for col in ["mood_score", "stock_fear_greed", "crypto_fear_greed", "vix_normalized"]:
        if col in hourly.columns:
            hourly[col] = hourly[col].interpolate(method="time").ffill().bfill()
    return hourly.reset_index()


def format_value(value: float | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{value:.{decimals}f}"


def format_signed(value: float | None, decimals: int = 1) -> str:
    if value is None or pd.isna(value):
        return "--"
    return f"{value:+.{decimals}f}"


def normal_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def forecast_regime_probabilities(mean: float, sigma: float) -> tuple[float, float, float]:
    sigma = max(0.5, float(sigma))
    p_stormy = normal_cdf((20.0 - mean) / sigma)
    p_shiny = 1.0 - normal_cdf((80.0 - mean) / sigma)
    p_cloudy = max(0.0, 1.0 - p_stormy - p_shiny)

    total = p_stormy + p_cloudy + p_shiny
    if total <= 0:
        return 0.0, 100.0, 0.0
    return (
        (p_stormy / total) * 100.0,
        (p_cloudy / total) * 100.0,
        (p_shiny / total) * 100.0,
    )


def points_from_series(
    values: list[float],
    width: float,
    height: float,
    min_value: float = 0.0,
    max_value: float = 100.0,
) -> str:
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not clean:
        return f"0,{height / 2:.2f} {width:.2f},{height / 2:.2f}"

    if max_value <= min_value:
        max_value = min_value + 1.0

    if len(clean) == 1:
        y = height - ((clean[0] - min_value) / (max_value - min_value) * height)
        y = max(0.0, min(height, y))
        return f"0,{y:.2f} {width:.2f},{y:.2f}"

    step = width / (len(clean) - 1)
    points: list[str] = []
    for idx, value in enumerate(clean):
        x = idx * step
        y = height - ((value - min_value) / (max_value - min_value) * height)
        y = max(0.0, min(height, y))
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def points_from_segment(
    values: list[float],
    start_index: int,
    total_points: int,
    width: float,
    height: float,
    min_value: float = 0.0,
    max_value: float = 100.0,
) -> str:
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not clean or total_points < 2:
        return f"0,{height / 2:.2f} {width:.2f},{height / 2:.2f}"

    if max_value <= min_value:
        max_value = min_value + 1.0

    step = width / (total_points - 1)
    points: list[str] = []
    for idx, value in enumerate(clean):
        x = (start_index + idx) * step
        y = height - ((value - min_value) / (max_value - min_value) * height)
        y = max(0.0, min(height, y))
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def build_quant_forecast_series(
    values: list[float], steps: int = 24, lookback: int = 96
) -> tuple[list[float], list[float], list[float], float, float, float]:
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    if not clean:
        return [], [], [], 0.0, 0.0, 0.0

    if len(clean) < 3:
        baseline = clean[-1]
        projection = [baseline for _ in range(steps)]
        return projection, projection, projection, 0.0, 0.0, 0.0

    window = clean[-lookback:]
    n = len(window)
    last_value = window[-1]

    # Model 1: AR(1)-style autoregressive forecast.
    x_prev = window[:-1]
    x_next = window[1:]
    mean_prev = sum(x_prev) / len(x_prev)
    mean_next = sum(x_next) / len(x_next)
    var_prev = sum((p - mean_prev) ** 2 for p in x_prev)
    covar = sum((p - mean_prev) * (nxt - mean_next) for p, nxt in zip(x_prev, x_next))
    phi = 0.0 if var_prev == 0 else covar / var_prev
    phi = max(-0.995, min(0.995, phi))
    intercept = mean_next - phi * mean_prev

    ar_errors: list[float] = []
    for p, nxt in zip(x_prev, x_next):
        ar_errors.append(nxt - (intercept + phi * p))
    sigma = float(pd.Series(ar_errors).std()) if len(ar_errors) >= 2 else 1.0
    sigma = max(1.0, sigma if not pd.isna(sigma) else 1.0)

    ar_forecast: list[float] = []
    ar_state = last_value
    for _ in range(steps):
        ar_state = intercept + phi * ar_state
        ar_forecast.append(max(0.0, min(100.0, ar_state)))

    # Model 2: Damped local trend regression.
    trend_window = window[-min(48, n):]
    m = len(trend_window)
    sum_x = (m - 1) * m / 2
    sum_x2 = (m - 1) * m * (2 * m - 1) / 6
    sum_y = sum(trend_window)
    sum_xy = sum(i * value for i, value in enumerate(trend_window))
    denom = m * sum_x2 - sum_x * sum_x
    raw_slope = 0.0 if denom == 0 else (m * sum_xy - sum_x * sum_y) / denom
    raw_slope = max(-3.0, min(3.0, raw_slope))

    damping = 0.92
    trend_forecast: list[float] = []
    for i in range(1, steps + 1):
        cumulative = raw_slope * ((1.0 - damping**i) / (1.0 - damping))
        trend_forecast.append(max(0.0, min(100.0, last_value + cumulative)))

    # Model 3: EWMA mean-reversion with fading momentum.
    alpha = 0.18
    ewma = window[0]
    for value in window[1:]:
        ewma = alpha * value + (1.0 - alpha) * ewma
    recent_span = min(6, n - 1)
    momentum = (window[-1] - window[-1 - recent_span]) / max(1, recent_span)
    reversion_speed = 0.12

    mr_forecast: list[float] = []
    for i in range(1, steps + 1):
        mean_revert = ewma + (last_value - ewma) * math.exp(-reversion_speed * i)
        momentum_tail = momentum * math.exp(-0.22 * (i - 1))
        mr_forecast.append(max(0.0, min(100.0, mean_revert + momentum_tail)))

    # Ensemble: weighted blend for base scenario.
    w_ar, w_trend, w_mr = 0.50, 0.30, 0.20
    base_forecast: list[float] = []
    bull_forecast: list[float] = []
    bear_forecast: list[float] = []

    for i in range(steps):
        base = (
            ar_forecast[i] * w_ar
            + trend_forecast[i] * w_trend
            + mr_forecast[i] * w_mr
        )
        horizon = i + 1
        spread = sigma * (0.8 + (horizon / steps) * 0.9) + abs(raw_slope) * horizon * 0.15
        base_forecast.append(max(0.0, min(100.0, base)))
        bull_forecast.append(max(0.0, min(100.0, base + spread)))
        bear_forecast.append(max(0.0, min(100.0, base - spread)))

    return base_forecast, bull_forecast, bear_forecast, raw_slope, sigma, phi


def ensure_series(values: list[float], fallback: float | None) -> list[float]:
    clean = [float(v) for v in values if v is not None and not pd.isna(v)]
    if clean:
        return clean
    if fallback is not None and not pd.isna(fallback):
        return [float(fallback)]
    return [50.0]


def blend_three(
    a: list[float], b: list[float], c: list[float], wa: float, wb: float, wc: float
) -> list[float]:
    n = min(len(a), len(b), len(c))
    return [
        max(0.0, min(100.0, a[i] * wa + b[i] * wb + c[i] * wc))
        for i in range(n)
    ]


def build_component_composite_forecast(
    stock_values: list[float],
    crypto_values: list[float],
    vix_inverse_values: list[float],
    stock_now: float | None,
    crypto_now: float | None,
    vix_inverse_now: float | None,
    steps: int = 24,
    lookback: int = 96,
) -> dict:
    stock_now = 50.0 if stock_now is None or pd.isna(stock_now) else float(stock_now)
    crypto_now = 50.0 if crypto_now is None or pd.isna(crypto_now) else float(crypto_now)
    vix_inverse_now = (
        50.0 if vix_inverse_now is None or pd.isna(vix_inverse_now) else float(vix_inverse_now)
    )

    stock_series = ensure_series(stock_values, stock_now)
    crypto_series = ensure_series(crypto_values, crypto_now)
    vix_series = ensure_series(vix_inverse_values, vix_inverse_now)

    s_base, s_bull, s_bear, s_slope, s_sigma, s_phi = build_quant_forecast_series(
        stock_series, steps=steps, lookback=lookback
    )
    c_base, c_bull, c_bear, c_slope, c_sigma, c_phi = build_quant_forecast_series(
        crypto_series, steps=steps, lookback=lookback
    )
    v_base, v_bull, v_bear, v_slope, v_sigma, v_phi = build_quant_forecast_series(
        vix_series, steps=steps, lookback=lookback
    )

    weights = (0.4, 0.3, 0.3)
    base = blend_three(s_base, c_base, v_base, *weights)
    bull = blend_three(s_bull, c_bull, v_bull, *weights)
    bear = blend_three(s_bear, c_bear, v_bear, *weights)

    stock_contribution = 0.4 * (s_base[-1] - stock_now)
    crypto_contribution = 0.3 * (c_base[-1] - crypto_now)
    vix_contribution = 0.3 * (v_base[-1] - vix_inverse_now)
    contributions = {
        "Stock": stock_contribution,
        "Crypto": crypto_contribution,
        "VIX-Inv": vix_contribution,
    }
    dominant_driver = max(contributions, key=lambda key: abs(contributions[key]))

    slope = 0.4 * s_slope + 0.3 * c_slope + 0.3 * v_slope
    sigma = 0.4 * s_sigma + 0.3 * c_sigma + 0.3 * v_sigma
    phi = 0.4 * s_phi + 0.3 * c_phi + 0.3 * v_phi

    return {
        "base": base,
        "bull": bull,
        "bear": bear,
        "slope": slope,
        "sigma": sigma,
        "phi": phi,
        "contrib_stock": stock_contribution,
        "contrib_crypto": crypto_contribution,
        "contrib_vix": vix_contribution,
        "dominant_driver": dominant_driver,
    }


def compute_forecast_backtest(
    data: pd.DataFrame,
    horizons: tuple[int, int, int] = (1, 6, 24),
    lookback: int = 96,
    max_windows: int = 220,
) -> dict:
    required = ["mood_score", "stock_fear_greed", "crypto_fear_greed", "vix_normalized"]
    if data.empty or any(col not in data.columns for col in required):
        return {
            "windows": 0,
            "mae_1h": None,
            "mae_6h": None,
            "mae_24h": None,
            "hit_24h": None,
            "rmse_24h": None,
            "coverage_24h": None,
            "bias_24h": None,
            "drift_24h": None,
            "error_series_24h": [],
        }

    ordered = data.dropna(subset=required).sort_values("timestamp").reset_index(drop=True)
    if ordered.empty:
        return {
            "windows": 0,
            "mae_1h": None,
            "mae_6h": None,
            "mae_24h": None,
            "hit_24h": None,
            "rmse_24h": None,
            "coverage_24h": None,
            "bias_24h": None,
            "drift_24h": None,
            "error_series_24h": [],
        }

    max_h = max(horizons)
    min_required = lookback + max_h + 2
    if len(ordered) < min_required:
        return {
            "windows": 0,
            "mae_1h": None,
            "mae_6h": None,
            "mae_24h": None,
            "hit_24h": None,
            "rmse_24h": None,
            "coverage_24h": None,
            "bias_24h": None,
            "drift_24h": None,
            "error_series_24h": [],
        }

    errors: dict[int, list[float]] = {h: [] for h in horizons}
    sq_errors_24: list[float] = []
    hit_24: list[int] = []
    coverage_24: list[int] = []
    signed_24: list[float] = []
    abs_24: list[float] = []

    end_idx = len(ordered) - max_h - 1
    start_idx = max(lookback, end_idx - max_windows + 1)

    for t in range(start_idx, end_idx + 1):
        hist = ordered.iloc[: t + 1]
        now = hist.iloc[-1]
        forecast = build_component_composite_forecast(
            stock_values=hist["stock_fear_greed"].tolist(),
            crypto_values=hist["crypto_fear_greed"].tolist(),
            vix_inverse_values=(100.0 - hist["vix_normalized"]).clip(0.0, 100.0).tolist(),
            stock_now=float(now["stock_fear_greed"]),
            crypto_now=float(now["crypto_fear_greed"]),
            vix_inverse_now=float(100.0 - now["vix_normalized"]),
            steps=max_h,
            lookback=lookback,
        )
        preds = forecast["base"]
        bull_preds = forecast["bull"]
        bear_preds = forecast["bear"]
        current = float(now["mood_score"])

        for h in horizons:
            pred_h = float(preds[h - 1])
            actual_h = float(ordered.iloc[t + h]["mood_score"])
            err = abs(pred_h - actual_h)
            errors[h].append(err)

            if h == 24:
                signed_err = pred_h - actual_h
                sq_errors_24.append(signed_err**2)
                signed_24.append(signed_err)
                abs_24.append(err)
                upper_h = float(bull_preds[h - 1])
                lower_h = float(bear_preds[h - 1])
                coverage_24.append(1 if lower_h <= actual_h <= upper_h else 0)
                pred_dir = pred_h - current
                actual_dir = actual_h - current
                if abs(actual_dir) < 0.25 and abs(pred_dir) < 0.25:
                    hit_24.append(1)
                elif pred_dir == 0:
                    hit_24.append(0)
                else:
                    hit_24.append(1 if (pred_dir > 0) == (actual_dir > 0) else 0)

    def mean_or_none(vals: list[float]) -> float | None:
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    mae_1h = mean_or_none(errors.get(1, []))
    mae_6h = mean_or_none(errors.get(6, []))
    mae_24h = mean_or_none(errors.get(24, []))
    rmse_24h = math.sqrt(mean_or_none(sq_errors_24)) if sq_errors_24 else None
    hit_24h = (sum(hit_24) / len(hit_24) * 100.0) if hit_24 else None
    coverage_24h = (sum(coverage_24) / len(coverage_24) * 100.0) if coverage_24 else None
    bias_24h = mean_or_none(signed_24)

    drift_24h = None
    if len(abs_24) >= 40:
        recent = abs_24[-20:]
        prior = abs_24[-40:-20]
        drift_24h = mean_or_none(recent) - mean_or_none(prior)
    elif len(abs_24) >= 20:
        split = len(abs_24) // 2
        recent = abs_24[split:]
        prior = abs_24[:split]
        drift_24h = mean_or_none(recent) - mean_or_none(prior)

    windows = len(errors[24]) if 24 in errors else 0
    return {
        "windows": windows,
        "mae_1h": mae_1h,
        "mae_6h": mae_6h,
        "mae_24h": mae_24h,
        "hit_24h": hit_24h,
        "rmse_24h": rmse_24h,
        "coverage_24h": coverage_24h,
        "bias_24h": bias_24h,
        "drift_24h": drift_24h,
        "error_series_24h": abs_24[-120:],
    }


def class_from_delta(delta: float | None, epsilon: float = 0.01) -> str:
    if delta is None or pd.isna(delta):
        return "neutral"
    if delta > epsilon:
        return "up"
    if delta < -epsilon:
        return "down"
    return "neutral"


def class_from_score(value: float | None, low: float, high: float) -> str:
    if value is None or pd.isna(value):
        return "neutral"
    if value >= high:
        return "up"
    if value <= low:
        return "down"
    return "neutral"


def compute_asset_weights() -> dict[str, float]:
    risk_assets = [a for a in ASSETS if not a["is_fear"]]
    fear_assets = [a for a in ASSETS if a["is_fear"]]

    country_groups: dict[str, list[str]] = {}
    for asset in risk_assets:
        country = TICKER_COUNTRY.get(asset["ticker"])
        if country is None:
            continue
        country_groups.setdefault(country, []).append(asset["ticker"])

    risk_weights_raw: dict[str, float] = {}
    for country, tickers in country_groups.items():
        gdp = GDP_BY_COUNTRY.get(country, 1.0)
        split = gdp / len(tickers)
        for ticker in tickers:
            risk_weights_raw[ticker] = split

    risk_total = sum(risk_weights_raw.values())
    risk_weights = {
        ticker: (weight / risk_total) * RISK_WEIGHT_SHARE if risk_total else 0.0
        for ticker, weight in risk_weights_raw.items()
    }

    fear_weight_each = FEAR_WEIGHT_SHARE / len(fear_assets) if fear_assets else 0.0
    fear_weights = {asset["ticker"]: fear_weight_each for asset in fear_assets}

    weights = {**fear_weights, **risk_weights}
    if not weights:
        equal = 1 / len(ASSETS) if ASSETS else 0.0
        return {asset["ticker"]: equal for asset in ASSETS}
    return weights


def compute_region_weights(weights: dict[str, float]) -> tuple[float, float, float]:
    us_weight = sum(weights.get(ticker, 0.0) for ticker in US_TICKERS) * 100
    asia_weight = sum(weights.get(ticker, 0.0) for ticker in ASIA_TICKERS) * 100
    eu_weight = sum(weights.get(ticker, 0.0) for ticker in EU_TICKERS) * 100
    return us_weight, asia_weight, eu_weight


def fetch_single_price(ticker: str, period: str) -> pd.Series | None:
    try:
        data = yf.download(
            tickers=ticker,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        return None
    if data.empty:
        return None

    if isinstance(data.columns, pd.MultiIndex):
        series = data["Adj Close"]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
    elif "Adj Close" in data.columns:
        series = data["Adj Close"]
    else:
        series = data.squeeze()
    return series.ffill()


def fetch_asset_prices(tickers: tuple[str, ...], period: str = "260d") -> pd.DataFrame:
    try:
        data = yf.download(
            tickers=list(tickers),
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    if isinstance(data.columns, pd.MultiIndex):
        prices = data["Adj Close"].copy()
    elif "Adj Close" in data.columns:
        prices = data["Adj Close"].copy()
    else:
        prices = data.copy()

    prices = prices.reindex(columns=list(tickers))
    missing = [
        ticker
        for ticker in tickers
        if ticker not in prices.columns or prices[ticker].dropna().empty
    ]
    for ticker in missing:
        series = fetch_single_price(ticker, period)
        if series is None or series.dropna().empty:
            continue
        combined_index = prices.index.union(series.index)
        prices = prices.reindex(combined_index)
        prices[ticker] = series.reindex(combined_index)

    return prices.sort_index().ffill()


def compute_rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    return 100 - (100 / (1 + rs))


def compute_asset_score(
    price: float | None,
    ma60: float | None,
    rsi: float | None,
    is_fear: bool,
    ticker: str = "",
) -> float | None:
    if price is None or ma60 is None or pd.isna(price) or pd.isna(ma60) or ma60 == 0:
        return None

    if ticker == "^VIX":
        vix_normalized = ((price - 10) / (80 - 10)) * 100
        vix_score = max(0.0, min(100.0, vix_normalized))
        return 100 - vix_score

    deviation = (price - ma60) / ma60
    deviation = max(-0.1, min(0.1, deviation))
    score = 50 + deviation * 500
    if is_fear:
        score = 100 - score

    if not is_fear and rsi is not None and not pd.isna(rsi):
        if score > 60 and rsi > 70:
            score -= 15
        elif score < 40 and rsi < 30:
            score += 10

    return max(0.0, min(100.0, score))


def trend_class(series: pd.Series) -> str:
    clean = series.dropna()
    if len(clean) < 2:
        return "flat"
    delta = float(clean.iloc[-1] - clean.iloc[0])
    if delta > 0.001:
        return "up"
    if delta < -0.001:
        return "down"
    return "flat"


def build_asset_rows(prices: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    if prices.empty:
        for asset in ASSETS:
            rows.append(
                {
                    "ticker": asset["ticker"],
                    "name": asset["name"],
                    "price": None,
                    "ma60": None,
                    "rsi": None,
                    "score": None,
                    "is_fear": asset["is_fear"],
                    "trend_points": "0,12 120,12",
                    "trend_class": "flat",
                }
            )
        return rows

    ma60 = prices.rolling(60).mean()
    rsi = compute_rsi(prices)
    latest_prices = prices.iloc[-1]
    latest_ma = ma60.iloc[-1]
    latest_rsi = rsi.iloc[-1]

    for asset in ASSETS:
        ticker = asset["ticker"]
        price = latest_prices.get(ticker)
        ma = latest_ma.get(ticker)
        rsi_value = latest_rsi.get(ticker)
        score = compute_asset_score(price, ma, rsi_value, asset["is_fear"], ticker)

        trend_series = prices.get(ticker, pd.Series(dtype=float)).dropna().tail(30)
        if trend_series.empty:
            spark_points = "0,12 120,12"
            spark_class = "flat"
        else:
            spark_points = points_from_series(
                trend_series.tolist(),
                width=120.0,
                height=24.0,
                min_value=float(trend_series.min()),
                max_value=float(trend_series.max()),
            )
            spark_class = trend_class(trend_series)

        rows.append(
            {
                "ticker": ticker,
                "name": asset["name"],
                "price": price,
                "ma60": ma,
                "rsi": rsi_value,
                "score": score,
                "is_fear": asset["is_fear"],
                "trend_points": spark_points,
                "trend_class": spark_class,
            }
        )
    return rows


def score_class(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "neutral"
    if value >= 70:
        return "bullish"
    if value >= 40:
        return "neutral"
    return "bearish"


def build_asset_card(row: dict, weight: float | None) -> str:
    score = row.get("score")
    rsi_value = row.get("rsi")
    price = row.get("price")
    ma60 = row.get("ma60")

    score_value = format_value(score, 1)
    rsi_label = format_value(rsi_value, 0)
    weight_label = format_value(weight * 100, 1) if weight is not None else "--"

    ma_text = "NO DATA"
    ma_class = "below"
    if price is not None and ma60 is not None and not pd.isna(price) and not pd.isna(ma60):
        if price >= ma60:
            ma_text = "ABOVE MA"
            ma_class = "above"
        else:
            ma_text = "BELOW MA"
            ma_class = "below"

    score_style = score_class(score)
    spark_class = row.get("trend_class", "flat")
    spark_points = row.get("trend_points", "0,12 120,12")

    return f"""
<div class="asset-card">
  <div class="asset-header">
    <div>
      <div class="asset-name">{row.get('name', '')}</div>
      <div class="asset-ticker">{row.get('ticker', '')}</div>
    </div>
    <div class="asset-ma-badge {ma_class}">{ma_text}</div>
  </div>
  <div class="asset-score {score_style}">{score_value}</div>
  <div class="asset-spark">
    <svg class="spark-svg" viewBox="0 0 120 24" preserveAspectRatio="none">
      <polyline class="spark-line {spark_class}" points="{spark_points}" />
    </svg>
  </div>
  <div class="asset-meta">
    <span>RSI: {rsi_label}</span>
    <span>Wt: {weight_label}%</span>
  </div>
</div>
"""


def build_history_item(timestamp: pd.Timestamp, score: float) -> str:
    label = score_label(score)
    badge_class = STATUS_BADGES[label]
    timestamp_str = timestamp.strftime("%b %d %H:%M")
    return f"""
<div class="history-item">
  <div class="history-header">
    <div class="history-time">{timestamp_str}</div>
    <div class="history-badge {badge_class}">{label}</div>
  </div>
  <div class="history-score">{format_value(score, 1)}</div>
  <div class="history-label">Composite mood</div>
</div>
"""


def build_signal_row(name: str, value: str, detail: str, state: str) -> str:
    return f"""
<div class="signal-row">
  <div class="signal-left">
    <div class="signal-name">{name}</div>
    <div class="signal-detail">{detail}</div>
  </div>
  <div class="signal-value {state}">{value}</div>
</div>
"""


def build_driver_waterfall_rows(
    stock_contrib: float, crypto_contrib: float, vix_contrib: float
) -> tuple[str, float]:
    rows_data = [
        ("Stock", stock_contrib),
        ("Crypto", crypto_contrib),
        ("VIX-Inv", vix_contrib),
    ]
    max_abs = max(max(abs(value) for _, value in rows_data), 0.5)
    rows: list[str] = []
    for label, value in rows_data:
        cls = class_from_delta(value, epsilon=0.05)
        width = max(4.0, (abs(value) / max_abs) * 100.0)
        rows.append(
            f"""
<div class="wf-row">
  <div class="wf-label">{label}</div>
  <div class="wf-track">
    <div class="wf-bar {cls}" style="width:{width:.1f}%"></div>
  </div>
  <div class="wf-value {cls}">{format_signed(value, 2)}</div>
</div>
"""
        )
    total = stock_contrib + crypto_contrib + vix_contrib
    return "\n".join(rows), total


def health_status_class(status: str) -> str:
    normalized = (status or "").strip().lower()
    if normalized == "live":
        return "up"
    if normalized == "fallback":
        return "neutral"
    if normalized in {"failed", "missing"}:
        return "down"
    return "neutral"


def health_status_label(status: str) -> str:
    normalized = (status or "").strip().lower()
    if not normalized:
        return "UNKNOWN"
    return normalized.upper()


def compute_feed_health_metrics(
    sentiment_data: pd.DataFrame, health_data: pd.DataFrame
) -> dict[str, float | str | None]:
    now_utc = pd.Timestamp.now(tz="UTC")
    age_minutes: float | None = None
    if not sentiment_data.empty and "timestamp" in sentiment_data.columns:
        latest_ts = sentiment_data["timestamp"].max()
        if pd.notna(latest_ts):
            age_minutes = max(0.0, (now_utc - latest_ts).total_seconds() / 60.0)

    if age_minutes is None:
        age_class = "down"
    elif age_minutes <= 90:
        age_class = "up"
    elif age_minutes <= 240:
        age_class = "neutral"
    else:
        age_class = "down"

    metrics = {
        "age_minutes": age_minutes,
        "age_class": age_class,
        "overall_score": None,
        "overall_class": "neutral",
        "fallback_rate": None,
        "failure_rate": None,
        "cnn_status": "unknown",
        "crypto_status": "unknown",
        "vix_status": "unknown",
        "cnn_score": None,
        "crypto_score": None,
        "vix_score": None,
    }

    if health_data.empty:
        return metrics

    latest = health_data.iloc[-1]
    metrics["cnn_status"] = str(latest.get("cnn_status", "unknown"))
    metrics["crypto_status"] = str(latest.get("crypto_status", "unknown"))
    metrics["vix_status"] = str(latest.get("vix_status", "unknown"))

    sample = health_data.tail(96)
    sources = ["cnn", "crypto", "vix"]
    score_map = {"live": 1.0, "fallback": 0.6, "failed": 0.0, "missing": 0.0}

    source_scores: dict[str, float] = {}
    fallback_count = 0
    fail_count = 0
    total_obs = 0
    for source in sources:
        col = f"{source}_status"
        statuses = sample[col].astype(str).str.lower().tolist() if col in sample.columns else []
        if not statuses:
            source_scores[source] = 0.0
            continue
        values = [score_map.get(status, 0.0) for status in statuses]
        source_scores[source] = (sum(values) / len(values)) * 100.0
        fallback_count += sum(1 for status in statuses if status == "fallback")
        fail_count += sum(1 for status in statuses if status in {"failed", "missing"})
        total_obs += len(statuses)

    metrics["cnn_score"] = source_scores.get("cnn")
    metrics["crypto_score"] = source_scores.get("crypto")
    metrics["vix_score"] = source_scores.get("vix")

    overall = sum(source_scores.values()) / max(len(source_scores), 1)
    metrics["overall_score"] = overall
    if overall >= 85:
        metrics["overall_class"] = "up"
    elif overall >= 60:
        metrics["overall_class"] = "neutral"
    else:
        metrics["overall_class"] = "down"

    if total_obs > 0:
        metrics["fallback_rate"] = (fallback_count / total_obs) * 100.0
        metrics["failure_rate"] = (fail_count / total_obs) * 100.0

    return metrics


def get_series_delta(series: pd.Series, lookback: int) -> float | None:
    clean = series.dropna()
    if len(clean) <= lookback:
        return None
    return float(clean.iloc[-1] - clean.iloc[-(lookback + 1)])

data = load_data(DATA_FILE)
if data.empty:
    raise SystemExit("No sentiment data yet. Run python sentiment_tracker.py first.")
health_data = load_health_data(HEALTH_FILE)

required_columns = {
    "timestamp",
    "stock_fear_greed",
    "crypto_fear_greed",
    "vix",
    "vix_normalized",
    "mood_score",
}
missing = required_columns.difference(data.columns)
if missing:
    raise SystemExit(f"Missing columns in {DATA_FILE}: {', '.join(sorted(missing))}")

scored_data = data.dropna(subset=["mood_score"]).copy()
if scored_data.empty:
    raise SystemExit("No valid mood_score records in sentiment_data.csv.")

recent = prepare_recent(scored_data, days=7)
if recent.empty:
    recent = scored_data.tail(7)

latest = scored_data.iloc[-1]
score = float(latest["mood_score"])
label = score_label(score)
status_color = STATUS_COLORS[label]
gauge_label = f"{label} | {STATUS_TAGLINES[label]}"
gauge_rotation = -90.0 + (score / 100.0) * 180.0
updated_time = latest["timestamp"].strftime("%Y-%m-%d %H:%M")

chart_recent = prepare_recent_hours(scored_data, hours=48)
if chart_recent.empty or len(chart_recent) < 2:
    chart_recent = recent.copy()

mood_series = chart_recent.get("mood_score", pd.Series(dtype=float)).dropna().tolist()
if not mood_series:
    mood_series = [score]
ma_series = (
    pd.Series(mood_series).rolling(window=6, min_periods=1).mean().tolist()
    if mood_series
    else []
)
index_points = points_from_series(mood_series, 200.0, 80.0)
ma_points = points_from_series(ma_series, 200.0, 80.0)

stock_latest = float(latest["stock_fear_greed"]) if pd.notna(latest["stock_fear_greed"]) else None
crypto_latest = (
    float(latest["crypto_fear_greed"]) if pd.notna(latest["crypto_fear_greed"]) else None
)
vix_norm_latest = float(latest["vix_normalized"]) if pd.notna(latest["vix_normalized"]) else None
vix_inverse_latest = (100.0 - vix_norm_latest) if vix_norm_latest is not None else None

stock_series_chart = chart_recent.get("stock_fear_greed", pd.Series(dtype=float)).dropna().tolist()
crypto_series_chart = chart_recent.get("crypto_fear_greed", pd.Series(dtype=float)).dropna().tolist()
vix_norm_series_chart = chart_recent.get("vix_normalized", pd.Series(dtype=float)).dropna()
vix_inv_series_chart = (100.0 - vix_norm_series_chart).clip(lower=0.0, upper=100.0).tolist()

stock_points = points_from_series(stock_series_chart, 190.0, 55.0)
crypto_points = points_from_series(crypto_series_chart, 190.0, 55.0)
vix_points = points_from_series(vix_inv_series_chart, 190.0, 55.0)

forecast_source = prepare_recent_hours(scored_data, hours=24 * 30)
if forecast_source.empty:
    forecast_source = chart_recent.copy()

forecast_stock_series = forecast_source.get("stock_fear_greed", pd.Series(dtype=float)).dropna().tolist()
forecast_crypto_series = forecast_source.get("crypto_fear_greed", pd.Series(dtype=float)).dropna().tolist()
forecast_vix_norm_series = forecast_source.get("vix_normalized", pd.Series(dtype=float)).dropna()
forecast_vix_inverse_series = (
    (100.0 - forecast_vix_norm_series).clip(lower=0.0, upper=100.0).tolist()
)

composite_forecast = build_component_composite_forecast(
    forecast_stock_series,
    forecast_crypto_series,
    forecast_vix_inverse_series,
    stock_now=stock_latest,
    crypto_now=crypto_latest,
    vix_inverse_now=vix_inverse_latest,
    steps=24,
    lookback=96,
)
forecast_values = composite_forecast["base"]
forecast_upper = composite_forecast["bull"]
forecast_lower = composite_forecast["bear"]
forecast_slope = composite_forecast["slope"]
forecast_sigma = composite_forecast["sigma"]
forecast_phi = composite_forecast["phi"]
forecast_driver = composite_forecast["dominant_driver"]
forecast_contrib_stock = composite_forecast["contrib_stock"]
forecast_contrib_crypto = composite_forecast["contrib_crypto"]
forecast_contrib_vix = composite_forecast["contrib_vix"]

historical_tail = mood_series[-24:] if len(mood_series) >= 2 else [score, score]
if not forecast_values:
    forecast_values = [historical_tail[-1]] * 24
    forecast_upper = [historical_tail[-1]] * 24
    forecast_lower = [historical_tail[-1]] * 24

forecast_total_points = len(historical_tail) + len(forecast_values)
forecast_actual_points = points_from_segment(
    historical_tail,
    start_index=0,
    total_points=forecast_total_points,
    width=200.0,
    height=75.0,
)
forecast_line_points = points_from_segment(
    [historical_tail[-1]] + forecast_values,
    start_index=len(historical_tail) - 1,
    total_points=forecast_total_points,
    width=200.0,
    height=75.0,
)
forecast_upper_points = points_from_segment(
    [historical_tail[-1]] + forecast_upper,
    start_index=len(historical_tail) - 1,
    total_points=forecast_total_points,
    width=200.0,
    height=75.0,
)
forecast_lower_points = points_from_segment(
    [historical_tail[-1]] + forecast_lower,
    start_index=len(historical_tail) - 1,
    total_points=forecast_total_points,
    width=200.0,
    height=75.0,
)
forecast_target = forecast_values[-1]
forecast_delta = forecast_target - historical_tail[-1]
forecast_state = class_from_delta(forecast_delta, epsilon=1.0)
forecast_regime = score_label(forecast_target)
forecast_bull_target = forecast_upper[-1]
forecast_bear_target = forecast_lower[-1]
forecast_spread = forecast_bull_target - forecast_bear_target

backtest_source = prepare_recent_hours(scored_data, hours=24 * 120)
if backtest_source.empty:
    backtest_source = forecast_source.copy()
forecast_backtest = compute_forecast_backtest(
    backtest_source,
    horizons=(1, 6, 24),
    lookback=96,
    max_windows=220,
)
bt_windows = forecast_backtest["windows"]
bt_mae_1h = forecast_backtest["mae_1h"]
bt_mae_6h = forecast_backtest["mae_6h"]
bt_mae_24h = forecast_backtest["mae_24h"]
bt_hit_24h = forecast_backtest["hit_24h"]
bt_rmse_24h = forecast_backtest["rmse_24h"]
bt_cov_24h = forecast_backtest["coverage_24h"]
bt_bias_24h = forecast_backtest["bias_24h"]
bt_drift_24h = forecast_backtest["drift_24h"]
bt_error_series = forecast_backtest["error_series_24h"]

bt_mae_24_class = class_from_score(
    -bt_mae_24h if bt_mae_24h is not None else None,
    low=-10.0,
    high=-5.0,
)
bt_hit_24_class = class_from_score(bt_hit_24h, low=45.0, high=60.0)
bt_cov_24_class = class_from_score(bt_cov_24h, low=55.0, high=75.0)
bt_bias_24_class = class_from_score(
    -abs(bt_bias_24h) if bt_bias_24h is not None else None,
    low=-5.0,
    high=-2.0,
)
bt_drift_24_class = class_from_score(
    -bt_drift_24h if bt_drift_24h is not None else None,
    low=-1.0,
    high=0.3,
)
bt_error_points = points_from_series(
    bt_error_series[-80:] if bt_error_series else [0.0],
    width=200.0,
    height=55.0,
    min_value=0.0,
    max_value=max(4.0, max(bt_error_series) if bt_error_series else 4.0),
)

regime_sigma = max(1.0, forecast_spread / 4.0)
prob_stormy, prob_cloudy, prob_shiny = forecast_regime_probabilities(
    forecast_target, regime_sigma
)
regime_probs = {
    "STORMY": prob_stormy,
    "CLOUDY": prob_cloudy,
    "SHINY": prob_shiny,
}
regime_top = max(regime_probs, key=regime_probs.get)
regime_top_class = (
    "down" if regime_top == "STORMY" else "up" if regime_top == "SHINY" else "neutral"
)

waterfall_rows_html, waterfall_total_delta = build_driver_waterfall_rows(
    forecast_contrib_stock,
    forecast_contrib_crypto,
    forecast_contrib_vix,
)
waterfall_total_class = class_from_delta(waterfall_total_delta, epsilon=0.05)

health_metrics = compute_feed_health_metrics(scored_data, health_data)

daily_window = prepare_recent(scored_data, days=14)
daily_scores = daily_window.get("mood_score", pd.Series(dtype=float))
day_delta = get_series_delta(daily_scores, lookback=1)
week_delta = get_series_delta(daily_scores, lookback=7)

mood_hourly = pd.Series(mood_series)
momentum_6h = get_series_delta(mood_hourly, lookback=6)
intraday_vol = float(mood_hourly.tail(24).std()) if len(mood_hourly) >= 3 else None

component_values = [
    value
    for value in [stock_latest, crypto_latest, vix_inverse_latest]
    if value is not None and not pd.isna(value)
]
dispersion = (
    float(max(component_values) - min(component_values))
    if len(component_values) >= 2
    else None
)

if momentum_6h is None:
    momentum_text = "--"
    momentum_state = "neutral"
    momentum_detail = "Insufficient data"
else:
    momentum_text = format_signed(momentum_6h, 1)
    momentum_state = class_from_delta(momentum_6h, epsilon=1.0)
    if momentum_6h > 2:
        momentum_detail = "Acceleration"
    elif momentum_6h < -2:
        momentum_detail = "Deceleration"
    else:
        momentum_detail = "Stable"

if intraday_vol is None:
    vol_state = "neutral"
    vol_detail = "Insufficient data"
else:
    vol_state = class_from_score(-intraday_vol, low=-8.0, high=-4.0)
    if intraday_vol >= 8:
        vol_detail = "Elevated volatility"
    elif intraday_vol >= 4:
        vol_detail = "Moderate volatility"
    else:
        vol_detail = "Calm"

if dispersion is None:
    dispersion_state = "neutral"
    consensus_text = "Consensus --"
else:
    if dispersion <= 15:
        consensus_text = "Consensus High"
    elif dispersion <= 30:
        consensus_text = "Consensus Medium"
    else:
        consensus_text = "Consensus Low"
    dispersion_state = class_from_score(-dispersion, low=-30.0, high=-15.0)

asset_weights = compute_asset_weights()
asset_prices = fetch_asset_prices(tuple(asset["ticker"] for asset in ASSETS))
asset_rows = build_asset_rows(asset_prices)

asset_cards_html = "\n".join(
    build_asset_card(row, asset_weights.get(row["ticker"], 0.0))
    for row in asset_rows
)
if not asset_cards_html:
    asset_cards_html = "<div>NO ASSET DATA</div>"

ticker_tape_html = "".join(
    f"""
<div class="tape-item {score_class(row.get('score'))}">
  <span class="tape-ticker">{row.get('ticker', '')}</span>
  <span class="tape-score">{format_value(row.get('score'), 0)}</span>
</div>
"""
    for row in asset_rows
)
if not ticker_tape_html:
    ticker_tape_html = '<div class="tape-item neutral"><span class="tape-ticker">NO DATA</span></div>'

valid_asset_scores = [
    float(row["score"])
    for row in asset_rows
    if row.get("score") is not None and not pd.isna(row.get("score"))
]
total_assets = len(valid_asset_scores)
bull_count = sum(1 for s in valid_asset_scores if s >= 60)
bear_count = sum(1 for s in valid_asset_scores if s < 40)
neutral_count = max(0, total_assets - bull_count - bear_count)

if total_assets > 0:
    bull_pct = (bull_count / total_assets) * 100.0
    neutral_pct = (neutral_count / total_assets) * 100.0
    bear_pct = (bear_count / total_assets) * 100.0
    breadth_score = ((bull_count - bear_count) / total_assets) * 100.0
else:
    bull_pct = neutral_pct = bear_pct = 0.0
    breadth_score = None

if breadth_score is None:
    breadth_state = "neutral"
else:
    breadth_state = class_from_score(breadth_score, low=-20.0, high=20.0)

us_weight, asia_weight, eu_weight = compute_region_weights(asset_weights)

history_rows = recent.sort_values("timestamp", ascending=False)
history_items = [
    build_history_item(row["timestamp"], float(row["mood_score"]))
    for _, row in history_rows.iterrows()
    if pd.notna(row.get("mood_score"))
]
history_items_html = "\n".join(history_items) or "<div>NO HISTORY</div>"

cnn_score = format_value(stock_latest, 1)
crypto_score = format_value(crypto_latest, 1)
vix_inverse_score = format_value(vix_inverse_latest, 1)
gap_value = (score - stock_latest) if stock_latest is not None else None
gap_color = "var(--accent-green)" if gap_value is not None and gap_value >= 0 else "var(--accent-red)"
gap_label = format_signed(gap_value, 1)
forecast_detail = (
    f"Driver {forecast_driver} | "
    f"S {format_signed(forecast_contrib_stock, 1)} "
    f"C {format_signed(forecast_contrib_crypto, 1)} "
    f"V {format_signed(forecast_contrib_vix, 1)}"
)

health_rows_html = "".join(
    [
        f"""
<div class="health-row">
  <span class="health-name">CNN</span>
  <span class="health-status {health_status_class(str(health_metrics['cnn_status']))}">{health_status_label(str(health_metrics['cnn_status']))}</span>
  <span class="health-score">{format_value(health_metrics['cnn_score'], 1)}%</span>
</div>
""",
        f"""
<div class="health-row">
  <span class="health-name">Crypto</span>
  <span class="health-status {health_status_class(str(health_metrics['crypto_status']))}">{health_status_label(str(health_metrics['crypto_status']))}</span>
  <span class="health-score">{format_value(health_metrics['crypto_score'], 1)}%</span>
</div>
""",
        f"""
<div class="health-row">
  <span class="health-name">VIX</span>
  <span class="health-status {health_status_class(str(health_metrics['vix_status']))}">{health_status_label(str(health_metrics['vix_status']))}</span>
  <span class="health-score">{format_value(health_metrics['vix_score'], 1)}%</span>
</div>
""",
    ]
)

signal_rows = [
    build_signal_row(
        "Regime",
        label,
        STATUS_TAGLINES[label],
        "up" if label == "SHINY" else "down" if label == "STORMY" else "neutral",
    ),
    build_signal_row(
        "Transition Prob",
        regime_top,
        f"S {format_value(prob_stormy, 1)}% | C {format_value(prob_cloudy, 1)}% | G {format_value(prob_shiny, 1)}%",
        regime_top_class,
    ),
    build_signal_row(
        "Momentum 6H",
        momentum_text,
        momentum_detail,
        momentum_state,
    ),
    build_signal_row(
        "Forecast 24H",
        format_value(forecast_target, 1),
        forecast_detail,
        forecast_state,
    ),
    build_signal_row(
        "Forecast Spread",
        format_value(forecast_spread, 1),
        f"sigma {format_value(forecast_sigma, 2)} | slope {format_signed(forecast_slope, 2)}/h",
        class_from_score(-forecast_spread, low=-16.0, high=-8.0),
    ),
    build_signal_row(
        "Backtest 24H",
        format_value(bt_mae_24h, 2),
        f"RMSE {format_value(bt_rmse_24h, 2)} | Hit {format_value(bt_hit_24h, 1)}% | N {bt_windows}",
        bt_mae_24_class,
    ),
    build_signal_row(
        "Intraday Vol",
        format_value(intraday_vol, 2),
        vol_detail,
        vol_state,
    ),
    build_signal_row(
        "Component Spread",
        format_value(dispersion, 1),
        consensus_text,
        dispersion_state,
    ),
    build_signal_row(
        "Market Breadth",
        format_signed(breadth_score, 1),
        f"Bull {bull_count} | Neutral {neutral_count} | Bear {bear_count}",
        breadth_state,
    ),
]
signals_html = "\n".join(signal_rows)

template_path = TEMPLATE_FILE
if not template_path.exists():
    raise SystemExit(f"Missing {template_path}.")

template_html = template_path.read_text(encoding="utf-8")

replacements = {
    "{GLOBAL_SCORE}": format_value(score, 1),
    "{STATUS_COLOR}": status_color,
    "{GAUGE_LABEL}": gauge_label,
    "{GAUGE_ROTATION}": f"{gauge_rotation:.1f}",
    "{MA_CHART_POINTS}": ma_points,
    "{INDEX_CHART_POINTS}": index_points,
    "{FORECAST_ACTUAL_POINTS}": forecast_actual_points,
    "{FORECAST_LINE_POINTS}": forecast_line_points,
    "{FORECAST_HIGH_POINTS}": forecast_upper_points,
    "{FORECAST_LOW_POINTS}": forecast_lower_points,
    "{FORECAST_TARGET}": format_value(forecast_target, 1),
    "{FORECAST_BULL_TARGET}": format_value(forecast_bull_target, 1),
    "{FORECAST_BEAR_TARGET}": format_value(forecast_bear_target, 1),
    "{FORECAST_SPREAD}": format_value(forecast_spread, 1),
    "{FORECAST_DELTA}": format_signed(forecast_delta, 1),
    "{FORECAST_CLASS}": forecast_state,
    "{FORECAST_REGIME}": forecast_regime,
    "{REGIME_P_STORMY}": format_value(prob_stormy, 1),
    "{REGIME_P_CLOUDY}": format_value(prob_cloudy, 1),
    "{REGIME_P_SHINY}": format_value(prob_shiny, 1),
    "{REGIME_TOP}": regime_top,
    "{REGIME_TOP_CLASS}": regime_top_class,
    "{WATERFALL_ROWS}": waterfall_rows_html,
    "{WATERFALL_TOTAL}": format_signed(waterfall_total_delta, 2),
    "{WATERFALL_TOTAL_CLASS}": waterfall_total_class,
    "{STOCK_POINTS}": stock_points,
    "{CRYPTO_POINTS}": crypto_points,
    "{VIX_POINTS}": vix_points,
    "{ASSET_CARDS}": asset_cards_html,
    "{HISTORY_ITEMS}": history_items_html,
    "{SIGNAL_ROWS}": signals_html,
    "{TICKER_TAPE}": ticker_tape_html,
    "{CNN_SCORE}": cnn_score,
    "{CRYPTO_SCORE}": crypto_score,
    "{VIX_INV_SCORE}": vix_inverse_score,
    "{CNN_GAP}": gap_label,
    "{GAP_COLOR}": gap_color,
    "{UPDATED_TIME}": updated_time,
    "{US_WEIGHT}": format_value(us_weight, 1),
    "{ASIA_WEIGHT}": format_value(asia_weight, 1),
    "{EU_WEIGHT}": format_value(eu_weight, 1),
    "{KPI_DAY_DELTA}": format_signed(day_delta, 1),
    "{KPI_DAY_CLASS}": class_from_delta(day_delta, epsilon=0.5),
    "{KPI_WEEK_DELTA}": format_signed(week_delta, 1),
    "{KPI_WEEK_CLASS}": class_from_delta(week_delta, epsilon=1.5),
    "{KPI_VOL}": format_value(intraday_vol, 2),
    "{KPI_VOL_CLASS}": vol_state,
    "{KPI_DISPERSION}": format_value(dispersion, 1),
    "{KPI_DISPERSION_CLASS}": dispersion_state,
    "{KPI_MOMENTUM}": momentum_text,
    "{KPI_MOMENTUM_CLASS}": momentum_state,
    "{BT_MAE_1H}": format_value(bt_mae_1h, 2),
    "{BT_MAE_6H}": format_value(bt_mae_6h, 2),
    "{BT_MAE_24H}": format_value(bt_mae_24h, 2),
    "{BT_RMSE_24H}": format_value(bt_rmse_24h, 2),
    "{BT_HIT_24H}": format_value(bt_hit_24h, 1),
    "{BT_COVERAGE_24H}": format_value(bt_cov_24h, 1),
    "{BT_BIAS_24H}": format_signed(bt_bias_24h, 2),
    "{BT_DRIFT_24H}": format_signed(bt_drift_24h, 2),
    "{BT_WINDOWS}": str(bt_windows),
    "{BT_MAE_24H_CLASS}": bt_mae_24_class,
    "{BT_HIT_24H_CLASS}": bt_hit_24_class,
    "{BT_COVERAGE_24H_CLASS}": bt_cov_24_class,
    "{BT_BIAS_24H_CLASS}": bt_bias_24_class,
    "{BT_DRIFT_24H_CLASS}": bt_drift_24_class,
    "{BT_ERROR_POINTS}": bt_error_points,
    "{HEALTH_ROWS}": health_rows_html,
    "{HEALTH_AGE_MIN}": format_value(health_metrics["age_minutes"], 1),
    "{HEALTH_AGE_CLASS}": str(health_metrics["age_class"]),
    "{HEALTH_OVERALL_SCORE}": format_value(health_metrics["overall_score"], 1),
    "{HEALTH_OVERALL_CLASS}": str(health_metrics["overall_class"]),
    "{HEALTH_FALLBACK_RATE}": format_value(health_metrics["fallback_rate"], 1),
    "{HEALTH_FAILURE_RATE}": format_value(health_metrics["failure_rate"], 1),
    "{BULL_COUNT}": str(bull_count),
    "{NEUTRAL_COUNT}": str(neutral_count),
    "{BEAR_COUNT}": str(bear_count),
    "{BULL_PCT}": format_value(bull_pct, 1),
    "{NEUTRAL_PCT}": format_value(neutral_pct, 1),
    "{BEAR_PCT}": format_value(bear_pct, 1),
}

for key, value in replacements.items():
    template_html = template_html.replace(key, value)

OUTPUT_FILE.write_text(template_html, encoding="utf-8")
print(f"Wrote static dashboard: {OUTPUT_FILE}")
