"""
Global Cloudy&Shiny Index

Builds a market sentiment gauge from risk and fear assets using price trend,
momentum (RSI), and inverse logic for safe havens. Produces a live terminal
dashboard and a 2-year backtest chart while logging each run.
"""

from __future__ import annotations

import datetime as dt
import io
import os
import time
import contextlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

from curl_cffi import requests as cf_requests
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf


# --- Phase 1: Architecture Plan ---
# DataManager: fetch recent and historical adjusted prices, align/fill gaps.
# SentimentEngine: compute MA, RSI, per-asset scores (with inverse logic), and
#                  aggregate the global index, both for snapshots and vectors.
# Visualizer: render terminal dashboard, ASCII bar, weather labels, persist
#             history, and chart the 2-year index with weather zones.


@dataclass
class Asset:
    ticker: str
    name: str
    is_fear: bool


ASSETS: List[Asset] = [
    Asset("SPY", "S&P 500", False),
    Asset("QQQ", "Nasdaq 100", False),
    Asset("^GDAXI", "DAX", False),
    Asset("^FCHI", "CAC 40", False),
    Asset("^N225", "Nikkei 225", False),
    Asset("000001.SS", "Shanghai Comp", False),
    Asset("^HSI", "Hang Seng", False),
    Asset("XU100.IS", "BIST 100", False),
    Asset("^VIX", "VIX", True),
    Asset("TLT", "US 20Y Treasury", True),
    Asset("GLD", "Gold", True),
    Asset("DX-Y.NYB", "US Dollar Index", True),
]

# GDP-based weighting (approx. 2023 nominal USD trillions; static constants)
GDP_BY_COUNTRY: Dict[str, float] = {
    "US": 26.9,
    "Germany": 4.4,
    "France": 3.0,
    "Japan": 4.2,
    "China": 17.5,
    "HongKong": 0.36,
    "Turkey": 1.15,
}

# Map tickers to country buckets for GDP-based weights.
TICKER_COUNTRY: Dict[str, str] = {
    "SPY": "US",
    "QQQ": "US",
    "^GDAXI": "Germany",
    "^FCHI": "France",
    "^N225": "Japan",
    "000001.SS": "China",
    "^HSI": "HongKong",
    "XU100.IS": "Turkey",
}

# Share of total weight allocated to risk vs. fear baskets.
RISK_WEIGHT_SHARE = 0.7
FEAR_WEIGHT_SHARE = 0.3

CNN_FGI_URL = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"


class DataManager:
    def __init__(self, assets: List[Asset]):
        self.assets = assets

    def _safe_download(
        self, tickers, period: str | None, start: dt.date | None, end: dt.date | None
    ) -> pd.DataFrame:
        """Wrap yfinance download with retries and quieted stderr/stdout noise."""
        last_err: Exception | None = None
        for attempt in range(3):
            try:
                buf_out, buf_err = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
                    data = yf.download(
                        tickers=tickers,
                        period=period,
                        start=start,
                        end=end,
                        interval="1d",
                        auto_adjust=False,
                        progress=False,
                        threads=False,
                    )
                if not isinstance(data, pd.DataFrame) or data.empty:
                    last_err = RuntimeError("yfinance returned empty data")
                else:
                    return data
            except Exception as exc:
                last_err = exc
            time.sleep(1.5)
        if last_err:
            print(f"yfinance download failed after retries for {tickers}: {last_err}")
        return pd.DataFrame()

    def _download_single(
        self, ticker: str, period: str | None, start: dt.date | None, end: dt.date | None
    ) -> pd.Series | None:
        """Single-ticker fallback fetch for when multi-download drops a column."""
        data = self._safe_download(ticker, period, start, end)
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            series = data["Adj Close"]
            if isinstance(series, pd.DataFrame):
                series = series.iloc[:, 0]
        elif "Adj Close" in data:
            series = data["Adj Close"]
        else:
            series = data.squeeze()
        return series.ffill()

    def fetch_recent(self, lookback_days: int = 260) -> pd.DataFrame:
        """Latest prices for dashboard; enough history for 60d MA/RSI."""
        return self._download(period=f"{lookback_days}d")

    def fetch_history(self, years: int = 2) -> pd.DataFrame:
        """History with buffer for indicators."""
        end = dt.date.today()
        # Add buffer days for indicators so early values are valid.
        start = end - dt.timedelta(days=years * 365 + 200)
        return self._download(start=start, end=end)

    def _download(
        self, period: str | None = None, start: dt.date | None = None, end: dt.date | None = None
    ) -> pd.DataFrame:
        tickers = [a.ticker for a in self.assets]
        data = self._safe_download(tickers, period, start, end)
        if data.empty:
            raise RuntimeError("No data returned from yfinance.")

        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Adj Close"].copy()
        else:
            prices = data.copy()

        prices = prices.reindex(columns=tickers)
        missing = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]

        if missing:
            print(f"Retrying tickers individually due to missing data: {', '.join(missing)}")
        for ticker in missing:
            fallback = self._download_single(ticker, period, start, end)
            if fallback is None or fallback.dropna().empty:
                continue
            combined_index = prices.index.union(fallback.index)
            prices = prices.reindex(combined_index)
            prices[ticker] = fallback.reindex(combined_index)

        prices = prices.sort_index().ffill()
        prices = prices.reindex(columns=tickers)
        if prices.dropna(how="all").empty:
            raise RuntimeError("No usable price data after fallback attempts.")
        return prices


class SentimentEngine:
    def __init__(self, assets: List[Asset]):
        self.assets = assets
        self.weights = self._build_weights()

    def _build_weights(self) -> Dict[str, float]:
        risk_assets = [a for a in self.assets if not a.is_fear]
        fear_assets = [a for a in self.assets if a.is_fear]

        # Country-level GDP weights, split among that country's tickers.
        country_groups: Dict[str, List[str]] = {}
        for asset in risk_assets:
            country = TICKER_COUNTRY.get(asset.ticker)
            if country is None:
                continue
            country_groups.setdefault(country, []).append(asset.ticker)

        risk_weights_raw: Dict[str, float] = {}
        for country, tickers in country_groups.items():
            gdp = GDP_BY_COUNTRY.get(country, 1.0)
            split = gdp / len(tickers)
            for t in tickers:
                risk_weights_raw[t] = split

        risk_total = sum(risk_weights_raw.values())
        risk_weights = {
            t: (w / risk_total) * RISK_WEIGHT_SHARE if risk_total else 0.0
            for t, w in risk_weights_raw.items()
        }

        fear_weight_each = FEAR_WEIGHT_SHARE / len(fear_assets) if fear_assets else 0.0
        fear_weights = {a.ticker: fear_weight_each for a in fear_assets}

        weights = {**fear_weights, **risk_weights}
        total = sum(weights.values())
        if total == 0:
            # Fallback to equal weights if something went wrong.
            equal = 1 / len(self.assets)
            weights = {a.ticker: equal for a in self.assets}
        return weights

    @staticmethod
    def rsi(prices: pd.DataFrame, window: int = 14) -> pd.DataFrame:
        delta = prices.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def score_snapshot(
        self, prices: pd.Series, ma60: pd.Series, rsi: pd.Series
    ) -> Tuple[List[Dict], float]:
        rows = []
        for asset in self.assets:
            price = prices[asset.ticker]
            ma = ma60[asset.ticker]
            asset_rsi = rsi[asset.ticker]
            score = self._score_single(price, ma, asset_rsi, asset.is_fear)
            status = self._status_label(score, asset_rsi, asset.is_fear)
            rows.append(
                {
                    "ticker": asset.ticker,
                    "name": asset.name,
                    "price": price,
                    "ma60": ma,
                    "rsi": asset_rsi,
                    "score": score,
                    "status": status,
                }
            )
        valid_rows = [r for r in rows if not pd.isna(r["score"])]
        if valid_rows:
            weight_sum = sum(self.weights.get(r["ticker"], 0) for r in valid_rows)
            weighted_scores = sum(r["score"] * self.weights.get(r["ticker"], 0) for r in valid_rows)
            if weight_sum:
                global_score = float(weighted_scores / weight_sum)
            else:
                global_score = float(np.nanmean([r["score"] for r in valid_rows]))
        else:
            global_score = float("nan")
        return rows, global_score

    def _score_single(self, price: float, ma60: float, rsi: float, is_fear: bool) -> float:
        if pd.isna(price) or pd.isna(ma60):
            return np.nan
        deviation = (price - ma60) / ma60
        deviation = float(np.clip(deviation, -0.1, 0.1))
        score = 50 + deviation * 500
        if is_fear:
            score = 100 - score

        # RSI adjustments apply only to risk assets.
        if not is_fear and not pd.isna(rsi):
            if score > 60 and rsi > 70:
                score -= 15
            elif score < 40 and rsi < 30:
                score += 10
        return float(np.clip(score, 0, 100))

    def _status_label(self, score: float, rsi: float, is_fear: bool) -> str:
        flags = []
        if not is_fear and not pd.isna(rsi):
            if score > 60 and rsi > 70:
                flags.append("Overheated")
            if score < 40 and rsi < 30:
                flags.append("Oversold")
        return ", ".join(flags) if flags else ""

    def score_history(self, prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        ma60 = prices.rolling(60).mean()
        rsi = self.rsi(prices)
        deviation = (prices - ma60) / ma60
        deviation = deviation.clip(-0.1, 0.1)
        scores = 50 + deviation * 500

        fear_tickers = [a.ticker for a in self.assets if a.is_fear]
        risk_tickers = [a.ticker for a in self.assets if not a.is_fear]

        if fear_tickers:
            scores[fear_tickers] = 100 - scores[fear_tickers]

        if risk_tickers:
            high_mask = (scores[risk_tickers] > 60) & (rsi[risk_tickers] > 70)
            low_mask = (scores[risk_tickers] < 40) & (rsi[risk_tickers] < 30)
            scores[risk_tickers] = scores[risk_tickers] - high_mask * 15 + low_mask * 10

        scores = scores.clip(0, 100)
        weight_series = pd.Series(self.weights)
        index_series = (scores * weight_series).sum(axis=1) / weight_series.sum()
        # Drop early NaNs from indicators for a clean chart.
        valid = (~scores.isna()).any(axis=1)
        index_series = index_series[valid]
        scores = scores.loc[index_series.index]
        return scores, index_series


class CNNFearGreedFetcher:
    """Fetch CNN Fear & Greed Index via their dataviz endpoint."""

    def fetch(self) -> Tuple[float | None, str | None]:
        try:
            resp = cf_requests.get(CNN_FGI_URL, impersonate="chrome120", timeout=10)
            if resp.status_code != 200:
                return None, None
            data = resp.json()
            fg = data.get("fear_and_greed") or data.get("fear_and_greed_index") or {}
            score = fg.get("score")
            label = fg.get("rating") or fg.get("classification")
            if score is None:
                return None, None
            label_out = label if label else ""
            return float(score), label_out
        except Exception:
            return None, None


class BacktestEvaluator:
    """Relate index levels to forward benchmark returns to gauge predictiveness."""

    def __init__(self, benchmark_ticker: str = "ACWI"):
        self.benchmark_ticker = benchmark_ticker

    def _fetch_benchmark(self, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series | None:
        data = yf.download(
            tickers=self.benchmark_ticker,
            start=start,
            end=end,
            interval="1d",
            auto_adjust=False,
            progress=False,
        )
        if data.empty:
            return None
        if isinstance(data.columns, pd.MultiIndex):
            series = data["Adj Close"].copy()
        else:
            series = data["Adj Close"]
        if isinstance(series, pd.DataFrame):
            series = series.iloc[:, 0]
        return series.ffill()

    def evaluate(self, index_series: pd.Series) -> None:
        if index_series.empty:
            print("Backtest evaluator: index series empty; skipping.")
            return
        start = index_series.index.min() - pd.Timedelta(days=5)
        end = index_series.index.max() + pd.Timedelta(days=5)
        bench = self._fetch_benchmark(start, end)
        if bench is None:
            print("Backtest evaluator: failed to fetch benchmark data.")
            return
        bench = bench.reindex(index_series.index).ffill()

        fwd = pd.DataFrame(
            {
                "score": index_series,
                "fwd_1d": bench.shift(-1) / bench - 1,
                "fwd_5d": bench.shift(-5) / bench - 1,
                "fwd_20d": bench.shift(-20) / bench - 1,
            }
        ).dropna()

        bins = [0, 20, 40, 60, 80, 100]
        fwd["bucket"] = pd.cut(fwd["score"], bins=bins, right=True, include_lowest=True)

        grouped = fwd.groupby("bucket")
        summary = grouped.agg(
            count=("score", "size"),
            mean_1d=("fwd_1d", "mean"),
            hit_1d=("fwd_1d", lambda x: (x > 0).mean()),
            mean_5d=("fwd_5d", "mean"),
            hit_5d=("fwd_5d", lambda x: (x > 0).mean()),
            mean_20d=("fwd_20d", "mean"),
            hit_20d=("fwd_20d", lambda x: (x > 0).mean()),
        )

        print("\n=== Backtest: ACWI forward returns by Cloudy&Shiny bucket ===")
        print(
            f"{'Bucket':<18} {'Obs':>5} {'Avg 1d':>10} {'Hit1d':>8} "
            f"{'Avg 5d':>10} {'Hit5d':>8} {'Avg 20d':>10} {'Hit20d':>8}"
        )
        for idx, row in summary.iterrows():
            print(
                f"{str(idx):<18} "
                f"{int(row['count']):>5d} "
                f"{row['mean_1d']*100:>9.2f}% {row['hit_1d']*100:>7.1f}% "
                f"{row['mean_5d']*100:>9.2f}% {row['hit_5d']*100:>7.1f}% "
                f"{row['mean_20d']*100:>9.2f}% {row['hit_20d']*100:>7.1f}%"
            )


class Visualizer:
    def __init__(self, assets: List[Asset], history_path: str = "weather_history.csv"):
        self.assets = assets
        self.history_path = history_path

    @staticmethod
    def weather_label(score: float) -> str:
        if score <= 20:
            return "Cloudy (Extreme Fear)"
        if score <= 40:
            return "Partially Cloudy"
        if score <= 60:
            return "Partially Sunny (Neutral)"
        if score <= 80:
            return "Almost Sunny"
        return "Sunny (Extreme Greed)"

    @staticmethod
    def progress_bar(score: float, width: int = 20) -> str:
        filled = int(round(score / 100 * width))
        filled = max(0, min(width, filled))
        return "[" + "#" * filled + "-" * (width - filled) + "]"

    def print_dashboard(
        self,
        rows: List[Dict],
        global_score: float,
        delta: float | None,
        cnn_score: float | None,
        cnn_label: str | None,
    ) -> None:
        print("\n=== Global Cloudy&Shiny Index (Live) ===")
        header = f"{'Ticker':<10} {'Asset':<18} {'Price':>10} {'60-MA':>10} {'RSI':>8} {'Score':>8}  Status"
        print(header)
        print("-" * len(header))
        for row in rows:
            print(
                f"{row['ticker']:<10} {row['name']:<18} "
                f"{row['price']:>10.2f} {row['ma60']:>10.2f} {row['rsi']:>8.2f} "
                f"{row['score']:>8.2f}  {row['status']}"
            )
        print("\nGlobal Score: {:.2f} {}".format(global_score, self.progress_bar(global_score)))
        label = self.weather_label(global_score)
        if delta is None:
            trend = "Trend: (first run)"
        else:
            arrow = "Up" if delta > 0 else ("Down" if delta < 0 else "Flat")
            trend = f"Trend: {arrow} {delta:+.2f} points vs last run"
        print(label)
        print(trend)
        if cnn_score is not None:
            gap = global_score - cnn_score
            label_text = cnn_label if cnn_label else ""
            print(f"CNN Fear & Greed: {cnn_score:.1f} ({label_text}) | Gap vs Cloudy&Shiny: {gap:+.2f}")
        else:
            print("CNN Fear & Greed: unavailable (fetch failed)")

    def log_history(self, timestamp: dt.datetime, score: float) -> Tuple[float | None, pd.DataFrame]:
        entry = pd.DataFrame([{"timestamp": timestamp.isoformat(), "score": score}])
        if os.path.exists(self.history_path):
            history = pd.read_csv(self.history_path)
            prev_score = float(history["score"].iloc[-1]) if not history.empty else None
            history = pd.concat([history, entry], ignore_index=True)
        else:
            history = entry
            prev_score = None
        history.to_csv(self.history_path, index=False)
        return prev_score, history

    def plot_history(self, index_series: pd.Series) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))
        index_series.plot(ax=ax, color="black", linewidth=1.2, label="Cloudy&Shiny Index")

        zones = [
            (0, 20, "#c0392b", "Cloudy"),
            (20, 40, "#e67e22", "Partially Cloudy"),
            (40, 60, "#f1c40f", "Partially Sunny"),
            (60, 80, "#27ae60", "Almost Sunny"),
            (80, 100, "#2980b9", "Sunny"),
        ]
        for low, high, color, label in zones:
            ax.fill_between(
                index_series.index,
                low,
                high,
                color=color,
                alpha=0.08,
                linewidth=0,
                label=label,
            )

        ax.set_title("Global Cloudy&Shiny Index (2Y Backtest)")
        ax.set_ylabel("Score (0-100)")
        ax.set_xlabel("Date")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper left", ncol=2, fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.tight_layout()
        # Save to the parent directory where index.html lives
        out_path = os.path.join(os.path.dirname(__file__), "../history_chart.png")
        plt.savefig(out_path)
        plt.close(fig) # Close figure to free memory


def main() -> None:
    data_manager = DataManager(ASSETS)
    engine = SentimentEngine(ASSETS)
    viz = Visualizer(ASSETS)
    cnn_fetcher = CNNFearGreedFetcher()
    evaluator = BacktestEvaluator()

    # Mode 1: Live dashboard
    recent = data_manager.fetch_recent()
    ma60_recent = recent.rolling(60).mean().iloc[-1]
    rsi_recent = engine.rsi(recent).iloc[-1]
    rows, global_score = engine.score_snapshot(recent.iloc[-1], ma60_recent, rsi_recent)

    now = dt.datetime.now(dt.timezone.utc)
    prev_score, history = viz.log_history(now, global_score)
    delta = None if prev_score is None else global_score - prev_score
    cnn_score, cnn_label = cnn_fetcher.fetch()
    viz.print_dashboard(rows, global_score, delta, cnn_score, cnn_label)

    # Mode 2: 2-year backtest and chart
    history_prices = data_manager.fetch_history()
    _, index_series = engine.score_history(history_prices)
    # evaluator.evaluate(index_series) # Skip evaluation for automation speed
    
    # Save plot instead of showing
    viz.plot_history(index_series)

    # Mode 3: HTML Generation
    html_gen = HTMLGenerator()
    html_gen.generate(
        global_score=global_score,
        rows=rows,
        history=history,
        cnn_score=cnn_score,
        cnn_label=cnn_label,
        weights=engine.weights
    )

class HTMLGenerator:
    def __init__(self, template_path: str = "../template.html", output_path: str = "../index.html"):
        self.template_path = os.path.join(os.path.dirname(__file__), template_path)
        self.output_path = os.path.join(os.path.dirname(__file__), output_path)

    def _get_status_color(self, score: float) -> str:
        """Return color based on score."""
        if score >= 60:
            return "var(--accent-green)"
        elif score >= 40:
            return "var(--accent-yellow)"
        return "var(--accent-red)"

    def _get_score_class(self, score: float) -> str:
        """Return CSS class based on score."""
        if score >= 60:
            return "bullish"
        elif score >= 40:
            return "neutral"
        return "bearish"

    def _generate_chart_points(self, history: pd.DataFrame, width: int = 200, height: int = 80) -> tuple:
        """Generate SVG polyline points for the trend chart."""
        if len(history) < 2:
            return "0,40 200,40", "0,40 200,40"
        
        # Get last 48 hours of data (or all available)
        recent = history.tail(48).copy()
        scores = recent["score"].astype(float).tolist()
        
        if not scores:
            return "0,40 200,40", "0,40 200,40"
        
        # Normalize scores to chart height (0-100 -> height-0)
        min_score, max_score = 0, 100
        
        # Generate index line points
        index_points = []
        for i, score in enumerate(scores):
            x = (i / max(len(scores) - 1, 1)) * width
            y = height - ((score - min_score) / (max_score - min_score)) * height
            index_points.append(f"{x:.1f},{y:.1f}")
        
        # Generate MA line (simple moving average of the scores)
        ma_window = min(7, len(scores))
        ma_scores = pd.Series(scores).rolling(ma_window, min_periods=1).mean().tolist()
        ma_points = []
        for i, score in enumerate(ma_scores):
            x = (i / max(len(scores) - 1, 1)) * width
            y = height - ((score - min_score) / (max_score - min_score)) * height
            ma_points.append(f"{x:.1f},{y:.1f}")
        
        return " ".join(index_points), " ".join(ma_points)

    def _fetch_crypto(self) -> float:
        """Fetch crypto fear & greed from alternative.me."""
        try:
            r = cf_requests.get("https://api.alternative.me/fng/", timeout=5)
            data = r.json()
            return float(data['data'][0]['value'])
        except:
            return 50.0

    def generate(
        self,
        global_score: float,
        rows: List[Dict],
        history: pd.DataFrame,
        cnn_score: float | None,
        cnn_label: str | None,
        weights: Dict[str, float] | None = None
    ) -> None:
        try:
            with open(self.template_path, "r", encoding="utf-8") as f:
                template = f.read()
            
            # 1. Global Score and Status
            template = template.replace("{GLOBAL_SCORE}", f"{global_score:.2f}")
            
            viz = Visualizer([])
            template = template.replace("{GAUGE_LABEL}", viz.weather_label(global_score).upper())
            template = template.replace("{STATUS_COLOR}", self._get_status_color(global_score))

            # 2. Chart Points
            index_points, ma_points = self._generate_chart_points(history)
            template = template.replace("{INDEX_CHART_POINTS}", index_points)
            template = template.replace("{MA_CHART_POINTS}", ma_points)

            # 3. Regional Weights (calculate from actual weights)
            us_weight = 0.0
            asia_weight = 0.0
            eu_weight = 0.0
            if weights:
                us_tickers = ["SPY", "QQQ"]
                asia_tickers = ["^N225", "000001.SS", "^HSI"]
                eu_tickers = ["^GDAXI", "^FCHI", "XU100.IS"]
                
                for ticker, w in weights.items():
                    if ticker in us_tickers:
                        us_weight += w * 100
                    elif ticker in asia_tickers:
                        asia_weight += w * 100
                    elif ticker in eu_tickers:
                        eu_weight += w * 100
            
            template = template.replace("{US_WEIGHT}", f"{us_weight:.1f}")
            template = template.replace("{ASIA_WEIGHT}", f"{asia_weight:.1f}")
            template = template.replace("{EU_WEIGHT}", f"{eu_weight:.1f}")

            # 4. Asset Cards with MA comparison
            asset_cards = ""
            for row in rows:
                ticker = row.get("ticker", "")
                name = row.get("name", "")
                score = row.get("score", 0)
                price = row.get("price", 0)
                ma60 = row.get("ma60", 0)
                rsi = row.get("rsi", 50)
                
                if pd.isna(score):
                    score = 0
                if pd.isna(price):
                    price = 0
                if pd.isna(ma60):
                    ma60 = price
                if pd.isna(rsi):
                    rsi = 50
                
                # Determine if above or below MA
                above_ma = price >= ma60 if ma60 > 0 else True
                ma_badge_class = "above" if above_ma else "below"
                ma_badge_text = "↑ ABOVE MA" if above_ma else "↓ BELOW MA"
                
                # Get weight percentage
                weight_pct = 0.0
                if weights and ticker in weights:
                    weight_pct = weights[ticker] * 100
                
                score_class = self._get_score_class(score)
                
                asset_cards += f'''
                    <div class="asset-card">
                        <div class="asset-header">
                            <div>
                                <div class="asset-name">{name}</div>
                                <div class="asset-ticker">{ticker}</div>
                            </div>
                            <div class="asset-ma-badge {ma_badge_class}">{ma_badge_text}</div>
                        </div>
                        <div class="asset-score {score_class}">{score:.1f}</div>
                        <div class="asset-meta">
                            <span>RSI: {rsi:.0f}</span>
                            <span>Wt: {weight_pct:.1f}%</span>
                        </div>
                    </div>
                '''
            
            template = template.replace("{ASSET_CARDS}", asset_cards)

            # 5. History Items
            history_items = ""
            recent_hist = history.tail(7).iloc[::-1]
            for _, row in recent_hist.iterrows():
                ts = pd.to_datetime(row["timestamp"])
                s = float(row["score"])
                
                if s < 25:
                    cls = "Extreme Fear"
                    badge_class = "fear"
                elif s < 40:
                    cls = "Fear"
                    badge_class = "fear"
                elif s > 75:
                    cls = "Extreme Greed"
                    badge_class = "greed"
                elif s > 60:
                    cls = "Greed"
                    badge_class = "greed"
                else:
                    cls = "Neutral"
                    badge_class = "neutral"
                
                history_items += f'''
                    <div class="history-item">
                        <div class="history-header">
                            <span class="history-time">{ts.strftime("%H:%M UTC")}</span>
                            <span class="history-badge {badge_class}">{cls}</span>
                        </div>
                        <div class="history-score">{s:.0f}</div>
                        <div class="history-label">{ts.strftime("%Y-%m-%d")}</div>
                    </div>
                '''
            
            template = template.replace("{HISTORY_ITEMS}", history_items)

            # 6. Benchmark Comparison
            crypto_score = self._fetch_crypto()
            template = template.replace("{CNN_SCORE}", f"{cnn_score:.0f}" if cnn_score else "N/A")
            template = template.replace("{CRYPTO_SCORE}", f"{crypto_score:.0f}")
            
            if cnn_score:
                gap = global_score - cnn_score
                gap_color = "var(--accent-green)" if gap >= 0 else "var(--accent-red)"
                gap_str = f"{gap:+.1f}"
            else:
                gap_color = "var(--text-muted)"
                gap_str = "N/A"
            
            template = template.replace("{CNN_GAP}", gap_str)
            template = template.replace("{GAP_COLOR}", gap_color)

            # 7. Timestamp
            now_str = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
            template = template.replace("{UPDATED_TIME}", now_str)
            
            with open(self.output_path, "w", encoding="utf-8") as f:
                f.write(template)
            
            print(f"Successfully generated HTML at {self.output_path}")

        except Exception as e:
            print(f"HTML Generation failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main()

