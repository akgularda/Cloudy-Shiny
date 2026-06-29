<div align="center">
  <img src="docs/logo.png" alt="Cloudy Shiny logo" width="140"/>
  <!-- CODEX: generate a clean product logo here — a minimalist "Cloudy & Shiny" weather/markets mark (a half-cloud / half-sun motif) over a deep navy (#0b1f3a) field, terminal-grade and corporate; square, transparent background. -->

  # Cloudy Shiny
  ### Hourly global fear &amp; greed market-sentiment dashboard

  <!-- badge row -->
  ![status](https://img.shields.io/badge/status-active-brightgreen)
  ![division](https://img.shields.io/badge/Financial%20Intelligence-0b1f3a)
  ![Monarch Castle](https://img.shields.io/badge/Monarch%20Castle-Holdings-1f6feb)
  ![license](https://img.shields.io/badge/license-see%20LICENSE-lightgrey)
</div>

> **Executive summary** — Cloudy Shiny is a self-refreshing market-sentiment terminal that condenses equities, crypto, and volatility signals into a single, decision-grade **Cloudy&Shiny Index** (0–100). It serves analysts, traders, and decision-makers who need a fast, defensible read on global risk appetite without staring at a dozen feeds. Every reading is timestamped, provenance-logged, and rebuilt automatically on an hourly cadence, then published as a static dashboard on GitHub Pages.

## ✨ Highlights
- **Composite Cloudy&Shiny Index (0–100)** blending stock momentum, crypto fear/greed, and inverted volatility into one read, classified into **STORMY** (extreme fear), **CLOUDY** (neutral), and **SHINY** (extreme greed) regimes.
- **Three independent live inputs** — equity-momentum scoring (SPY/QQQ), the crypto Fear &amp; Greed Index, and the CBOE VIX — each normalised to a common 0–100 scale before weighting.
- **Global market-breadth panel** spanning 12 instruments across the US, Europe, and Asia (S&amp;P 500, Nasdaq 100, DAX, CAC 40, Nikkei 225, Shanghai Composite, Hang Seng, BIST 100) plus the classic "fear" assets (VIX, US 20Y Treasuries, Gold, US Dollar Index).
- **Quantitative forecast layer** — an ensemble of AR(1), damped-trend regression, and EWMA mean-reversion models projects the index forward and assigns regime probabilities.
- **Feed-health telemetry** — every refresh records per-source status, latency, and fallback usage to `feed_health.csv`, so stale or degraded inputs are visible, not hidden.
- **Graceful degradation** — the pipeline falls back to the last good values on partial outages and never crashes the dashboard on a single failed feed.
- **Fully automated** — a GitHub Actions cron workflow refreshes the data, rebuilds the static `index.html`, and publishes to GitHub Pages on an hourly cadence with retry and concurrency guards.

## 🖼️ Preview
<!-- CODEX: drop product screenshots into docs/ -->
![Cloudy Shiny — main view](docs/screenshot-1.png)
<!-- CODEX: capture the live dashboard at https://monarchcastletech.github.io/Cloudy-Shiny/ — the hero view showing the Cloudy&Shiny Index half-circle gauge, the composite score, and the STORMY/CLOUDY/SHINY regime label. Terminal-style dark theme. -->

![Cloudy Shiny — global breadth &amp; forecast detail](docs/screenshot-2.png)
<!-- CODEX: capture the lower panels — the 12-instrument global market-breadth grid and the quant forecast / regime-probability section. -->

## 🧭 What it does
Cloudy Shiny turns a handful of market feeds into a single, legible sentiment signal and a supporting analytics terminal.

### The Cloudy&Shiny Index
The headline number is a weighted composite computed each refresh:

```
index = (stock × 0.4) + (crypto × 0.3) + ((100 − vix_normalized) × 0.3)
```

- **Stock sentiment (40%)** — derived from SPY (0.6) and QQQ (0.4) via 50-day moving-average deviation, with an RSI(14) overbought/oversold adjustment, clamped to 0–100.
- **Crypto sentiment (30%)** — the alternative.me Crypto Fear &amp; Greed Index, used directly on its native 0–100 scale.
- **Volatility (30%)** — the CBOE VIX, normalised against its trailing one-year range and **inverted**, so calm markets push the index up and spikes pull it down.

The resulting score maps to a regime band: **STORMY** (≤ 20), **CLOUDY** (21–80), **SHINY** (> 80).

### Global market breadth
Beyond the headline index, the dashboard renders a GDP-weighted view of 12 instruments across the US, Europe, and Asia, separating "risk-on" indices from "risk-off" assets (VIX, long Treasuries, Gold, the Dollar Index) to show where global appetite is concentrated.

### Forecast &amp; regime probabilities
A lightweight quant layer projects the index forward using an ensemble of three models — an AR(1) autoregression, a damped local-trend regression, and an EWMA mean-reversion estimate — and converts the projected distribution into STORMY / CLOUDY / SHINY probabilities via a normal-CDF regime model.

## 🗂️ Data &amp; provenance
Per Monarch Castle doctrine — **evidence before assertion**. Cloudy Shiny collects only from open, lawfully accessible market sources and records the trail for every reading.

| Source | Provider | Access | Cadence |
|---|---|---|---|
| Equity prices (SPY, QQQ, global indices) | Yahoo Finance via `yfinance` | Public market data | Hourly |
| Crypto Fear &amp; Greed Index | alternative.me (`api.alternative.me/fng`) | Public JSON API | Hourly |
| Volatility (VIX) | Yahoo Finance via `yfinance` (`^VIX`) | Public market data | Hourly |

- **`sentiment_data.csv`** is the historical store — one timestamped row per refresh carrying the raw stock, crypto, VIX, and normalised VIX inputs alongside the computed `mood_score` and `mood_label`.
- **`feed_health.csv`** is the provenance/audit log — per-source status (`live` / `fallback` / `failed` / `missing`), per-source latency in milliseconds, whether a fallback was used, and any error notes, all UTC-timestamped.
- Network calls use bounded timeouts and retry-with-backoff; on failure the pipeline reuses the last known values and flags the substitution rather than emitting an un-provenanced number.

## 🛠️ Tech stack
- **Language:** Python 3.11
- **Data &amp; analytics:** `pandas`, `yfinance`, `requests`
- **Visualisation:** static HTML/SVG terminal template (`template.html` → `index.html`), with `plotly` available for charting
- **Interactive app:** `streamlit` (`app.py`) for local exploration
- **Automation:** GitHub Actions (scheduled cron) — `.github/workflows/hourly.yml`
- **Hosting:** GitHub Pages (static `index.html`)

## 🚀 Getting started

**Live dashboard:** https://monarchcastletech.github.io/Cloudy-Shiny/

### Run locally
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fetch the latest sentiment reading (writes sentiment_data.csv + feed_health.csv)
python sentiment_tracker.py

# 3a. Build the static dashboard (writes index.html)
python build_index.py
# open index.html in a browser

# 3b. …or run the interactive Streamlit terminal
streamlit run app.py
```
On Windows, `open_system.bat` provides a one-click launch wrapper.

### Automated refresh
`.github/workflows/hourly.yml` runs on a scheduled cron (~50-minute cadence) plus manual `workflow_dispatch`. Each run fetches fresh data, rebuilds `index.html`, and commits the updated `index.html`, `sentiment_data.csv`, and `feed_health.csv` back to the branch — with retry logic and a concurrency guard so runs never collide. GitHub Pages serves the committed `index.html` directly.

## 🧱 Part of Monarch Castle
> A product of **Financial Intelligence** · **Monarch Castle Technologies** — an operating company of **[Monarch Castle Holdings](https://github.com/MonarchCastleHoldings)**.
> Sister companies: [Monarch Castle Technologies](https://github.com/monarchcastletech) · [Strategic Data Company of Ankara](https://github.com/SDCofA)

## 📜 License
See `LICENSE`. © 2026 Monarch Castle Holdings · Ankara, Türkiye.

<div align="center"><sub>🏰 Monarch Castle Holdings — turning open-source noise into lawful, verified, decision-grade intelligence.</sub></div>
