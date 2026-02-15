# CloudyShiny Fear and Greed Upgrade Plan

## Goal
Deliver a production-grade "Monarch Global Mood" module that:
- Uses the required composite formula.
- Preserves the current dashboard baseline look.
- Adds deeper analytics and Bloomberg-terminal-like professionalism.

## Constraints
- Keep existing root files as the primary runtime:
  - `sentiment_tracker.py`
  - `app.py`
  - `template.html`
  - `sentiment_data.csv`
  - `requirements.txt`
- Maintain backward compatibility with existing CSV history.
- Avoid breaking the current visual structure; enhance progressively.

## Formula and Regimes
- Normalize each component to 0-100.
- Composite score:
  - `mood = (stock * 0.4) + (crypto * 0.3) + ((100 - vix_normalized) * 0.3)`
- Regime mapping:
  - `0-20`: STORMY
  - `21-80`: CLOUDY
  - `81-100`: SHINY

## Implementation Phases

### Phase 1: Documentation and Architecture
- Create or refresh:
  - `plan.md`
  - `agents.md`
  - `skills.md`
- Define responsibilities, workflows, and quality gates.

### Phase 2: Data Engine Hardening (`sentiment_tracker.py`)
- Keep source coverage:
  - CNN Fear and Greed
  - alternative.me Crypto Fear and Greed
  - VIX from `yfinance`
- Enforce formula exactly as specified.
- Keep resilient retries and fallback-to-last-row behavior.
- Ensure consistent CSV schema and safe appends.

### Phase 3: Analytics Layer (`app.py`)
- Preserve current layout and style anchors.
- Add richer metrics derived from history:
  - Daily/weekly deltas
  - Component dispersion
  - Short-term volatility
  - Momentum and trend context
  - Risk-on vs risk-off framing
- Prepare additional placeholders for template rendering.

### Phase 4: Professional Terminal UI (`template.html`)
- Keep existing panels and typography language.
- Add Bloomberg-style density and polish:
  - Primary gauge / speedometer emphasis
  - Compact KPI strip
  - Component micro-trends
  - Market regime and diagnostics block
  - Better hierarchy, spacing, and data readability
- Keep responsive behavior for desktop and mobile.

### Phase 5: Validation and Delivery
- Run:
  - `python sentiment_tracker.py`
  - Streamlit syntax smoke check for `app.py`
- Verify:
  - CSV append works
  - Dashboard renders with old and new rows
  - No missing placeholder errors

## Done Criteria
- All required files exist and run.
- Formula is implemented exactly.
- Dashboard is visibly more professional while preserving base look.
- Historical and component views are available for at least 7 days.
- Project can be launched with:
  - `python sentiment_tracker.py`
  - `streamlit run app.py`
