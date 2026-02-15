# CloudyShiny Agents

## Mission
Build and operate the Global Fear and Greed module with a stable data pipeline and a terminal-grade analytics dashboard.

## Scope
- Data ingestion and scoring: `sentiment_tracker.py`
- Visualization and interaction: `app.py`, `template.html`
- History persistence: `sentiment_data.csv`
- Runtime dependencies: `requirements.txt`

## Agent Roles

### 1. Data Agent
- Owns API and market data fetch logic.
- Maintains retries, timeouts, and fallback behavior.
- Guarantees normalized component outputs.

### 2. Quant Agent
- Owns scoring math and interpretation bands.
- Enforces formula consistency:
  - `(stock * 0.4) + (crypto * 0.3) + ((100 - vix_normalized) * 0.3)`
- Adds derived metrics (momentum, volatility, dispersion) without changing base formula.

### 3. UI Agent
- Owns dashboard quality and layout continuity.
- Preserves base visual identity while adding denser professional panels.
- Maintains responsive behavior.

### 4. Reliability Agent
- Validates file integrity and schema compatibility.
- Ensures app runs on stale data and partial data source outages.
- Verifies end-to-end run commands.

## Handoff Contract
- `sentiment_tracker.py` writes a complete row or exits with a clear error.
- `app.py` never crashes on missing optional metrics; it degrades gracefully.
- `template.html` placeholders map 1:1 with `app.py` replacements.

## Quality Gates
- Formula correctness gate.
- CSV schema compatibility gate.
- Render gate (template placeholders resolved).
- Runtime gate:
  - `python sentiment_tracker.py`
  - `streamlit run app.py`

## Delivery Standard
- Production-safe defaults.
- Deterministic calculations.
- Professional information density similar to institutional terminals.
