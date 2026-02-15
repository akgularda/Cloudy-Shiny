# CloudyShiny Immediate Revision Plan

## Objective
Apply the requested production UI/data revisions without changing the overall terminal-style layout.

## Requested Changes
1. Fix the Sentiment Gauge to a clean geometric half-circle.
2. Replace the `M` badge with the provided MC logo.
3. Add explicit methodology + weights text on-page.
4. Rename score heading to `Cloudy&Shiny Index`.
5. Remove CNN references from the page and data source usage.
6. Replace footer label:
   - From: `CSI-008 | Monarch Castle Technologies`
   - To: `Cloudy&Shiny Index | Monarch Castle Technologies`

## Implementation Plan
1. `template.html`
   - Update header logo markup to image asset.
   - Rebuild gauge arc SVG to true semicircle segmentation.
   - Add methodology formula block and keep weights panel.
   - Rename all visible CNN labels to stock-centric wording.
   - Update footer branding string exactly as requested.
2. `logo.svg`
   - Add a reusable logo asset for both static and Streamlit rendering.
3. `sentiment_tracker.py`
   - Replace external CNN fetch with internal stock sentiment model from market prices.
   - Keep the same composite formula weights: `0.4 / 0.3 / 0.3`.
   - Preserve fallback/retry behavior and CSV compatibility.
4. `build_index.py` and `app.py`
   - Align placeholders and labels (`STOCK_SCORE`, `STOCK_GAP`, `Stock` feed label).
   - Ensure rendered HTML has no CNN wording.
5. Validation
   - `python sentiment_tracker.py`
   - `python build_index.py`
   - Confirm `index.html` contains updated heading, footer, methodology section, and cleaned labels.
