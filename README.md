# Economic Stress Monitor

Streamlit dashboard tracking US / EU macro-financial stress as z-scores
(vs a 5-year rolling window) across vectors: banking, credit, housing,
consumer, labor, liquidity, inflation, early-warning.

## Architecture
- `ingestion/` — pulls raw series from FRED, ECB, Eurostat
- `compute/`   — normalizes to z-scores and builds vector/overall composites
- `storage/`   — DuckDB store (`data/stress.duckdb`)
- `dashboard/` — Streamlit app (`dashboard/app.py`)
- `run.py`     — pipeline runner (`init` / `backfill` / `daily` / `rebuild` / `status`)

## Data freshness (branches)
- **`main`** — source code. The DuckDB file is git-ignored here.
- **`live`** — deploy branch. A daily GitHub Action rebuilds the DB and
  force-publishes `main`'s code + fresh data as a single commit.
  **Deploy the Streamlit app from the `live` branch.**

## Local run
```bash
pip install -r requirements.txt
export FRED_API_KEY=xxxx      # https://fred.stlouisfed.org/docs/api/api_key.html
python run.py backfill        # one-time full history
streamlit run dashboard/app.py
```
