"""
Storage layer — DuckDB.

DuckDB chosen over SQLite because:
  - Columnar, faster for time-series aggregations
  - Native pandas integration (db.df())
  - Single-file, zero-server, same operational simplicity as SQLite
  - Window functions, percentile_cont, regr_slope all built-in

Schema is intentionally narrow. Three tables do all the work:
  - observations:   raw values, one row per (series, date)
  - signals:        normalized stress signals (z-scores, transforms applied)
  - composite:      vector-level and overall stress scores per date
"""

from pathlib import Path
import duckdb

DB_PATH = Path(__file__).parent.parent / "data" / "stress.duckdb"


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS observations (
    series_id   VARCHAR NOT NULL,
    obs_date    DATE    NOT NULL,
    value       DOUBLE,
    source      VARCHAR NOT NULL,
    pulled_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (series_id, obs_date)
);

CREATE INDEX IF NOT EXISTS idx_obs_date ON observations(obs_date);

CREATE TABLE IF NOT EXISTS signals (
    series_id    VARCHAR NOT NULL,
    obs_date     DATE    NOT NULL,
    raw_value    DOUBLE,
    transformed  DOUBLE,    -- after yoy/mom/diff
    z_score      DOUBLE,    -- direction-adjusted, positive = stress
    vector       VARCHAR NOT NULL,
    region       VARCHAR NOT NULL DEFAULT 'US',
    PRIMARY KEY (series_id, obs_date)
);

CREATE INDEX IF NOT EXISTS idx_sig_date ON signals(obs_date);
CREATE INDEX IF NOT EXISTS idx_sig_vector ON signals(vector);

CREATE TABLE IF NOT EXISTS composite (
    obs_date         DATE NOT NULL,
    region           VARCHAR NOT NULL DEFAULT 'US',
    liquidity        DOUBLE,
    credit           DOUBLE,
    labor            DOUBLE,
    consumer         DOUBLE,
    housing          DOUBLE,
    banking          DOUBLE,
    inflation        DOUBLE,
    early_warning    DOUBLE,
    overall          DOUBLE,
    overall_roc_30d  DOUBLE,    -- the rate-of-change leading indicator
    computed_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (obs_date, region)
);

CREATE TABLE IF NOT EXISTS pull_log (
    pulled_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source       VARCHAR,
    series_id    VARCHAR,
    rows_added   INTEGER,
    success      BOOLEAN,
    error_msg    VARCHAR
);
"""


def get_conn(read_only: bool = False) -> duckdb.DuckDBPyConnection:
    """Return a DuckDB connection. Creates the data dir if missing."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DB_PATH), read_only=read_only)
    return conn


def init_db() -> None:
    """Idempotent schema creation."""
    with get_conn() as conn:
        conn.execute(SCHEMA_SQL)
    print(f"DB initialized at {DB_PATH}")


def upsert_observations(rows: list[tuple]) -> int:
    """
    Insert/update observations. rows = [(series_id, obs_date, value, source), ...]
    Returns number of rows written.
    """
    if not rows:
        return 0
    with get_conn() as conn:
        conn.executemany(
            """
            INSERT INTO observations (series_id, obs_date, value, source)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (series_id, obs_date) DO UPDATE SET
                value = EXCLUDED.value,
                pulled_at = now()
            """,
            rows,
        )
    return len(rows)


def log_pull(source: str, series_id: str, rows_added: int,
             success: bool, error_msg: str | None = None) -> None:
    with get_conn() as conn:
        conn.execute(
            """
            INSERT INTO pull_log (source, series_id, rows_added, success, error_msg)
            VALUES (?, ?, ?, ?, ?)
            """,
            (source, series_id, rows_added, success, error_msg),
        )


if __name__ == "__main__":
    init_db()
