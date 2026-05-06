from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT_DIR / "data" / "raw" / "sample_tyre_dataset.csv"
DB_PATH = ROOT_DIR / "data" / "processed" / "tyrewear.sqlite"


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(RAW_PATH).drop_duplicates("tyre_id")
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql("tyre_measurements", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tyre_country ON tyre_measurements(country)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tyre_brand ON tyre_measurements(brand)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tyre_date ON tyre_measurements(measurement_date)")
    print(f"SQLite database created: {DB_PATH}")


if __name__ == "__main__":
    main()
