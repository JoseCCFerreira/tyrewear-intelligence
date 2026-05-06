from __future__ import annotations

from pathlib import Path

import duckdb


ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT_DIR / "data" / "raw" / "sample_tyre_dataset.csv"
DB_PATH = ROOT_DIR / "data" / "processed" / "tyrewear.duckdb"


def drop_relation(conn: duckdb.DuckDBPyConnection, name: str) -> None:
    for relation_type in ("VIEW", "TABLE"):
        try:
            conn.execute(f"DROP {relation_type} IF EXISTS {name}")
        except duckdb.CatalogException:
            continue


def main() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(DB_PATH))
    drop_relation(conn, "tyre_measurements")
    conn.execute(
        f"""
        CREATE TABLE tyre_measurements AS
        SELECT * FROM read_csv_auto('{RAW_PATH.as_posix()}', header=true)
        QUALIFY row_number() OVER (PARTITION BY tyre_id ORDER BY measurement_date DESC) = 1
        """
    )
    drop_relation(conn, "mart_tread_depth_analysis")
    conn.execute(
        """
        CREATE VIEW mart_tread_depth_analysis AS
        SELECT
            brand, tyre_size, wheel_position, country, region, season_type,
            COUNT(*) AS tyres,
            AVG(td_current_mm) AS td_avg_mm,
            MIN(td_current_mm) AS td_min_mm,
            AVG(wear_rate_mm_10000km) AS wear_rate_avg,
            AVG(remaining_life_km) AS remaining_life_avg_km,
            AVG(cost_per_1000km) AS cost_per_1000km_avg
        FROM tyre_measurements
        GROUP BY ALL
        """
    )
    drop_relation(conn, "mart_geo_analysis")
    conn.execute(
        """
        CREATE VIEW mart_geo_analysis AS
        SELECT
            country, region, city, AVG(latitude) AS latitude, AVG(longitude) AS longitude,
            COUNT(*) AS tyres,
            AVG(td_current_mm) AS td_avg_mm,
            AVG(wear_rate_mm_10000km) AS wear_rate_avg,
            AVG(CASE WHEN risk_class IN ('high', 'critical') THEN 1 ELSE 0 END) * 100 AS risk_share_pct
        FROM tyre_measurements
        GROUP BY country, region, city
        """
    )
    drop_relation(conn, "mart_ml_features")
    conn.execute(
        """
        CREATE VIEW mart_ml_features AS
        SELECT * FROM tyre_measurements
        """
    )
    conn.close()
    print(f"DuckDB analytical database created: {DB_PATH}")


if __name__ == "__main__":
    main()
