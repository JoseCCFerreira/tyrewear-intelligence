# TyreWear Intelligence Snapshot

Generated at: `2026-05-06T21:53:17`

## Repository

- Latest commit: `d8b2ab9 feat: add tyre size intelligence and real reference data`
- Remote: `origin	https://github.com/JoseCCFerreira/tyrewear-intelligence.git (fetch)
origin	https://github.com/JoseCCFerreira/tyrewear-intelligence.git (push)`

## Files

| File | Size |
|---|---:|
| `index.html` | 12958 bytes |
| `style.css` | 16355 bytes |
| `technical_support.html` | 20737 bytes |
| `README.md` | 8473 bytes |
| `requirements.txt` | 123 bytes |
| `app.py` | 197 bytes |
| `src/tyrewear_app.py` | 17537 bytes |
| `scripts/generate_tyre_data.py` | 9415 bytes |
| `scripts/fetch_real_tyre_reference.py` | 3581 bytes |
| `scripts/analyze_tyre_data.py` | 18652 bytes |
| `scripts/setup_sqlite.py` | 917 bytes |
| `scripts/setup_duckdb.py` | 3013 bytes |
| `scripts/run_full_pipeline.py` | 637 bytes |
| `scripts/test_deep_learning.py` | 5077 bytes |
| `scripts/create_snapshot.py` | 3631 bytes |
| `data/raw/sample_tyre_dataset.csv` | 695529 bytes |
| `data/reference/nhtsa_utqg_distribution.csv` | 2136 bytes |
| `data/reference/real_tyre_data_sources.csv` | 1054 bytes |
| `data/processed/tyrewear.sqlite` | 937984 bytes |
| `data/processed/tyrewear.duckdb` | 5779456 bytes |
| `data/processed/tyrewear_europe_clean.csv` | 825363 bytes |
| `data/processed/tyrewear_europe_monthly_resampled.csv` | 56497 bytes |
| `data/processed/tyrewear_europe_annual_country.csv` | 4885 bytes |
| `data/outputs/hypothesis_tests.csv` | 942 bytes |
| `data/outputs/analysis_summary.json` | 4496 bytes |
| `data/outputs/tyre_clusters.csv` | 476619 bytes |
| `data/outputs/cluster_summary.csv` | 278 bytes |
| `data/outputs/dimension_performance.csv` | 670 bytes |
| `data/outputs/dimension_monthly_patterns.csv` | 18763 bytes |


## Deep Learning Smoke Tests

- pytorch: passed | MSE 0.002565 | version 2.2.2
- tensorflow: passed | MSE 5e-06 | version 2.16.1
