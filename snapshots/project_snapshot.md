# TyreWear Intelligence Snapshot

Generated at: `2026-05-06T10:51:42`

## Repository

- Latest commit: `f291126 feat: build professional tyrewear intelligence platform`
- Remote: `origin	https://github.com/JoseCCFerreira/tyrewear-intelligence.git (fetch)
origin	https://github.com/JoseCCFerreira/tyrewear-intelligence.git (push)`

## Files

| File | Size |
|---|---:|
| `index.html` | 12958 bytes |
| `style.css` | 16355 bytes |
| `technical_support.html` | 20737 bytes |
| `README.md` | 7344 bytes |
| `requirements.txt` | 123 bytes |
| `app.py` | 197 bytes |
| `src/tyrewear_app.py` | 10938 bytes |
| `scripts/generate_tyre_data.py` | 9415 bytes |
| `scripts/analyze_tyre_data.py` | 15937 bytes |
| `scripts/setup_sqlite.py` | 917 bytes |
| `scripts/setup_duckdb.py` | 2353 bytes |
| `scripts/run_full_pipeline.py` | 597 bytes |
| `scripts/test_deep_learning.py` | 5077 bytes |
| `scripts/create_snapshot.py` | 3371 bytes |
| `data/raw/sample_tyre_dataset.csv` | 695529 bytes |
| `data/processed/tyrewear.sqlite` | 937984 bytes |
| `data/processed/tyrewear.duckdb` | 4730880 bytes |
| `data/processed/tyrewear_europe_clean.csv` | 825363 bytes |
| `data/processed/tyrewear_europe_monthly_resampled.csv` | 56497 bytes |
| `data/processed/tyrewear_europe_annual_country.csv` | 4885 bytes |
| `data/outputs/hypothesis_tests.csv` | 798 bytes |
| `data/outputs/analysis_summary.json` | 3650 bytes |
| `data/outputs/tyre_clusters.csv` | 450790 bytes |
| `data/outputs/cluster_summary.csv` | 283 bytes |


## Deep Learning Smoke Tests

- pytorch: passed | MSE 0.002565 | version 2.2.2
- tensorflow: passed | MSE 5e-06 | version 2.16.1
