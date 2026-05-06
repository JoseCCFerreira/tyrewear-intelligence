# 05. SQLite, DuckDB and dbt

SQLite stores a transactional version of the tyre records.

DuckDB stores analytical tables and views for fast local querying.

dbt documents and transforms the data:

- staging: clean source-level models
- intermediate: calculated features
- marts: dashboard and ML-ready outputs
- tests: not null, unique, accepted values and valid ranges

Run:

```bash
python3 scripts/setup_sqlite.py
python3 scripts/setup_duckdb.py
```

Run dbt in a separate environment from TensorFlow because the two toolchains currently require different `protobuf` versions:

```bash
python3 -m venv .venv-dbt
source .venv-dbt/bin/activate
pip install -r requirements-dbt.txt
cd dbt_tyre_warehouse
dbt seed --profiles-dir .
dbt build --profiles-dir .
dbt docs generate --profiles-dir .
```
