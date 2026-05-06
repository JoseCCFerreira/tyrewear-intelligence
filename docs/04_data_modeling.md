# 04. Data Modeling

The dataset is modeled around one tyre measurement row per tyre.

Important entities:

- tyre catalog: brand, model, tyre size, segment
- vehicle profile: type, weight, drive type, wheel position
- geography: country, region, city, latitude, longitude
- measurements: TD initial, TD current, inner/center/outer TD, mileage
- performance: wear rate, remaining life, risk class, cost

This structure supports analytics, dbt marts and ML feature tables.
