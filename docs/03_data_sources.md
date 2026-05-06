# 03. Data Sources

The project uses two data layers.

## Real official reference data

`data/reference/nhtsa_utqg_distribution.csv` stores official published NHTSA TireWise / UTQGS aggregate distributions for treadwear, traction and temperature ratings.

`data/reference/real_tyre_data_sources.csv` documents real source URLs and limitations for NHTSA, Data.gov UTQGS and European Commission EPREL tyres.

These records are real reference data, useful for benchmarking rating distributions and explaining official tyre performance frameworks.

## Analytical measurement data

`data/raw/sample_tyre_dataset.csv` is a realistic simulated measurement panel. It supports end-to-end analysis of tread depth, mileage, wear rate, cost, geography, ML and recommendation without licensing barriers.

This distinction matters: public official datasets usually expose tyre label/rating information, while repeated tread-depth observations by mileage and usage profile are normally owned by fleets, garages, insurers or individual users.

Future sources can include:

- NHTSA / UTQG
- EPREL
- allowed tyre review datasets
- weather APIs
- geographic datasets
- owned tread-depth measurements
- tyre images
