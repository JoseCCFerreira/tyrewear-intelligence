# TyreWear Intelligence

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red)
![DuckDB](https://img.shields.io/badge/DuckDB-analytics-yellow)
![dbt](https://img.shields.io/badge/dbt-modeling-orange)
![ML](https://img.shields.io/badge/Machine%20Learning-scikit--learn-green)

**Predictive tyre analytics for safety, durability and cost optimization.**

TyreWear Intelligence is a professional portfolio project for tyre analysis. It studies tread depth, wear rate, remaining life, cost per km, safety, energy efficiency, noise, geography and user profile to support better tyre decisions.

## Business Problem

Buying tyres only by price is incomplete. A cheaper tyre can wear faster, increase risk, create higher cost per km and reduce safety. This project compares tyres using statistical evidence, machine learning and interactive decision support.

## Solution Overview

The platform provides a realistic end-to-end MVP:

- realistic tyre dataset
- SQLite transactional layer
- DuckDB analytical layer
- dbt staging/intermediate/marts
- Streamlit multipage app
- descriptive statistics and p-value tests
- correlation analysis
- K-Means, DBSCAN and GMM clustering
- ML regression and classification
- TensorFlow/PyTorch examples used with clear purpose
- geographic map
- personalized weighted recommender
- AI advisor MVP

## Architecture

```text
Public / Simulated Data Sources
        ↓
SQLite transactional database
        ↓
DuckDB analytical database
        ↓
dbt staging → intermediate → marts
        ↓
Statistics → Clustering → Machine Learning
        ↓
Streamlit App → AI Advisor
```

## Role-Based Project Breakdown

### Data Analyst Skills

- KPIs, dashboards, filters, maps and visual storytelling
- Tread depth analysis by brand, tyre size, wheel position and region
- Business metrics: average TD, minimum TD, wear rate, remaining life, cost per 1000 km and risk share
- Tools: SQL, pandas, Plotly, Streamlit, DuckDB, CSV

### Analytics Engineer Skills

- dbt staging, intermediate and marts
- data lineage, metric definitions and model documentation
- tests for accepted values, not-null, uniqueness and valid ranges
- Tools: SQL, dbt, DuckDB, SQLite, YAML

### Data Scientist Skills

- hypothesis tests, p-values, normality, Levene, t-tests, ANOVA, Mann-Whitney, Kruskal-Wallis and chi-square
- correlation analysis using Pearson and Spearman
- clustering using K-Means, DBSCAN, GMM and PCA
- predictive modelling and model evaluation
- Tools: Python, pandas, numpy, scipy, scikit-learn, Plotly

### Data Engineer Skills

- local ingestion pipeline
- raw, processed and output data layers
- SQLite and DuckDB creation scripts
- data quality checks and reproducible orchestration
- Tools: Python, pathlib, pandas, SQLite, DuckDB, logging-ready scripts

### Machine Learning Engineer Skills

- scikit-learn pipelines
- model comparison
- local model-output generation
- TensorFlow tabular regression
- PyTorch tabular regression/classification smoke tests
- Streamlit integration for outputs and inference-oriented recommendations

## Features

- Overview KPIs
- Data Explorer with CSV upload and download
- Tread Depth Analysis
- Statistical Tests
- Correlation Analysis
- Clustering
- Machine Learning
- Deep Learning
- Geo Map
- Tyre Recommendation
- AI Advisor
- Role-Based Skills Map
- Methodology

## Data Sources

The MVP uses a realistic simulated dataset. Future sources can include:

- NHTSA / UTQG
- EPREL
- public tyre tests when licensing permits
- weather data
- geographic data
- owned tread-depth measurements
- tyre images

## Data Model

Main dataset columns include tyre identity, brand, model, tyre size, TD measurements, wear, cost, risk, vehicle profile, geography and coordinates.

The generated file is:

```text
data/raw/sample_tyre_dataset.csv
```

## dbt Workflow

Use a separate dbt environment because current dbt releases require `protobuf>=6`, while TensorFlow 2.16 requires `protobuf<5`.

```bash
python3 -m venv .venv-dbt
source .venv-dbt/bin/activate
pip install -r ../requirements-dbt.txt
cd dbt_tyre_warehouse
dbt seed --profiles-dir .
dbt build --profiles-dir .
dbt docs generate --profiles-dir .
deactivate
```

The dbt project contains:

- staging models
- intermediate models
- marts
- schema tests
- range validation test

## Statistical Methodology

The project includes:

- Shapiro-Wilk
- Kolmogorov-Smirnov
- Levene
- t-test
- Welch t-test
- ANOVA
- Mann-Whitney
- Kruskal-Wallis
- Chi-square
- Pearson
- Spearman

## Clustering Methodology

Implemented:

- K-Means
- DBSCAN
- Gaussian Mixture Models
- PCA projection
- Silhouette Score
- Davies-Bouldin Score

Roadmap:

- UMAP
- t-SNE
- hierarchical clustering

## Machine Learning Methodology

Targets:

- `td_future_mm`
- `wear_rate_mm_10000km`
- `remaining_life_km`
- `risk_class`
- `cost_per_1000km`
- `recommended_tyre_score`

Implemented MVP:

- Ridge regression
- Random Forest regression
- Gradient Boosting regression
- Logistic Regression classification
- Random Forest classification

## TensorFlow / PyTorch Usage

TensorFlow and PyTorch are not used just for fashion. They are included where they can add value:

- tabular neural-network comparison
- future image classification of tyre wear
- future time series for repeated TD measurements
- future embeddings for personalized recommendations

For small tabular datasets, scikit-learn models remain the MVP baseline.

## Geographic Analysis

The Streamlit app includes a map of European country/city observations with:

- average TD
- wear rate
- risk share
- number of tyres
- filters by country, region and brand

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For dbt, use `requirements-dbt.txt` in a separate virtual environment.

## How To Run

```bash
python3 scripts/run_full_pipeline.py
streamlit run app.py --server.port 8580
python3 -m http.server 8570
```

Open:

```text
http://localhost:8580
http://localhost:8570/docs_html/index.html
```

## Repository Structure

```text
app.py
pages/
src/
scripts/
data/raw/
data/processed/
data/outputs/
dbt_tyre_warehouse/
docs/
docs_html/
snapshots/
```

## Roadmap

### Phase 1 — MVP Analytics

Dataset, overview, filters, KPIs, TD analysis, map, README and HTML page.

### Phase 2 — Analytics Engineering

SQLite, DuckDB, dbt models, dbt tests, dbt docs and data dictionary.

### Phase 3 — Statistical & Clustering Layer

Automatic tests, correlations, K-Means, DBSCAN, PCA/UMAP and interpretation.

### Phase 4 — Machine Learning

Wear-rate prediction, remaining-life prediction, risk classification and feature importance.

### Phase 5 — Deep Learning

TensorFlow tabular regression, PyTorch classification, CNN for images and embeddings.

### Phase 6 — Productization

API, Docker, CI/CD, MLflow, deployment and monitoring.

## Limitations

- The dataset is simulated but realistic.
- Deep learning examples are educational MVP examples.
- Image classification is documented as a future phase.
- dbt requires a local dbt-duckdb installation to run.

## Author

Created by Jose Carlos Ferreira as a professional portfolio project covering Data Analyst, Analytics Engineer, Data Scientist, Data Engineer and Machine Learning Engineer skills.
