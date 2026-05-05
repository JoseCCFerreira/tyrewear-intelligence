from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
RAW_PATH = DATA_DIR / "raw" / "tyrewear_europe_raw.csv"
CLEAN_PATH = DATA_DIR / "processed" / "tyrewear_europe_clean.csv"
MONTHLY_PATH = DATA_DIR / "processed" / "tyrewear_europe_monthly_resampled.csv"
ANNUAL_PATH = DATA_DIR / "processed" / "tyrewear_europe_annual_country.csv"
OUTPUT_DIR = DATA_DIR / "outputs"
HYPOTHESIS_PATH = OUTPUT_DIR / "hypothesis_tests.csv"
SUMMARY_PATH = OUTPUT_DIR / "analysis_summary.json"
CLUSTER_PATH = OUTPUT_DIR / "tyre_clusters.csv"
CLUSTER_SUMMARY_PATH = OUTPUT_DIR / "cluster_summary.csv"
CORRELATION_PATH = OUTPUT_DIR / "correlation_matrix.csv"
COUNTRY_PATTERNS_PATH = OUTPUT_DIR / "country_risk_patterns.csv"
BRAND_PATTERNS_PATH = OUTPUT_DIR / "brand_value_patterns.csv"
SNAPSHOT_DIR = ROOT_DIR / "snapshots"
DL_RESULTS_PATH = SNAPSHOT_DIR / "deep_learning_test_results.json"
PROJECT_SNAPSHOT_PATH = SNAPSHOT_DIR / "project_snapshot.json"


st.set_page_config(page_title="TyreWear Intelligence", page_icon="TW", layout="wide")


@st.cache_data
def load_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, **kwargs)


@st.cache_data
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


raw = load_csv(RAW_PATH)
clean = load_csv(CLEAN_PATH, parse_dates=["observed_at"])
monthly = load_csv(MONTHLY_PATH, parse_dates=["month"])
annual = load_csv(ANNUAL_PATH)
hypothesis = load_csv(HYPOTHESIS_PATH)
clusters = load_csv(CLUSTER_PATH)
cluster_summary = load_csv(CLUSTER_SUMMARY_PATH)
correlation = load_csv(CORRELATION_PATH, index_col=0)
country_patterns = load_csv(COUNTRY_PATTERNS_PATH)
brand_patterns = load_csv(BRAND_PATTERNS_PATH)
summary = load_json(SUMMARY_PATH)
deep_learning = load_json(DL_RESULTS_PATH)
snapshot = load_json(PROJECT_SNAPSHOT_PATH)


st.title("TyreWear Intelligence")
st.caption("Raw data, cleaning, resampling, hypothesis testing, machine learning, clustering and deep learning results.")

if clean.empty:
    st.error("No processed data found. Run `python3 scripts/generate_tyre_data.py && python3 scripts/analyze_tyre_data.py`.")
    st.stop()

countries = sorted(clean["country"].unique())
regions = sorted(clean["europe_region"].unique())
selected_countries = st.sidebar.multiselect("Countries", countries, default=countries)
selected_regions = st.sidebar.multiselect("European regions", regions, default=regions)
selected_years = st.sidebar.slider("Year range", int(clean["year"].min()), int(clean["year"].max()), (int(clean["year"].min()), int(clean["year"].max())))

filtered = clean[
    clean["country"].isin(selected_countries)
    & clean["europe_region"].isin(selected_regions)
    & clean["year"].between(selected_years[0], selected_years[1])
].copy()
filtered_monthly = monthly[
    monthly["country"].isin(selected_countries)
    & monthly["europe_region"].isin(selected_regions)
    & monthly["month"].dt.year.between(selected_years[0], selected_years[1])
].copy()
filtered_annual = annual[
    annual["country"].isin(selected_countries)
    & annual["europe_region"].isin(selected_regions)
    & annual["year"].between(selected_years[0], selected_years[1])
].copy()

metric_cols = st.columns(6)
metric_cols[0].metric("Raw rows", f"{len(raw):,}")
metric_cols[1].metric("Clean rows", f"{len(clean):,}")
metric_cols[2].metric("Filtered rows", f"{len(filtered):,}")
metric_cols[3].metric("Countries", filtered["country"].nunique())
metric_cols[4].metric("Avg wear rate", round(filtered["wear_rate_mm_per_1000km"].mean(), 4))
metric_cols[5].metric("High risk share", f"{round(filtered['high_wear_risk'].mean() * 100, 2)}%")

tabs = st.tabs(
    [
        "Raw & Cleaning",
        "Distributions",
        "Europe Over Time",
        "Hypothesis Tests",
        "Machine Learning",
        "Clusters",
        "PyTorch & TensorFlow",
        "Outputs",
    ]
)

with tabs[0]:
    st.subheader("Raw data and cleaning")
    cleaning = summary.get("cleaning", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Duplicates removed", cleaning.get("duplicates_removed", 0))
    c2.metric("Countries", cleaning.get("countries", 0))
    c3.metric("Regions", cleaning.get("regions", 0))
    c4.metric("Window", f"{cleaning.get('date_min')} → {cleaning.get('date_max')}")
    st.write("Raw sample with intentionally dirty values: missing metrics, duplicated rows and invalid prices.")
    st.dataframe(raw.head(250), width="stretch")
    missing = pd.Series(cleaning.get("missing_values_before_imputation", {}), name="missing").reset_index()
    missing.columns = ["column", "missing_before_imputation"]
    if not missing.empty:
        st.plotly_chart(px.bar(missing, x="column", y="missing_before_imputation", title="Missing values before imputation"), width="stretch")
    st.write("Cleaned data sample after deduplication, type normalization, invalid-value handling and median imputation.")
    st.dataframe(filtered.head(250), width="stretch")

with tabs[1]:
    st.subheader("Distribution analysis")
    d1, d2 = st.columns(2)
    with d1:
        st.plotly_chart(px.histogram(filtered, x="wear_rate_mm_per_1000km", color="europe_region", nbins=45, marginal="box", title="Wear-rate distribution by European region"), width="stretch")
        st.plotly_chart(px.box(filtered, x="europe_region", y="tread_depth_mm", color="europe_region", title="Tread depth by region"), width="stretch")
    with d2:
        st.plotly_chart(px.histogram(filtered, x="projected_life_km", color="tyre_season", nbins=45, marginal="box", title="Projected tyre life distribution"), width="stretch")
        st.plotly_chart(px.scatter(filtered.sample(min(3000, len(filtered)), random_state=42), x="price_eur", y="projected_life_km", color="brand", size="wet_grip_score", hover_data=["country", "dimension"], title="Price vs projected life"), width="stretch")
    if not correlation.empty:
        st.plotly_chart(px.imshow(correlation, text_auto=True, aspect="auto", title="Correlation matrix"), width="stretch")

with tabs[2]:
    st.subheader("Countries and European regions over time")
    st.plotly_chart(px.line(filtered_monthly, x="month", y="avg_wear_rate", color="country", title="Monthly average wear rate by country"), width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        region_year = filtered_annual.groupby(["year", "europe_region"], as_index=False).agg(avg_wear_rate=("avg_wear_rate", "mean"), high_risk_share=("high_risk_share", "mean"), avg_cost_per_km=("avg_cost_per_km", "mean"))
        st.plotly_chart(px.line(region_year, x="year", y="high_risk_share", color="europe_region", markers=True, title="High-risk share by region over time"), width="stretch")
    with c2:
        st.plotly_chart(px.bar(filtered_annual, x="country", y="avg_cost_per_km", color="europe_region", animation_frame="year", title="Cost per km by country and year"), width="stretch")
    if not country_patterns.empty:
        st.write("Country-level risk patterns")
        st.dataframe(country_patterns, width="stretch", hide_index=True)

with tabs[3]:
    st.subheader("Hypothesis tests")
    st.write("These tests validate whether differences in wear rate or projected life are statistically supported.")
    st.dataframe(hypothesis, width="stretch", hide_index=True)
    if not hypothesis.empty:
        hyp_chart = hypothesis.copy()
        hyp_chart["minus_log10_p"] = -hyp_chart["p_value"].clip(lower=1e-300).apply(lambda x: __import__("math").log10(x))
        st.plotly_chart(px.bar(hyp_chart, x="test", y="minus_log10_p", color="decision", hover_data=["question", "p_value"], title="-log10(p-value) by hypothesis test"), width="stretch")
    st.info("Rule of thumb: p-value below 0.05 suggests a statistically significant difference, but interpretation still needs effect size and business context.")

with tabs[4]:
    st.subheader("Machine learning: regression and classification")
    ml = summary.get("machine_learning", {})
    classification = ml.get("classification_high_wear_risk", {})
    regression = ml.get("regression_projected_life_km", {})
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", classification.get("accuracy"))
    m2.metric("F1", classification.get("f1"))
    m3.metric("ROC AUC", classification.get("roc_auc"))
    m4.metric("MAE life km", regression.get("mae"))
    m5.metric("R2 life km", regression.get("r2"))
    if not brand_patterns.empty:
        st.plotly_chart(px.bar(brand_patterns, x="brand", y="avg_projected_life_km", color="avg_cost_per_km", title="Brand value: projected life vs cost per km"), width="stretch")
        st.dataframe(brand_patterns, width="stretch", hide_index=True)
    patterns = summary.get("patterns", {})
    st.write("Strongest correlations with wear rate")
    st.json(patterns.get("strongest_correlations_with_wear_rate", {}))

with tabs[5]:
    st.subheader("Clustering and pattern discovery")
    if not cluster_summary.empty:
        st.dataframe(cluster_summary, width="stretch", hide_index=True)
        st.plotly_chart(px.scatter(cluster_summary, x="avg_wear_rate", y="avg_projected_life_km", size="observations", color="high_risk_share", text="cluster", title="Cluster profiles"), width="stretch")
    if not clusters.empty:
        plot_clusters = clusters.sample(min(4500, len(clusters)), random_state=42)
        st.plotly_chart(px.scatter(plot_clusters, x="wear_rate_mm_per_1000km", y="projected_life_km", color="cluster", hover_data=["country", "brand", "tyre_season"], title="Observation clusters: wear rate vs projected life"), width="stretch")

with tabs[6]:
    st.subheader("PyTorch and TensorFlow application")
    dl_tests = deep_learning.get("tests", [])
    dl_dataset = deep_learning.get("dataset", {})
    d1, d2, d3 = st.columns(3)
    d1.metric("DL dataset rows", f"{dl_dataset.get('rows', 0):,}")
    d2.metric("DL features", dl_dataset.get("features", 0))
    d3.metric("Target", dl_dataset.get("target", "n/a"))
    if dl_tests:
        dl_df = pd.DataFrame(dl_tests)
        st.dataframe(dl_df, width="stretch", hide_index=True)
        st.plotly_chart(px.bar(dl_df, x="framework", y="final_mse", color="status", title="Deep learning MSE: lower is better"), width="stretch")
    st.write("The deep learning script trains compact neural networks to predict standardized projected tyre life from cleaned tyre features.")

with tabs[7]:
    st.subheader("Generated outputs")
    output_files = [
        RAW_PATH,
        CLEAN_PATH,
        MONTHLY_PATH,
        ANNUAL_PATH,
        HYPOTHESIS_PATH,
        SUMMARY_PATH,
        CLUSTER_PATH,
        CLUSTER_SUMMARY_PATH,
        COUNTRY_PATTERNS_PATH,
        BRAND_PATTERNS_PATH,
        DL_RESULTS_PATH,
        PROJECT_SNAPSHOT_PATH,
    ]
    output_df = pd.DataFrame(
        [{"path": str(path.relative_to(ROOT_DIR)), "exists": path.exists(), "size_kb": round(path.stat().st_size / 1024, 2) if path.exists() else 0} for path in output_files]
    )
    st.dataframe(output_df, width="stretch", hide_index=True)
    st.link_button("Open landing page", "http://localhost:8570/index.html")
    st.link_button("Open technical support", "http://localhost:8570/technical_support.html")
