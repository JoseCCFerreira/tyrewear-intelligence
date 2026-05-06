from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_PATH = DATA_DIR / "raw" / "sample_tyre_dataset.csv"
CLEAN_PATH = DATA_DIR / "processed" / "tyrewear_europe_clean.csv"
MONTHLY_PATH = DATA_DIR / "processed" / "tyrewear_europe_monthly_resampled.csv"
ANNUAL_PATH = DATA_DIR / "processed" / "tyrewear_europe_annual_country.csv"
OUTPUT_DIR = DATA_DIR / "outputs"


@st.cache_data
def csv(path: Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(path, **kwargs) if path.exists() else pd.DataFrame()


@st.cache_data
def js(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def load_data() -> dict:
    clean = csv(CLEAN_PATH, parse_dates=["measurement_date"])
    raw = csv(RAW_PATH)
    return {
        "raw": raw,
        "clean": clean,
        "monthly": csv(MONTHLY_PATH, parse_dates=["month"]),
        "annual": csv(ANNUAL_PATH),
        "tests": csv(OUTPUT_DIR / "hypothesis_tests.csv"),
        "corr": csv(OUTPUT_DIR / "correlation_matrix.csv", index_col=0),
        "clusters": csv(OUTPUT_DIR / "tyre_clusters.csv"),
        "cluster_summary": csv(OUTPUT_DIR / "cluster_summary.csv"),
        "regression": csv(OUTPUT_DIR / "ml_regression_metrics.csv"),
        "classification": csv(OUTPUT_DIR / "ml_classification_metrics.csv"),
        "recommendations": csv(OUTPUT_DIR / "recommendation_scores.csv"),
        "country": csv(OUTPUT_DIR / "country_risk_patterns.csv"),
        "brand": csv(OUTPUT_DIR / "brand_value_patterns.csv"),
        "summary": js(OUTPUT_DIR / "analysis_summary.json"),
        "dl": js(ROOT_DIR / "snapshots" / "deep_learning_test_results.json"),
    }


def filtered(clean: pd.DataFrame) -> pd.DataFrame:
    countries = sorted(clean["country"].unique()) if not clean.empty else []
    regions = sorted(clean["region"].unique()) if not clean.empty else []
    sel_countries = st.sidebar.multiselect("Countries", countries, default=countries)
    sel_regions = st.sidebar.multiselect("Regions", regions, default=regions)
    brands = sorted(clean["brand"].unique()) if not clean.empty else []
    sel_brands = st.sidebar.multiselect("Brands", brands, default=brands)
    if clean.empty:
        return clean
    return clean[clean["country"].isin(sel_countries) & clean["region"].isin(sel_regions) & clean["brand"].isin(sel_brands)]


def overview() -> None:
    data = load_data()
    df = filtered(data["clean"])
    st.title("TyreWear Intelligence")
    st.caption("Predictive tyre analytics for safety, durability and cost optimization.")
    c = st.columns(6)
    c[0].metric("Tyres", f"{len(df):,}")
    c[1].metric("Avg TD", round(df["td_current_mm"].mean(), 2))
    c[2].metric("Min TD", round(df["td_current_mm"].min(), 2))
    c[3].metric("Wear rate", round(df["wear_rate_mm_10000km"].mean(), 3))
    c[4].metric("Remaining life", f"{int(df['remaining_life_km'].mean()):,} km")
    c[5].metric("Risk share", f"{round(df['risk_binary'].mean()*100, 1)}%")
    st.plotly_chart(px.bar(df.groupby("brand", as_index=False).agg(td_avg=("td_current_mm", "mean"), wear=("wear_rate_mm_10000km", "mean")), x="brand", y="td_avg", color="wear", title="Brand ranking: tread depth and wear rate"), width="stretch")
    st.plotly_chart(px.histogram(df, x="td_current_mm", color="risk_class", nbins=40, marginal="box", title="Tread depth distribution"), width="stretch")


def data_explorer() -> None:
    data = load_data()
    df = filtered(data["clean"])
    st.title("Data Explorer")
    uploaded = st.file_uploader("Upload CSV for temporary exploration", type=["csv"])
    if uploaded:
        df = pd.read_csv(uploaded)
    st.dataframe(df, width="stretch")
    st.download_button("Download filtered data", df.to_csv(index=False), "filtered_tyre_data.csv")


def tread_depth() -> None:
    df = filtered(load_data()["clean"])
    st.title("Tread Depth Analysis")
    st.plotly_chart(px.box(df, x="brand", y="td_current_mm", color="brand", title="TD by brand"), width="stretch")
    st.plotly_chart(px.box(df, x="tyre_size", y="wear_rate_mm_10000km", color="wheel_position", title="Wear rate by tyre size and wheel position"), width="stretch")
    sample = df.sample(min(4000, len(df)), random_state=42)
    st.plotly_chart(px.scatter(sample, x="mileage_km", y="td_current_mm", color="risk_class", trendline="ols", title="TD vs mileage"), width="stretch")


def statistical_tests() -> None:
    tests = load_data()["tests"]
    st.title("Statistical Tests")
    st.dataframe(tests, width="stretch", hide_index=True)
    if not tests.empty:
        tests = tests.copy()
        tests["minus_log10_p"] = -tests["p_value"].clip(lower=1e-300).apply(lambda x: __import__("math").log10(x))
        st.plotly_chart(px.bar(tests, x="test", y="minus_log10_p", color="decision", hover_data=["question", "p_value"], title="Hypothesis-test strength"), width="stretch")
    st.info("Automatic interpretation: p-value < 0.05 indicates evidence against the null hypothesis, but business effect size still matters.")


def correlation() -> None:
    corr = load_data()["corr"]
    st.title("Correlation Analysis")
    st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto", title="Pearson correlation heatmap"), width="stretch")


def clustering() -> None:
    data = load_data()
    st.title("Clustering")
    st.dataframe(data["cluster_summary"], width="stretch", hide_index=True)
    clusters = data["clusters"]
    st.plotly_chart(px.scatter(clusters, x="pca_1", y="pca_2", color="kmeans_cluster", hover_data=["brand", "country", "risk_class"], title="PCA projection of K-Means clusters"), width="stretch")
    st.plotly_chart(px.scatter(clusters, x="wear_rate_mm_10000km", y="remaining_life_km", color="dbscan_cluster", title="DBSCAN pattern view"), width="stretch")


def machine_learning() -> None:
    data = load_data()
    st.title("Machine Learning")
    st.subheader("Regression: remaining life")
    st.dataframe(data["regression"], width="stretch", hide_index=True)
    st.subheader("Classification: high/critical risk")
    st.dataframe(data["classification"], width="stretch", hide_index=True)
    st.plotly_chart(px.bar(data["regression"], x="model", y="r2", title="Regression model comparison"), width="stretch")
    st.plotly_chart(px.bar(data["classification"], x="model", y="f1", title="Classification model comparison"), width="stretch")


def deep_learning() -> None:
    dl = load_data()["dl"]
    st.title("Deep Learning")
    st.write("TensorFlow/PyTorch are used here only where they add value: neural tabular comparison today, image CNNs and embeddings later.")
    st.json(dl.get("dataset", {}))
    df = pd.DataFrame(dl.get("tests", []))
    if not df.empty:
        st.dataframe(df, width="stretch", hide_index=True)
        st.plotly_chart(px.bar(df, x="framework", y="final_mse", color="status", title="PyTorch vs TensorFlow MSE"), width="stretch")


def geo_map() -> None:
    df = filtered(load_data()["clean"])
    st.title("Geo Map")
    geo = df.groupby(["country", "region", "city"], as_index=False).agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"), td_avg=("td_current_mm", "mean"), wear_rate=("wear_rate_mm_10000km", "mean"), risk_share=("risk_binary", lambda s: s.mean() * 100), tyres=("tyre_id", "count"))
    st.plotly_chart(px.scatter_mapbox(geo, lat="latitude", lon="longitude", size="tyres", color="risk_share", hover_name="country", hover_data=["region", "td_avg", "wear_rate"], zoom=3, height=620, title="Geographic tyre risk and wear patterns", mapbox_style="open-street-map"), width="stretch")


def recommendation() -> None:
    recs = load_data()["recommendations"]
    st.title("Tyre Recommendation")
    country = st.selectbox("Country/region", sorted(recs["country"].unique()))
    vehicle = st.selectbox("Vehicle type", sorted(recs["vehicle_type"].unique()))
    budget = st.slider("Max price (€)", 50, 260, 180)
    priority = st.selectbox("Priority", ["balanced", "durability", "safety", "price", "comfort", "efficiency"])
    filtered_recs = recs[(recs["country"] == country) & (recs["vehicle_type"] == vehicle) & (recs["price_eur"] <= budget)].copy()
    if priority == "durability":
        filtered_recs = filtered_recs.sort_values("remaining_life_km", ascending=False)
    elif priority == "price":
        filtered_recs = filtered_recs.sort_values("cost_per_1000km")
    else:
        filtered_recs = filtered_recs.sort_values("recommended_tyre_score", ascending=False)
    st.dataframe(filtered_recs[["tyre_id", "brand", "model", "tyre_size", "remaining_life_km", "cost_per_1000km", "risk_class", "recommended_tyre_score"]].head(20), width="stretch", hide_index=True)


def ai_advisor() -> None:
    data = load_data()
    st.title("AI Advisor")
    q = st.text_input("Ask a question about tyres, risk, clusters or recommendations")
    if q:
        best = data["recommendations"].sort_values("recommended_tyre_score", ascending=False).iloc[0]
        risk = data["summary"]["patterns"]["highest_risk_region"]
        st.write(f"Based on current outputs, the strongest general recommendation is **{best['brand']} {best['model']}** in size **{best['tyre_size']}**.")
        st.write(f"The highest risk pattern is currently in **{risk['country']} / {risk['region']}**, with risk share around **{round(risk['risk_share_pct'], 2)}%**.")
        st.caption("MVP advisor: rule/template based. Future version can use retrieval over outputs and documentation.")


def role_skills() -> None:
    st.title("Role-Based Skills Map")
    roles = {
        "Data Analyst": ["KPIs", "Dashboards", "Filters", "Maps", "Business storytelling"],
        "Analytics Engineer": ["SQLite/DuckDB", "dbt staging/intermediate/marts", "Tests", "Data lineage"],
        "Data Scientist": ["Hypothesis tests", "Correlation", "Regression", "Clustering", "Model evaluation"],
        "Data Engineer": ["Ingestion", "Pipelines", "Data quality", "Raw/processed/curated layers"],
        "ML Engineer": ["Training scripts", "Model comparison", "Deep learning examples", "Inference integration"],
    }
    for role, skills in roles.items():
        with st.expander(role, expanded=True):
            st.write(", ".join(skills))


def methodology() -> None:
    st.title("Methodology")
    st.markdown(
        """
        **Wear:** `wear_mm = td_initial_mm - td_current_mm`

        **Wear rate:** `wear_rate_mm_10000km = wear_mm / mileage_km * 10000`

        **Remaining life:** `remaining_life_km = max(td_current_mm - 1.6, 0) / wear_rate * 10000`

        **Recommendation score:** weighted combination of durability, safety, cost, comfort and efficiency.

        Deep learning is intentionally a later-stage layer: useful for images, embeddings and large repeated-measurement data, not for every dashboard problem.
        """
    )
