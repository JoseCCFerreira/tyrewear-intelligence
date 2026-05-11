from __future__ import annotations

import json
from html import escape
from pathlib import Path

import numpy as np
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
REFERENCE_DIR = DATA_DIR / "reference"


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
        "dimension": csv(OUTPUT_DIR / "dimension_performance.csv"),
        "dimension_monthly": csv(OUTPUT_DIR / "dimension_monthly_patterns.csv", parse_dates=["month"]),
        "utqg": csv(REFERENCE_DIR / "nhtsa_utqg_distribution.csv"),
        "real_sources": csv(REFERENCE_DIR / "real_tyre_data_sources.csv"),
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
    sizes = sorted(clean["tyre_size"].unique()) if not clean.empty else []
    default_sizes = sizes[:]
    sel_sizes = st.sidebar.multiselect("Tyre sizes", sizes, default=default_sizes)
    seasons = sorted(clean["season_type"].unique()) if not clean.empty else []
    sel_seasons = st.sidebar.multiselect("Season type", seasons, default=seasons)
    if clean.empty:
        return clean
    return clean[
        clean["country"].isin(sel_countries)
        & clean["region"].isin(sel_regions)
        & clean["brand"].isin(sel_brands)
        & clean["tyre_size"].isin(sel_sizes)
        & clean["season_type"].isin(sel_seasons)
    ]


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
    st.plotly_chart(
        px.scatter(
            df,
            x="rim_inch",
            y="wear_rate_mm_10000km",
            size="width_mm",
            color="risk_class",
            hover_data=["brand", "model", "tyre_size", "aspect_ratio", "remaining_life_km"],
            title="Dimension effect: rim, width and wear rate",
        ),
        width="stretch",
    )
    sample = df.sample(min(4000, len(df)), random_state=42)
    fig = px.scatter(sample, x="mileage_km", y="td_current_mm", color="risk_class", title="TD vs mileage")
    regression_data = sample[["mileage_km", "td_current_mm"]].dropna().sort_values("mileage_km")
    if len(regression_data) > 1:
        slope, intercept = np.polyfit(regression_data["mileage_km"], regression_data["td_current_mm"], 1)
        fig.add_scatter(
            x=regression_data["mileage_km"],
            y=slope * regression_data["mileage_km"] + intercept,
            mode="lines",
            name="Linear trend",
            line={"color": "#111827", "width": 3},
        )
    st.plotly_chart(fig, width="stretch")


def tyre_size_intelligence() -> None:
    data = load_data()
    df = filtered(data["clean"])
    dimension = data["dimension"]
    st.title("Tyre Size Intelligence")
    st.caption("Dimension-aware analytics: size, width, aspect ratio and rim are treated as explicit analytical and ML features.")
    if df.empty:
        st.warning("No data available for the selected filters.")
        return

    kpis = st.columns(5)
    kpis[0].metric("Selected records", f"{len(df):,}")
    kpis[1].metric("Unique sizes", df["tyre_size"].nunique())
    kpis[2].metric("Avg rim", round(df["rim_inch"].mean(), 1))
    kpis[3].metric("Avg width", f"{round(df['width_mm'].mean(), 1)} mm")
    kpis[4].metric("Risk share", f"{round(df['risk_binary'].mean() * 100, 1)}%")

    filtered_dimension = dimension[dimension["tyre_size"].isin(df["tyre_size"].unique())].copy()
    metric = st.selectbox(
        "Metric for dimension ranking",
        ["wear_rate_avg", "risk_share_pct", "remaining_life_avg_km", "cost_per_1000km_avg", "td_avg_mm"],
    )
    top_n = st.slider("Number of dimensions", 5, 25, 12)
    ranked = filtered_dimension.sort_values(metric, ascending=metric not in ["remaining_life_avg_km", "td_avg_mm"]).head(top_n)
    st.plotly_chart(
        px.bar(
            ranked,
            x="tyre_size",
            y=metric,
            color="rim_inch",
            hover_data=["width_mm", "aspect_ratio", "observations", "brands", "risk_share_pct"],
            title="Dimension ranking",
        ),
        width="stretch",
    )

    heatmap = df.groupby(["width_mm", "rim_inch"], as_index=False).agg(
        wear_rate=("wear_rate_mm_10000km", "mean"),
        risk_share=("risk_binary", lambda s: s.mean() * 100),
        observations=("tyre_id", "count"),
    )
    st.plotly_chart(
        px.density_heatmap(
            heatmap,
            x="width_mm",
            y="rim_inch",
            z="wear_rate",
            histfunc="avg",
            text_auto=".2f",
            title="Heatmap: average wear rate by width and rim",
        ),
        width="stretch",
    )

    x_axis = st.selectbox("X axis", ["width_mm", "aspect_ratio", "rim_inch", "mileage_km", "vehicle_weight_kg"], index=0)
    y_axis = st.selectbox("Y axis", ["wear_rate_mm_10000km", "remaining_life_km", "td_current_mm", "cost_per_1000km"], index=0)
    color_axis = st.selectbox("Colour", ["tyre_size", "risk_class", "brand", "season_type", "vehicle_type"], index=1)
    st.plotly_chart(
        px.scatter(
            df.sample(min(2500, len(df)), random_state=42),
            x=x_axis,
            y=y_axis,
            color=color_axis,
            size="rim_inch",
            hover_data=["brand", "model", "tyre_size", "width_mm", "aspect_ratio", "rim_inch"],
            title="Dynamic dimension scatter",
        ),
        width="stretch",
    )

    st.plotly_chart(
        px.treemap(
            df,
            path=["season_type", "tyre_segment", "tyre_size"],
            values="price_eur",
            color="wear_rate_mm_10000km",
            hover_data=["risk_class", "remaining_life_km"],
            title="Portfolio map by season, segment and tyre size",
        ),
        width="stretch",
    )

    monthly = data["dimension_monthly"]
    selected_sizes = ranked["tyre_size"].head(8).tolist()
    monthly = monthly[monthly["tyre_size"].isin(selected_sizes)]
    st.plotly_chart(
        px.line(
            monthly,
            x="month",
            y="wear_rate_avg",
            color="tyre_size",
            markers=True,
            title="Monthly wear-rate pattern for top selected dimensions",
        ),
        width="stretch",
    )


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
    st.info("Dimension variables are included: width_mm, aspect_ratio and rim_inch.")


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
    st.info("The ML feature set includes tyre_size as a categorical feature plus width_mm, aspect_ratio and rim_inch as numeric features.")


def real_data_reference() -> None:
    data = load_data()
    utqg = data["utqg"]
    sources = data["real_sources"]
    st.title("Real Data Reference")
    st.caption("Official real-data layer used for benchmarking and source transparency.")
    if not sources.empty:
        st.dataframe(sources, width="stretch", hide_index=True)
    if not utqg.empty:
        st.plotly_chart(
            px.bar(
                utqg,
                x="class_label",
                y="share_pct",
                color="rating_system",
                facet_col="rating_system",
                facet_col_wrap=3,
                title="NHTSA UTQGS official published rating distributions",
            ),
            width="stretch",
        )
        st.dataframe(utqg, width="stretch", hide_index=True)
    st.info("The official reference layer is real. The tread-depth-by-mileage panel remains simulated because repeated TD measurements by tyre, km and geography are not generally published as open data.")


def minmax_score(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    spread = numeric.max() - numeric.min()
    if pd.isna(spread) or spread == 0:
        return pd.Series(0.5, index=series.index)
    score = (numeric - numeric.min()) / spread
    return score if higher_is_better else 1 - score


def decision_report_html(best: pd.Series, candidates: pd.DataFrame, weights: dict[str, float], filters: dict[str, object]) -> str:
    rows = []
    for _, item in candidates.head(10).iterrows():
        rows.append(
            "<tr>"
            f"<td>{escape(str(item['brand']))}</td>"
            f"<td>{escape(str(item['model']))}</td>"
            f"<td>{escape(str(item['tyre_size']))}</td>"
            f"<td>{round(float(item['decision_score']), 3)}</td>"
            f"<td>{int(item['remaining_life_km'])}</td>"
            f"<td>{round(float(item['cost_per_1000km']), 2)}</td>"
            f"<td>{escape(str(item['risk_class']))}</td>"
            "</tr>"
        )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>TyreWear Decision Report</title>
  <style>
    body{{font-family:Arial,sans-serif;margin:0;background:#f6f9fb;color:#111827;line-height:1.55}}
    main{{max-width:1050px;margin:auto;padding:42px 22px}}
    h1{{font-size:42px;letter-spacing:-.04em;margin:0 0 8px}} h2{{margin-top:34px}}
    .card{{background:#fff;border:1px solid #dce9ee;border-radius:12px;padding:22px;margin:18px 0;box-shadow:0 14px 40px rgba(0,40,60,.08)}}
    .grid{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px}} .metric{{background:#ecf8fb;border-radius:10px;padding:14px}}
    table{{width:100%;border-collapse:collapse;background:#fff}} th,td{{padding:10px;border-bottom:1px solid #e5edf1;text-align:left}} th{{font-size:12px;text-transform:uppercase;color:#65737f}}
    @media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}
  </style>
</head>
<body>
<main>
  <p>TyreWear Intelligence</p>
  <h1>Decision Report</h1>
  <div class="card">
    <h2>Recommended option</h2>
    <p><strong>{escape(str(best['brand']))} {escape(str(best['model']))}</strong> in size <strong>{escape(str(best['tyre_size']))}</strong>.</p>
    <div class="grid">
      <div class="metric"><b>Decision score</b><br>{round(float(best['decision_score']), 3)}</div>
      <div class="metric"><b>Remaining life</b><br>{int(best['remaining_life_km']):,} km</div>
      <div class="metric"><b>Cost / 1000 km</b><br>{round(float(best['cost_per_1000km']), 2)} EUR</div>
      <div class="metric"><b>Risk</b><br>{escape(str(best['risk_class']))}</div>
    </div>
  </div>
  <div class="card">
    <h2>Decision logic</h2>
    <p>The score combines durability, safety, cost, efficiency, comfort and risk using the selected business weights.</p>
    <p><strong>Filters:</strong> {escape(json.dumps(filters, ensure_ascii=False))}</p>
    <p><strong>Weights:</strong> {escape(json.dumps(weights, ensure_ascii=False))}</p>
  </div>
  <h2>Top candidates</h2>
  <table>
    <thead><tr><th>Brand</th><th>Model</th><th>Size</th><th>Score</th><th>Life km</th><th>Cost</th><th>Risk</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</main>
</body>
</html>"""


def decision_center() -> None:
    recs = load_data()["recommendations"].copy()
    st.title("Decision Center & Export")
    st.caption("Turn the analysis into an explicit tyre decision, then export the evidence.")
    if recs.empty:
        st.warning("Recommendation data is not available. Run the pipeline first.")
        return

    left, right = st.columns([0.35, 0.65])
    with left:
        st.subheader("Decision profile")
        countries = ["All"] + sorted(recs["country"].dropna().unique())
        vehicles = ["All"] + sorted(recs["vehicle_type"].dropna().unique())
        sizes = ["All"] + sorted(recs["tyre_size"].dropna().unique())
        country = st.selectbox("Country", countries)
        vehicle = st.selectbox("Vehicle type", vehicles)
        tyre_size = st.selectbox("Tyre size", sizes)
        budget = st.slider("Maximum price EUR", 50, 360, 220)
        durability = st.slider("Durability weight", 0, 100, 30)
        safety = st.slider("Safety weight", 0, 100, 25)
        cost = st.slider("Cost weight", 0, 100, 20)
        efficiency = st.slider("Efficiency weight", 0, 100, 10)
        comfort = st.slider("Comfort weight", 0, 100, 10)
        risk = st.slider("Risk reduction weight", 0, 100, 5)

    candidates = recs[recs["price_eur"] <= budget].copy()
    if country != "All":
        candidates = candidates[candidates["country"] == country]
    if vehicle != "All":
        candidates = candidates[candidates["vehicle_type"] == vehicle]
    if tyre_size != "All":
        candidates = candidates[candidates["tyre_size"] == tyre_size]
    if candidates.empty:
        st.warning("No tyre candidates match the current decision profile.")
        return

    raw_weights = {
        "durability": durability,
        "safety": safety,
        "cost": cost,
        "efficiency": efficiency,
        "comfort": comfort,
        "risk": risk,
    }
    total_weight = sum(raw_weights.values()) or 1
    weights = {key: value / total_weight for key, value in raw_weights.items()}
    risk_penalty = candidates["risk_class"].map({"low": 1.0, "medium": 0.66, "high": 0.33, "critical": 0.0}).fillna(0.5)
    candidates["decision_score"] = (
        minmax_score(candidates["remaining_life_km"]) * weights["durability"]
        + candidates["safety_score"].fillna(0.5) * weights["safety"]
        + minmax_score(candidates["cost_per_1000km"], higher_is_better=False) * weights["cost"]
        + candidates["efficiency_score"].fillna(0.5) * weights["efficiency"]
        + candidates["comfort_score"].fillna(0.5) * weights["comfort"]
        + risk_penalty * weights["risk"]
    )
    candidates = candidates.sort_values("decision_score", ascending=False)
    best = candidates.iloc[0]

    with right:
        st.subheader("Recommended decision")
        metrics = st.columns(5)
        metrics[0].metric("Best tyre", f"{best['brand']} {best['model']}")
        metrics[1].metric("Size", best["tyre_size"])
        metrics[2].metric("Decision score", round(float(best["decision_score"]), 3))
        metrics[3].metric("Life", f"{int(best['remaining_life_km']):,} km")
        metrics[4].metric("Cost", f"{round(float(best['cost_per_1000km']), 2)} EUR")
        st.plotly_chart(
            px.bar(
                candidates.head(12),
                x="model",
                y="decision_score",
                color="risk_class",
                hover_data=["brand", "tyre_size", "remaining_life_km", "cost_per_1000km", "price_eur"],
                title="Top decision candidates",
            ),
            width="stretch",
        )
        st.plotly_chart(
            px.scatter(
                candidates.head(80),
                x="cost_per_1000km",
                y="remaining_life_km",
                color="risk_class",
                size="decision_score",
                hover_data=["brand", "model", "tyre_size", "price_eur"],
                title="Decision frontier: cost vs remaining life",
            ),
            width="stretch",
        )

    export_cols = [
        "tyre_id", "brand", "model", "tyre_size", "country", "vehicle_type", "price_eur",
        "remaining_life_km", "cost_per_1000km", "risk_class", "decision_score",
        "safety_score", "durability_score", "cost_score", "comfort_score", "efficiency_score",
    ]
    export_df = candidates[export_cols].head(100)
    filters = {"country": country, "vehicle_type": vehicle, "tyre_size": tyre_size, "max_price_eur": budget}
    top_records = json.loads(export_df.head(20).to_json(orient="records"))
    best_record = json.loads(best[export_cols].to_frame().T.to_json(orient="records"))[0]
    report = {
        "recommended_tyre": best_record,
        "filters": filters,
        "weights": weights,
        "top_candidates": top_records,
    }
    html_report = decision_report_html(best, export_df, weights, filters)

    st.subheader("Export analysis")
    st.dataframe(export_df, width="stretch", hide_index=True)
    d1, d2, d3 = st.columns(3)
    d1.download_button("Download CSV", export_df.to_csv(index=False), "tyrewear_decision_analysis.csv", "text/csv")
    d2.download_button("Download JSON", json.dumps(report, indent=2), "tyrewear_decision_analysis.json", "application/json")
    d3.download_button("Download HTML report", html_report, "tyrewear_decision_report.html", "text/html")


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
