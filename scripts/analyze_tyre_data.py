from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT_DIR / "data" / "raw" / "tyrewear_europe_raw.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "data" / "outputs"


def ensure_dirs() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    before_rows = len(raw)
    df = raw.drop_duplicates(subset=["observation_id"]).copy()
    df["observed_at"] = pd.to_datetime(df["observed_at"])
    numeric_cols = [
        "monthly_km",
        "cumulative_km",
        "tyre_age_months",
        "initial_tread_depth_mm",
        "tread_depth_mm",
        "wear_rate_mm_per_1000km",
        "projected_life_km",
        "price_eur",
        "cost_per_km",
        "wet_grip_score",
        "energy_efficiency_score",
        "noise_db",
        "road_roughness",
        "avg_temp_c",
        "rain_index",
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df.loc[df["price_eur"] <= 0, "price_eur"] = np.nan
    df.loc[~df["tread_depth_mm"].between(1.0, 9.5), "tread_depth_mm"] = np.nan
    df.loc[~df["wear_rate_mm_per_1000km"].between(0.03, 0.5), "wear_rate_mm_per_1000km"] = np.nan
    df.loc[~df["wet_grip_score"].between(1, 5), "wet_grip_score"] = np.nan
    missing_before = df[numeric_cols].isna().sum().to_dict()
    for col in numeric_cols:
        df[col] = df.groupby(["country", "tyre_season"])[col].transform(lambda s: s.fillna(s.median()))
        df[col] = df[col].fillna(df[col].median())
    df["cost_per_km"] = (df["price_eur"] / df["projected_life_km"]).round(5)
    df["high_wear_risk"] = df["high_wear_risk"].astype(int)
    summary = {
        "raw_rows": before_rows,
        "clean_rows": len(df),
        "duplicates_removed": int(before_rows - len(df)),
        "missing_values_before_imputation": {k: int(v) for k, v in missing_before.items() if v},
        "countries": int(df["country"].nunique()),
        "regions": int(df["europe_region"].nunique()),
        "date_min": df["observed_at"].min().date().isoformat(),
        "date_max": df["observed_at"].max().date().isoformat(),
    }
    return df, summary


def build_resamples(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = (
        df.set_index("observed_at")
        .groupby(["country", "europe_region", pd.Grouper(freq="ME")])
        .agg(
            observations=("observation_id", "count"),
            avg_tread_depth_mm=("tread_depth_mm", "mean"),
            avg_wear_rate=("wear_rate_mm_per_1000km", "mean"),
            avg_cost_per_km=("cost_per_km", "mean"),
            high_risk_share=("high_wear_risk", "mean"),
            avg_projected_life_km=("projected_life_km", "mean"),
        )
        .reset_index()
        .rename(columns={"observed_at": "month"})
    )
    monthly["high_risk_share"] = (monthly["high_risk_share"] * 100).round(2)
    annual = (
        df.groupby(["year", "country", "europe_region"], as_index=False)
        .agg(
            observations=("observation_id", "count"),
            avg_tread_depth_mm=("tread_depth_mm", "mean"),
            avg_wear_rate=("wear_rate_mm_per_1000km", "mean"),
            avg_cost_per_km=("cost_per_km", "mean"),
            high_risk_share=("high_wear_risk", "mean"),
            avg_projected_life_km=("projected_life_km", "mean"),
        )
    )
    annual["high_risk_share"] = (annual["high_risk_share"] * 100).round(2)
    numeric_monthly = monthly.select_dtypes(include=[np.number]).columns
    numeric_annual = annual.select_dtypes(include=[np.number]).columns
    monthly[numeric_monthly] = monthly[numeric_monthly].round(4)
    annual[numeric_annual] = annual[numeric_annual].round(4)
    return monthly, annual


def hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sample = df["wear_rate_mm_per_1000km"].sample(n=min(5000, len(df)), random_state=42)
    shapiro_stat, shapiro_p = stats.shapiro(sample)
    rows.append({"test": "Shapiro-Wilk", "question": "Is wear rate normally distributed?", "statistic": shapiro_stat, "p_value": shapiro_p, "decision": "normal" if shapiro_p >= 0.05 else "non-normal"})

    summer = df[df["tyre_season"] == "summer"]["wear_rate_mm_per_1000km"]
    winter = df[df["tyre_season"] == "winter"]["wear_rate_mm_per_1000km"]
    t_stat, t_p = stats.ttest_ind(summer, winter, equal_var=False)
    rows.append({"test": "Welch t-test", "question": "Summer vs winter wear rate mean", "statistic": t_stat, "p_value": t_p, "decision": "different" if t_p < 0.05 else "not different"})

    mw_stat, mw_p = stats.mannwhitneyu(summer, winter, alternative="two-sided")
    rows.append({"test": "Mann-Whitney U", "question": "Summer vs winter wear distribution", "statistic": mw_stat, "p_value": mw_p, "decision": "different" if mw_p < 0.05 else "not different"})

    groups = [frame["wear_rate_mm_per_1000km"].values for _, frame in df.groupby("europe_region")]
    anova_stat, anova_p = stats.f_oneway(*groups)
    rows.append({"test": "ANOVA", "question": "Wear rate mean across European regions", "statistic": anova_stat, "p_value": anova_p, "decision": "different" if anova_p < 0.05 else "not different"})

    kw_stat, kw_p = stats.kruskal(*groups)
    rows.append({"test": "Kruskal-Wallis", "question": "Wear rate distribution across European regions", "statistic": kw_stat, "p_value": kw_p, "decision": "different" if kw_p < 0.05 else "not different"})

    premium = df[df["brand"].isin(["Michelin", "Continental", "Goodyear"])]["projected_life_km"]
    value = df[~df["brand"].isin(["Michelin", "Continental", "Goodyear"])]["projected_life_km"]
    premium_t, premium_p = stats.ttest_ind(premium, value, equal_var=False)
    rows.append({"test": "Welch t-test", "question": "Premium vs value brand projected life", "statistic": premium_t, "p_value": premium_p, "decision": "different" if premium_p < 0.05 else "not different"})
    out = pd.DataFrame(rows)
    out["statistic"] = out["statistic"].astype(float).round(5)
    out["p_value"] = out["p_value"].astype(float).map(lambda x: float(f"{x:.8g}"))
    return out


def train_ml(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    features = [
        "country",
        "europe_region",
        "brand",
        "dimension",
        "tyre_season",
        "weather_season",
        "vehicle_type",
        "monthly_km",
        "cumulative_km",
        "tyre_age_months",
        "tread_depth_mm",
        "price_eur",
        "wet_grip_score",
        "energy_efficiency_score",
        "noise_db",
        "road_roughness",
        "avg_temp_c",
        "rain_index",
        "season_match",
    ]
    categorical = ["country", "europe_region", "brand", "dimension", "tyre_season", "weather_season", "vehicle_type"]
    numeric = [f for f in features if f not in categorical]
    preprocessor = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ]
    )
    X = df[features]
    y_class = df["high_wear_risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y_class, test_size=0.2, random_state=42, stratify=y_class)
    clf = Pipeline(
        [
            ("prep", preprocessor),
            ("model", RandomForestClassifier(n_estimators=220, max_depth=12, min_samples_leaf=12, random_state=42, n_jobs=-1)),
        ]
    )
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    proba = clf.predict_proba(X_test)[:, 1]

    y_reg = df["projected_life_km"]
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    reg = Pipeline(
        [
            ("prep", preprocessor),
            ("model", RandomForestRegressor(n_estimators=220, max_depth=14, min_samples_leaf=8, random_state=42, n_jobs=-1)),
        ]
    )
    reg.fit(X_train_r, y_train_r)
    pred_r = reg.predict(X_test_r)

    metrics = {
        "classification_high_wear_risk": {
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "f1": round(float(f1_score(y_test, pred)), 4),
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
        },
        "regression_projected_life_km": {
            "mae": round(float(mean_absolute_error(y_test_r, pred_r)), 2),
            "r2": round(float(r2_score(y_test_r, pred_r)), 4),
        },
    }

    transformed = preprocessor.fit_transform(X)
    cluster_features = transformed
    kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto")
    clusters = kmeans.fit_predict(cluster_features)
    cluster_df = df[["observation_id", "country", "europe_region", "brand", "tyre_season", "wear_rate_mm_per_1000km", "projected_life_km", "cost_per_km", "high_wear_risk"]].copy()
    cluster_df["cluster"] = clusters
    cluster_summary = (
        cluster_df.groupby("cluster", as_index=False)
        .agg(
            observations=("observation_id", "count"),
            avg_wear_rate=("wear_rate_mm_per_1000km", "mean"),
            avg_projected_life_km=("projected_life_km", "mean"),
            avg_cost_per_km=("cost_per_km", "mean"),
            high_risk_share=("high_wear_risk", "mean"),
        )
        .round(4)
    )
    cluster_summary["high_risk_share"] = (cluster_summary["high_risk_share"] * 100).round(2)
    return metrics, cluster_df, cluster_summary


def pattern_outputs(df: pd.DataFrame) -> dict:
    corr = df[["wear_rate_mm_per_1000km", "tread_depth_mm", "projected_life_km", "cost_per_km", "road_roughness", "rain_index", "avg_temp_c", "noise_db"]].corr(numeric_only=True)
    top_country_risk = (
        df.groupby(["country", "europe_region"], as_index=False)
        .agg(high_risk_share=("high_wear_risk", "mean"), avg_wear_rate=("wear_rate_mm_per_1000km", "mean"), avg_cost_per_km=("cost_per_km", "mean"))
        .assign(high_risk_share=lambda d: (d["high_risk_share"] * 100).round(2))
        .sort_values("high_risk_share", ascending=False)
    )
    brand_value = (
        df.groupby("brand", as_index=False)
        .agg(avg_projected_life_km=("projected_life_km", "mean"), avg_cost_per_km=("cost_per_km", "mean"), avg_wear_rate=("wear_rate_mm_per_1000km", "mean"), high_risk_share=("high_wear_risk", "mean"))
        .assign(high_risk_share=lambda d: (d["high_risk_share"] * 100).round(2))
        .sort_values(["avg_cost_per_km", "avg_projected_life_km"], ascending=[True, False])
    )
    corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    top_country_risk.round(4).to_csv(OUTPUT_DIR / "country_risk_patterns.csv", index=False)
    brand_value.round(4).to_csv(OUTPUT_DIR / "brand_value_patterns.csv", index=False)
    return {
        "strongest_correlations_with_wear_rate": corr["wear_rate_mm_per_1000km"].drop("wear_rate_mm_per_1000km").abs().sort_values(ascending=False).head(5).round(4).to_dict(),
        "highest_risk_country": top_country_risk.iloc[0].to_dict(),
        "best_value_brand": brand_value.iloc[0].to_dict(),
    }


def main() -> None:
    ensure_dirs()
    raw = pd.read_csv(RAW_PATH)
    clean, cleaning_summary = clean_data(raw)
    monthly, annual = build_resamples(clean)
    tests = hypothesis_tests(clean)
    ml_metrics, clusters, cluster_summary = train_ml(clean)
    patterns = pattern_outputs(clean)

    clean.to_csv(PROCESSED_DIR / "tyrewear_europe_clean.csv", index=False)
    monthly.to_csv(PROCESSED_DIR / "tyrewear_europe_monthly_resampled.csv", index=False)
    annual.to_csv(PROCESSED_DIR / "tyrewear_europe_annual_country.csv", index=False)
    tests.to_csv(OUTPUT_DIR / "hypothesis_tests.csv", index=False)
    clusters.to_csv(OUTPUT_DIR / "tyre_clusters.csv", index=False)
    cluster_summary.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)

    summary = {
        "cleaning": cleaning_summary,
        "hypothesis_tests": tests.to_dict(orient="records"),
        "machine_learning": ml_metrics,
        "patterns": patterns,
    }
    (OUTPUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
