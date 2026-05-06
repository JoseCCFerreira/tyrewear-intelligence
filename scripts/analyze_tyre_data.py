from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT_DIR / "data" / "raw" / "sample_tyre_dataset.csv"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
OUTPUT_DIR = ROOT_DIR / "data" / "outputs"


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.drop_duplicates("tyre_id").copy()
    df["measurement_date"] = pd.to_datetime(df["measurement_date"])
    numeric = [
        "vehicle_weight_kg", "latitude", "longitude", "mileage_km", "td_initial_mm", "td_current_mm",
        "td_inner_mm", "td_center_mm", "td_outer_mm", "wear_mm", "wear_rate_mm_10000km",
        "predicted_life_km", "remaining_life_km", "noise_db", "rolling_resistance", "price_eur",
        "cost_per_1000km",
    ]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.loc[df["price_eur"] <= 0, "price_eur"] = np.nan
    df.loc[~df["td_current_mm"].between(0, 12), "td_current_mm"] = np.nan
    df.loc[~df["td_initial_mm"].between(5, 12), "td_initial_mm"] = np.nan
    df.loc[df["wear_rate_mm_10000km"] < 0, "wear_rate_mm_10000km"] = np.nan
    df.loc[df["remaining_life_km"] < 0, "remaining_life_km"] = np.nan
    missing = {k: int(v) for k, v in df[numeric].isna().sum().to_dict().items() if v}
    for col in numeric:
        df[col] = df.groupby(["country", "season_type"])[col].transform(lambda s: s.fillna(s.median()))
        df[col] = df[col].fillna(df[col].median())
    df["wear_mm"] = (df["td_initial_mm"] - df["td_current_mm"]).clip(lower=0)
    df["wear_rate_mm_10000km"] = (df["wear_mm"] / df["mileage_km"].replace(0, np.nan) * 10000).fillna(df["wear_rate_mm_10000km"]).clip(lower=0)
    df["remaining_life_km"] = ((df["td_current_mm"] - 1.6).clip(lower=0) / df["wear_rate_mm_10000km"].replace(0, np.nan) * 10000).fillna(0)
    df["predicted_life_km"] = df["mileage_km"] + df["remaining_life_km"]
    df["cost_per_1000km"] = df["price_eur"] / df["predicted_life_km"].replace(0, np.nan) * 1000
    df["risk_binary"] = df["risk_class"].isin(["high", "critical"]).astype(int)
    summary = {
        "raw_rows": int(len(raw)),
        "clean_rows": int(len(df)),
        "duplicates_removed": int(len(raw) - len(df)),
        "missing_values_before_imputation": missing,
        "countries": int(df["country"].nunique()),
        "regions": int(df["region"].nunique()),
        "date_min": df["measurement_date"].min().date().isoformat(),
        "date_max": df["measurement_date"].max().date().isoformat(),
    }
    return df, summary


def resample_outputs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = (
        df.set_index("measurement_date")
        .groupby(["country", "region", pd.Grouper(freq="ME")])
        .agg(
            observations=("tyre_id", "count"),
            td_avg_mm=("td_current_mm", "mean"),
            wear_rate_avg=("wear_rate_mm_10000km", "mean"),
            remaining_life_avg_km=("remaining_life_km", "mean"),
            cost_per_1000km_avg=("cost_per_1000km", "mean"),
            risk_share_pct=("risk_binary", lambda s: s.mean() * 100),
        )
        .reset_index()
        .rename(columns={"measurement_date": "month"})
    )
    annual = (
        df.assign(year=df["measurement_date"].dt.year)
        .groupby(["year", "country", "region"], as_index=False)
        .agg(
            observations=("tyre_id", "count"),
            td_avg_mm=("td_current_mm", "mean"),
            wear_rate_avg=("wear_rate_mm_10000km", "mean"),
            remaining_life_avg_km=("remaining_life_km", "mean"),
            cost_per_1000km_avg=("cost_per_1000km", "mean"),
            risk_share_pct=("risk_binary", lambda s: s.mean() * 100),
        )
    )
    for frame in (monthly, annual):
        numeric_cols = frame.select_dtypes(include=[np.number]).columns
        frame[numeric_cols] = frame[numeric_cols].round(4)
    return monthly, annual


def hypothesis_tests(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    sample = df["wear_rate_mm_10000km"].sample(min(2500, len(df)), random_state=42)
    shapiro = stats.shapiro(sample)
    rows.append(["Shapiro-Wilk", "Wear-rate normality", shapiro.statistic, shapiro.pvalue, "normal" if shapiro.pvalue >= 0.05 else "non-normal"])
    ks = stats.kstest((sample - sample.mean()) / sample.std(), "norm")
    rows.append(["Kolmogorov-Smirnov", "Wear-rate normality vs standard normal", ks.statistic, ks.pvalue, "normal" if ks.pvalue >= 0.05 else "non-normal"])
    premium = df[df["tyre_segment"] == "premium"]["wear_rate_mm_10000km"]
    budget = df[df["tyre_segment"] == "budget"]["wear_rate_mm_10000km"]
    levene = stats.levene(premium, budget)
    rows.append(["Levene", "Equal variance premium vs budget", levene.statistic, levene.pvalue, "equal variance" if levene.pvalue >= 0.05 else "different variance"])
    ttest = stats.ttest_ind(premium, budget, equal_var=True)
    rows.append(["t-test", "Premium vs budget wear-rate mean", ttest.statistic, ttest.pvalue, "different" if ttest.pvalue < 0.05 else "not different"])
    welch = stats.ttest_ind(premium, budget, equal_var=False)
    rows.append(["Welch t-test", "Premium vs budget wear-rate mean", welch.statistic, welch.pvalue, "different" if welch.pvalue < 0.05 else "not different"])
    groups = [g["wear_rate_mm_10000km"].values for _, g in df.groupby("region")]
    anova = stats.f_oneway(*groups)
    rows.append(["ANOVA", "Wear-rate means across regions", anova.statistic, anova.pvalue, "different" if anova.pvalue < 0.05 else "not different"])
    mw = stats.mannwhitneyu(premium, budget)
    rows.append(["Mann-Whitney", "Premium vs budget distribution", mw.statistic, mw.pvalue, "different" if mw.pvalue < 0.05 else "not different"])
    kw = stats.kruskal(*groups)
    rows.append(["Kruskal-Wallis", "Wear-rate distributions across regions", kw.statistic, kw.pvalue, "different" if kw.pvalue < 0.05 else "not different"])
    chi = stats.chi2_contingency(pd.crosstab(df["region"], df["risk_class"]))
    rows.append(["Chi-square", "Region vs risk class association", chi.statistic, chi.pvalue, "associated" if chi.pvalue < 0.05 else "not associated"])
    pearson = stats.pearsonr(df["mileage_km"], df["td_current_mm"])
    rows.append(["Pearson", "Mileage vs current tread depth", pearson.statistic, pearson.pvalue, "correlated" if pearson.pvalue < 0.05 else "not correlated"])
    spearman = stats.spearmanr(df["mileage_km"], df["td_current_mm"])
    rows.append(["Spearman", "Mileage vs current tread depth rank", spearman.statistic, spearman.pvalue, "correlated" if spearman.pvalue < 0.05 else "not correlated"])
    return pd.DataFrame(rows, columns=["test", "question", "statistic", "p_value", "decision"]).round({"statistic": 5, "p_value": 8})


def preprocessors(features: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric = [f for f in features if f not in categorical]
    return ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric),
        ]
    )


def ml_and_clusters(df: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    features = [
        "brand", "tyre_size", "season_type", "tyre_segment", "vehicle_type", "drive_type", "wheel_position",
        "country", "region", "climate_zone", "road_type", "mileage_km", "td_current_mm", "wear_rate_mm_10000km",
        "vehicle_weight_kg", "noise_db", "rolling_resistance", "price_eur", "cost_per_1000km",
    ]
    categorical = ["brand", "tyre_size", "season_type", "tyre_segment", "vehicle_type", "drive_type", "wheel_position", "country", "region", "climate_zone", "road_type"]
    X = df[features]
    prep = preprocessors(features, categorical)

    y_reg = df["remaining_life_km"]
    X_train, X_test, y_train, y_test = train_test_split(X, y_reg, test_size=0.2, random_state=42)
    regressors = {
        "Ridge": Ridge(alpha=1.0),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=180, random_state=42, min_samples_leaf=5, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
    }
    reg_metrics = []
    for name, model in regressors.items():
        pipe = Pipeline([("prep", prep), ("model", model)])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        reg_metrics.append({"model": name, "mae": round(mean_absolute_error(y_test, pred), 2), "r2": round(r2_score(y_test, pred), 4)})

    y_clf = df["risk_binary"]
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=180, max_depth=12, min_samples_leaf=6, random_state=42, n_jobs=-1),
    }
    clf_metrics = []
    for name, model in classifiers.items():
        pipe = Pipeline([("prep", prep), ("model", model)])
        pipe.fit(X_train_c, y_train_c)
        pred = pipe.predict(X_test_c)
        proba = pipe.predict_proba(X_test_c)[:, 1]
        clf_metrics.append({"model": name, "accuracy": round(accuracy_score(y_test_c, pred), 4), "f1": round(f1_score(y_test_c, pred), 4), "roc_auc": round(roc_auc_score(y_test_c, proba), 4)})

    cluster_features = ["td_current_mm", "wear_rate_mm_10000km", "remaining_life_km", "cost_per_1000km", "vehicle_weight_kg", "rolling_resistance"]
    scaled = StandardScaler().fit_transform(df[cluster_features])
    kmeans = KMeans(n_clusters=5, random_state=42, n_init="auto").fit(scaled)
    gmm = GaussianMixture(n_components=5, random_state=42).fit(scaled)
    dbscan = DBSCAN(eps=0.85, min_samples=20).fit(scaled)
    pca = PCA(n_components=2, random_state=42).fit_transform(scaled)
    clusters = df[["tyre_id", "brand", "tyre_size", "country", "region", "risk_class", "td_current_mm", "wear_rate_mm_10000km", "remaining_life_km", "cost_per_1000km", "latitude", "longitude"]].copy()
    clusters["kmeans_cluster"] = kmeans.labels_
    clusters["gmm_cluster"] = gmm.predict(scaled)
    clusters["dbscan_cluster"] = dbscan.labels_
    clusters["pca_1"] = pca[:, 0]
    clusters["pca_2"] = pca[:, 1]
    cluster_summary = clusters.groupby("kmeans_cluster", as_index=False).agg(
        observations=("tyre_id", "count"),
        td_avg_mm=("td_current_mm", "mean"),
        wear_rate_avg=("wear_rate_mm_10000km", "mean"),
        remaining_life_avg_km=("remaining_life_km", "mean"),
        cost_per_1000km_avg=("cost_per_1000km", "mean"),
    ).round(4)
    clustering_metrics = {
        "kmeans_silhouette": round(float(silhouette_score(scaled, kmeans.labels_)), 4),
        "kmeans_davies_bouldin": round(float(davies_bouldin_score(scaled, kmeans.labels_)), 4),
        "dbscan_clusters": int(len(set(dbscan.labels_)) - (1 if -1 in dbscan.labels_ else 0)),
    }
    metrics = {"regression": reg_metrics, "classification": clf_metrics, "clustering": clustering_metrics}
    return metrics, clusters, cluster_summary, pd.DataFrame(pca, columns=["pca_1", "pca_2"])


def recommendation_outputs(df: pd.DataFrame) -> pd.DataFrame:
    scored = df.copy()
    scored["safety_score"] = (5 - scored["wet_grip_class"].map({c: i for i, c in enumerate("ABCDE", start=1)})) / 4
    scored["durability_score"] = scored["remaining_life_km"].rank(pct=True)
    scored["cost_score"] = 1 - scored["cost_per_1000km"].rank(pct=True)
    scored["comfort_score"] = 1 - scored["noise_db"].rank(pct=True)
    scored["efficiency_score"] = (5 - scored["fuel_efficiency_class"].map({c: i for i, c in enumerate("ABCDE", start=1)})) / 4
    scored["recommended_tyre_score"] = (
        scored["durability_score"] * 0.30
        + scored["safety_score"] * 0.25
        + scored["cost_score"] * 0.20
        + scored["efficiency_score"] * 0.15
        + scored["comfort_score"] * 0.10
    )
    return scored.sort_values("recommended_tyre_score", ascending=False)


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(RAW_PATH)
    clean, cleaning = clean_data(raw)
    monthly, annual = resample_outputs(clean)
    tests = hypothesis_tests(clean)
    ml_metrics, clusters, cluster_summary, _ = ml_and_clusters(clean)
    recommended = recommendation_outputs(clean)
    corr = clean[["td_current_mm", "wear_rate_mm_10000km", "remaining_life_km", "predicted_life_km", "cost_per_1000km", "vehicle_weight_kg", "rolling_resistance", "noise_db", "mileage_km"]].corr()
    country_patterns = clean.groupby(["country", "region"], as_index=False).agg(
        td_avg_mm=("td_current_mm", "mean"),
        wear_rate_avg=("wear_rate_mm_10000km", "mean"),
        remaining_life_avg_km=("remaining_life_km", "mean"),
        cost_per_1000km_avg=("cost_per_1000km", "mean"),
        risk_share_pct=("risk_binary", lambda s: s.mean() * 100),
    ).round(4)
    brand_patterns = clean.groupby(["brand", "tyre_segment"], as_index=False).agg(
        td_avg_mm=("td_current_mm", "mean"),
        wear_rate_avg=("wear_rate_mm_10000km", "mean"),
        remaining_life_avg_km=("remaining_life_km", "mean"),
        cost_per_1000km_avg=("cost_per_1000km", "mean"),
        recommended_tyre_score=("tyre_id", lambda _: 0),
    )
    brand_scores = recommended.groupby("brand")["recommended_tyre_score"].mean()
    brand_patterns["recommended_tyre_score"] = brand_patterns["brand"].map(brand_scores)
    clean.to_csv(PROCESSED_DIR / "tyrewear_europe_clean.csv", index=False)
    monthly.to_csv(PROCESSED_DIR / "tyrewear_europe_monthly_resampled.csv", index=False)
    annual.to_csv(PROCESSED_DIR / "tyrewear_europe_annual_country.csv", index=False)
    tests.to_csv(OUTPUT_DIR / "hypothesis_tests.csv", index=False)
    pd.DataFrame(ml_metrics["regression"]).to_csv(OUTPUT_DIR / "ml_regression_metrics.csv", index=False)
    pd.DataFrame(ml_metrics["classification"]).to_csv(OUTPUT_DIR / "ml_classification_metrics.csv", index=False)
    clusters.to_csv(OUTPUT_DIR / "tyre_clusters.csv", index=False)
    cluster_summary.to_csv(OUTPUT_DIR / "cluster_summary.csv", index=False)
    recommended.to_csv(OUTPUT_DIR / "recommendation_scores.csv", index=False)
    corr.to_csv(OUTPUT_DIR / "correlation_matrix.csv")
    country_patterns.to_csv(OUTPUT_DIR / "country_risk_patterns.csv", index=False)
    brand_patterns.round(4).to_csv(OUTPUT_DIR / "brand_value_patterns.csv", index=False)
    summary = {
        "cleaning": cleaning,
        "hypothesis_tests": tests.to_dict(orient="records"),
        "machine_learning": ml_metrics,
        "patterns": {
            "highest_risk_region": country_patterns.sort_values("risk_share_pct", ascending=False).iloc[0].to_dict(),
            "best_recommended_tyre": recommended.iloc[0][["tyre_id", "brand", "model", "tyre_size", "country", "recommended_tyre_score"]].to_dict(),
        },
    }
    (OUTPUT_DIR / "analysis_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
