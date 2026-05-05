from __future__ import annotations

import random
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
SEED = 42

COUNTRIES = {
    "Portugal": {"region": "Southern Europe", "temp": 17.5, "road": 0.50, "rain": 0.42},
    "Spain": {"region": "Southern Europe", "temp": 18.5, "road": 0.48, "rain": 0.35},
    "France": {"region": "Western Europe", "temp": 13.5, "road": 0.56, "rain": 0.55},
    "Germany": {"region": "Western Europe", "temp": 10.5, "road": 0.62, "rain": 0.58},
    "Netherlands": {"region": "Western Europe", "temp": 10.0, "road": 0.54, "rain": 0.66},
    "Italy": {"region": "Southern Europe", "temp": 16.0, "road": 0.53, "rain": 0.44},
    "Sweden": {"region": "Northern Europe", "temp": 5.5, "road": 0.65, "rain": 0.50},
    "Norway": {"region": "Northern Europe", "temp": 4.5, "road": 0.68, "rain": 0.57},
    "Poland": {"region": "Eastern Europe", "temp": 8.7, "road": 0.60, "rain": 0.50},
    "Czechia": {"region": "Eastern Europe", "temp": 9.2, "road": 0.58, "rain": 0.48},
}

BRANDS = ["Michelin", "Continental", "Goodyear", "Pirelli", "Bridgestone", "Hankook", "Nokian", "Yokohama"]
SEASONS = ["summer", "winter", "all-season"]
DIMENSIONS = ["205/55 R16", "225/45 R17", "235/45 R18", "195/65 R15", "255/40 R19"]
VEHICLE_TYPES = ["city", "family", "suv", "fleet", "ev"]


def season_for_month(month: int) -> str:
    if month in {12, 1, 2}:
        return "winter"
    if month in {6, 7, 8}:
        return "summer"
    return "mild"


def build_rows(rows: int = 18_000) -> pd.DataFrame:
    random.seed(SEED)
    rng = np.random.default_rng(SEED)
    dates = pd.date_range("2019-01-01", "2025-12-31", freq="D")
    records = []
    for idx in range(rows):
        country = random.choice(list(COUNTRIES))
        context = COUNTRIES[country]
        observed_at = random.choice(dates)
        weather_season = season_for_month(observed_at.month)
        tyre_season = random.choices(SEASONS, weights=[0.42, 0.24, 0.34], k=1)[0]
        brand = random.choice(BRANDS)
        vehicle_type = random.choice(VEHICLE_TYPES)
        dimension = random.choice(DIMENSIONS)

        brand_quality = {
            "Michelin": 1.12,
            "Continental": 1.09,
            "Goodyear": 1.04,
            "Pirelli": 1.00,
            "Bridgestone": 1.03,
            "Hankook": 0.96,
            "Nokian": 1.06,
            "Yokohama": 0.94,
        }[brand]
        season_match = int(
            (weather_season == "winter" and tyre_season == "winter")
            or (weather_season == "summer" and tyre_season == "summer")
            or (weather_season == "mild" and tyre_season == "all-season")
        )
        monthly_km = max(220, rng.normal(1350, 390))
        tyre_age_months = rng.integers(1, 54)
        cumulative_km = monthly_km * tyre_age_months * rng.uniform(0.82, 1.18)
        initial_tread_depth_mm = rng.normal(8.1, 0.45)
        road_roughness = np.clip(rng.normal(context["road"], 0.08), 0.25, 0.88)
        avg_temp_c = rng.normal(context["temp"], 5.2)
        rain_index = np.clip(rng.normal(context["rain"], 0.12), 0.05, 0.95)
        ev_penalty = 0.08 if vehicle_type == "ev" else 0.0
        suv_penalty = 0.06 if vehicle_type == "suv" else 0.0

        wear_rate_mm_per_1000km = (
            0.115
            + road_roughness * 0.085
            + rain_index * 0.035
            + max(avg_temp_c - 20, 0) * 0.002
            + ev_penalty
            + suv_penalty
            - (brand_quality - 1.0) * 0.075
            - season_match * 0.018
            + rng.normal(0, 0.018)
        )
        wear_rate_mm_per_1000km = max(0.055, wear_rate_mm_per_1000km)
        tread_depth_mm = initial_tread_depth_mm - wear_rate_mm_per_1000km * (cumulative_km / 1000)
        tread_depth_mm = np.clip(tread_depth_mm + rng.normal(0, 0.22), 1.2, 8.8)
        projected_life_km = max(12_000, ((initial_tread_depth_mm - 1.6) / wear_rate_mm_per_1000km) * 1000)
        price_eur = (
            62
            + brand_quality * 45
            + DIMENSIONS.index(dimension) * 16
            + (vehicle_type == "ev") * 22
            + rng.normal(0, 13)
        )
        price_eur = max(48, price_eur)
        cost_per_km = price_eur / projected_life_km
        wet_grip_score = np.clip(2.2 + brand_quality * 0.7 + season_match * 0.22 - road_roughness * 0.2 + rng.normal(0, 0.35), 1, 5)
        energy_efficiency_score = np.clip(3.7 - wear_rate_mm_per_1000km * 2.2 + brand_quality * 0.25 + rng.normal(0, 0.35), 1, 5)
        noise_db = np.clip(73 - brand_quality * 1.4 + road_roughness * 2.5 + rng.normal(0, 1.7), 66, 78)
        risk_score = (
            (3.2 - tread_depth_mm) * 0.55
            + wear_rate_mm_per_1000km * 3.2
            + road_roughness * 0.65
            + (1 - wet_grip_score / 5) * 0.75
            - season_match * 0.25
        )
        high_wear_risk = int(risk_score > 0.72)

        records.append(
            {
                "observation_id": f"OBS{idx + 1:06d}",
                "observed_at": observed_at.date().isoformat(),
                "year": observed_at.year,
                "month": observed_at.month,
                "country": country,
                "europe_region": context["region"],
                "brand": brand,
                "model": f"{brand[:3].upper()}-{random.choice(['Eco', 'Grip', 'Tour', 'Drive', 'Sport'])}-{random.randint(100, 999)}",
                "dimension": dimension,
                "tyre_season": tyre_season,
                "weather_season": weather_season,
                "vehicle_type": vehicle_type,
                "monthly_km": round(monthly_km, 1),
                "cumulative_km": round(cumulative_km, 1),
                "tyre_age_months": int(tyre_age_months),
                "initial_tread_depth_mm": round(initial_tread_depth_mm, 2),
                "tread_depth_mm": round(tread_depth_mm, 2),
                "wear_rate_mm_per_1000km": round(wear_rate_mm_per_1000km, 4),
                "projected_life_km": round(projected_life_km, 0),
                "price_eur": round(price_eur, 2),
                "cost_per_km": round(cost_per_km, 5),
                "wet_grip_score": round(wet_grip_score, 2),
                "energy_efficiency_score": round(energy_efficiency_score, 2),
                "noise_db": round(noise_db, 1),
                "road_roughness": round(road_roughness, 3),
                "avg_temp_c": round(avg_temp_c, 1),
                "rain_index": round(rain_index, 3),
                "season_match": season_match,
                "high_wear_risk": high_wear_risk,
            }
        )

    df = pd.DataFrame(records)
    dirty = df.copy()
    for col in ["tread_depth_mm", "price_eur", "wear_rate_mm_per_1000km", "wet_grip_score"]:
        missing_idx = dirty.sample(frac=0.012, random_state=SEED + len(col)).index
        dirty.loc[missing_idx, col] = np.nan
    dupes = dirty.sample(n=120, random_state=SEED)
    dirty = pd.concat([dirty, dupes], ignore_index=True)
    dirty.loc[dirty.sample(frac=0.004, random_state=99).index, "price_eur"] *= -1
    return dirty


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    df = build_rows()
    output = RAW_DIR / "tyrewear_europe_raw.csv"
    df.to_csv(output, index=False)
    print(f"Created raw tyre dataset: {output} ({len(df):,} rows)")


if __name__ == "__main__":
    main()
