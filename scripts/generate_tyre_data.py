from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
DBT_SEED_DIR = ROOT_DIR / "dbt_tyre_warehouse" / "seeds"
SEED = 42

COUNTRIES = {
    "Portugal": ("Southern Europe", "Lisbon", 38.7223, -9.1393, "mediterranean", 0.50),
    "Spain": ("Southern Europe", "Madrid", 40.4168, -3.7038, "mediterranean", 0.48),
    "Italy": ("Southern Europe", "Milan", 45.4642, 9.19, "mediterranean", 0.53),
    "France": ("Western Europe", "Lyon", 45.764, 4.8357, "temperate", 0.56),
    "Germany": ("Western Europe", "Berlin", 52.52, 13.405, "continental", 0.62),
    "Netherlands": ("Western Europe", "Amsterdam", 52.3676, 4.9041, "oceanic", 0.54),
    "Sweden": ("Northern Europe", "Stockholm", 59.3293, 18.0686, "cold", 0.65),
    "Norway": ("Northern Europe", "Oslo", 59.9139, 10.7522, "cold", 0.68),
    "Poland": ("Eastern Europe", "Warsaw", 52.2297, 21.0122, "continental", 0.60),
    "Czechia": ("Eastern Europe", "Prague", 50.0755, 14.4378, "continental", 0.58),
}
BRANDS = {
    "Michelin": ("premium", 1.14, 1.22),
    "Continental": ("premium", 1.11, 1.18),
    "Goodyear": ("premium", 1.06, 1.11),
    "Pirelli": ("premium", 1.03, 1.12),
    "Bridgestone": ("mid-range", 1.02, 1.04),
    "Hankook": ("mid-range", 0.97, 0.92),
    "Yokohama": ("mid-range", 0.95, 0.90),
    "BudgetRoad": ("budget", 0.84, 0.70),
}
TYRE_SIZES = ["195/65 R15", "205/55 R16", "225/45 R17", "235/45 R18", "255/40 R19"]
SEASON_TYPES = ["summer", "winter", "all-season"]
VEHICLE_TYPES = ["city", "family", "suv", "ev", "fleet"]
DRIVE_TYPES = ["fwd", "rwd", "awd"]
WHEEL_POSITIONS = ["front_left", "front_right", "rear_left", "rear_right"]
ROAD_TYPES = ["urban", "highway", "mixed", "mountain", "rural"]


def parse_size(size: str) -> tuple[int, int, int]:
    width = int(size.split("/")[0])
    aspect = int(size.split("/")[1].split(" ")[0])
    rim = int(size.split("R")[1])
    return width, aspect, rim


def season_of_year(month: int) -> str:
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def build_dataset(rows: int = 2500) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    random.seed(SEED)
    dates = pd.date_range("2019-01-01", "2025-12-31", freq="D")
    records = []
    for idx in range(rows):
        country = random.choice(list(COUNTRIES))
        region, city, lat, lon, climate_zone, country_roughness = COUNTRIES[country]
        brand = random.choice(list(BRANDS))
        segment, durability_factor, price_factor = BRANDS[brand]
        tyre_size = random.choice(TYRE_SIZES)
        width, aspect, rim = parse_size(tyre_size)
        season_type = random.choice(SEASON_TYPES)
        vehicle_type = random.choice(VEHICLE_TYPES)
        drive_type = random.choice(DRIVE_TYPES)
        wheel_position = random.choice(WHEEL_POSITIONS)
        road_type = random.choice(ROAD_TYPES)
        measurement_date = random.choice(dates)
        season = season_of_year(measurement_date.month)

        vehicle_weight_kg = {
            "city": rng.normal(1220, 120),
            "family": rng.normal(1500, 170),
            "suv": rng.normal(1840, 210),
            "ev": rng.normal(1980, 240),
            "fleet": rng.normal(1650, 190),
        }[vehicle_type]
        mileage_km = max(1200, rng.gamma(4.5, 6200))
        td_initial_mm = rng.uniform(7.2, 8.9)
        position_factor = 1.12 if wheel_position.startswith("front") and drive_type == "fwd" else 1.0
        position_factor += 0.08 if wheel_position.startswith("rear") and drive_type == "rwd" else 0
        road_factor = {"urban": 0.95, "highway": 0.88, "mixed": 1.0, "mountain": 1.16, "rural": 1.08}[road_type]
        climate_factor = {"mediterranean": 1.03, "temperate": 1.0, "continental": 1.07, "oceanic": 1.04, "cold": 1.11}[climate_zone]
        season_match = (season_type == "winter" and season == "winter") or (season_type == "summer" and season == "summer") or (season_type == "all-season" and season in {"spring", "autumn"})
        weight_factor = 1 + max(vehicle_weight_kg - 1450, 0) / 6000
        wear_rate_mm_10000km = (
            1.18
            * road_factor
            * climate_factor
            * position_factor
            * weight_factor
            * country_roughness
            / durability_factor
            * (0.92 if season_match else 1.08)
            + rng.normal(0, 0.12)
        )
        wear_rate_mm_10000km = max(0.35, wear_rate_mm_10000km)
        wear_mm = wear_rate_mm_10000km * mileage_km / 10000
        td_current_mm = float(np.clip(td_initial_mm - wear_mm, 1.1, 8.7))
        side_bias = rng.normal(0, 0.18)
        td_inner_mm = float(np.clip(td_current_mm - abs(side_bias) - rng.uniform(0, 0.22), 1.0, 8.8))
        td_center_mm = float(np.clip(td_current_mm + rng.normal(0, 0.08), 1.0, 8.8))
        td_outer_mm = float(np.clip(td_current_mm - abs(rng.normal(0, 0.16)), 1.0, 8.8))
        remaining_life_km = max(0, (td_current_mm - 1.6) / wear_rate_mm_10000km * 10000)
        predicted_life_km = mileage_km + remaining_life_km
        price_eur = max(45, (58 + width * 0.22 + rim * 8.5 + (vehicle_type == "ev") * 20) * price_factor + rng.normal(0, 12))
        cost_per_1000km = price_eur / max(predicted_life_km, 1) * 1000
        fuel_efficiency_class = random.choices(list("ABCDE"), weights=[durability_factor, 1.3, 1.1, 0.8, 0.4], k=1)[0]
        wet_grip_class = random.choices(list("ABCDE"), weights=[durability_factor * (1.15 if season_match else 0.9), 1.2, 0.9, 0.55, 0.3], k=1)[0]
        rolling_resistance = float(np.clip(7.5 - "ABCDE".index(fuel_efficiency_class) + rng.normal(0, 0.45), 1, 10))
        noise_db = float(np.clip(73 - durability_factor * 1.2 + road_factor * 1.5 + rng.normal(0, 1.3), 66, 78))
        if td_current_mm < 1.8 or remaining_life_km < 2500:
            risk_class = "critical"
        elif td_current_mm < 2.8 or remaining_life_km < 8000:
            risk_class = "high"
        elif td_current_mm < 4.0 or remaining_life_km < 18000:
            risk_class = "medium"
        else:
            risk_class = "low"
        cluster_id = {"low": 0, "medium": 1, "high": 2, "critical": 3}[risk_class]

        records.append(
            {
                "tyre_id": f"TYR{idx + 1:05d}",
                "brand": brand,
                "model": f"{brand[:3].upper()}-{random.choice(['Eco', 'Grip', 'Tour', 'Drive', 'Sport'])}-{random.randint(100, 999)}",
                "tyre_size": tyre_size,
                "width_mm": width,
                "aspect_ratio": aspect,
                "rim_inch": rim,
                "season_type": season_type,
                "tyre_segment": segment,
                "vehicle_type": vehicle_type,
                "vehicle_weight_kg": round(vehicle_weight_kg, 0),
                "drive_type": drive_type,
                "wheel_position": wheel_position,
                "country": country,
                "region": region,
                "city": city,
                "latitude": round(lat + rng.normal(0, 0.08), 5),
                "longitude": round(lon + rng.normal(0, 0.08), 5),
                "climate_zone": climate_zone,
                "road_type": road_type,
                "measurement_date": measurement_date.date().isoformat(),
                "season_of_year": season,
                "mileage_km": round(mileage_km, 0),
                "td_initial_mm": round(td_initial_mm, 2),
                "td_current_mm": round(td_current_mm, 2),
                "td_inner_mm": round(td_inner_mm, 2),
                "td_center_mm": round(td_center_mm, 2),
                "td_outer_mm": round(td_outer_mm, 2),
                "wear_mm": round(max(0, wear_mm), 2),
                "wear_rate_mm_10000km": round(wear_rate_mm_10000km, 3),
                "predicted_life_km": round(predicted_life_km, 0),
                "remaining_life_km": round(remaining_life_km, 0),
                "fuel_efficiency_class": fuel_efficiency_class,
                "wet_grip_class": wet_grip_class,
                "noise_db": round(noise_db, 1),
                "rolling_resistance": round(rolling_resistance, 2),
                "price_eur": round(price_eur, 2),
                "cost_per_1000km": round(cost_per_1000km, 2),
                "risk_class": risk_class,
                "cluster_id": cluster_id,
            }
        )
    df = pd.DataFrame(records)
    dirty = pd.concat([df, df.sample(60, random_state=SEED)], ignore_index=True)
    for col in ["td_current_mm", "wear_rate_mm_10000km", "price_eur", "remaining_life_km"]:
        dirty.loc[dirty.sample(frac=0.01, random_state=SEED + len(col)).index, col] = np.nan
    dirty.loc[dirty.sample(frac=0.004, random_state=99).index, "price_eur"] *= -1
    return dirty


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    DBT_SEED_DIR.mkdir(parents=True, exist_ok=True)
    df = build_dataset()
    raw_path = RAW_DIR / "sample_tyre_dataset.csv"
    seed_path = DBT_SEED_DIR / "sample_tyre_dataset.csv"
    df.to_csv(raw_path, index=False)
    df.drop_duplicates("tyre_id").to_csv(seed_path, index=False)
    print(f"Created dataset: {raw_path} ({len(df):,} rows including dirty duplicates)")
    print(f"Updated dbt seed: {seed_path}")


if __name__ == "__main__":
    main()
