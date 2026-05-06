from __future__ import annotations

from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent.parent
REFERENCE_DIR = ROOT_DIR / "data" / "reference"


def main() -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    source = "NHTSA TireWise / Uniform Tire Quality Grading System"
    url = "https://www.nhtsa.gov/vehicle-safety/tires"
    distributions = [
        ("treadwear", "below_200", "<200", 15.0, "Lower relative treadwear grade"),
        ("treadwear", "201_300", "201-300", 25.0, "NHTSA published treadwear distribution"),
        ("treadwear", "301_400", "301-400", 32.0, "NHTSA published treadwear distribution"),
        ("treadwear", "401_500", "401-500", 20.0, "NHTSA published treadwear distribution"),
        ("treadwear", "501_600", "501-600", 6.0, "NHTSA published treadwear distribution"),
        ("treadwear", "above_600", ">600", 2.0, "Higher relative treadwear grade"),
        ("traction", "AA", "AA", 15.0, "Highest wet traction grade"),
        ("traction", "A", "A", 77.0, "NHTSA published traction distribution"),
        ("traction", "B", "B", 7.0, "NHTSA published traction distribution"),
        ("traction", "C", "C", 0.2, "NHTSA reports only four current tire lines rated C"),
        ("temperature", "A", "A", 62.0, "Highest temperature resistance grade"),
        ("temperature", "B", "B", 34.0, "NHTSA published temperature distribution"),
        ("temperature", "C", "C", 4.0, "NHTSA published temperature distribution"),
    ]
    reference = pd.DataFrame(
        distributions,
        columns=["rating_system", "class_code", "class_label", "share_pct", "interpretation"],
    )
    reference["source"] = source
    reference["source_url"] = url
    reference.to_csv(REFERENCE_DIR / "nhtsa_utqg_distribution.csv", index=False)

    sources = pd.DataFrame(
        [
            {
                "source_name": "NHTSA TireWise / UTQGS Tire Safety Ratings",
                "source_url": url,
                "data_type": "official_real_aggregate_reference",
                "used_for": "Benchmarking treadwear, traction and temperature rating distributions.",
                "limitation": "Public page provides official distributions and rating definitions, not repeated tread-depth measurements by mileage.",
            },
            {
                "source_name": "Data.gov UTQGS Tire Rating Lookup catalog",
                "source_url": "https://catalog.data.gov/dataset/uniform-tire-quality-grading-system-utqgs-tire-rating-lookup",
                "data_type": "official_real_catalog_metadata",
                "used_for": "Documents the public NHTSA UTQGS tire-line rating lookup dataset and public-domain license.",
                "limitation": "The Socrata resource is exposed as a non-tabular landing page in the public endpoint tested here.",
            },
            {
                "source_name": "European Commission EPREL Tyres",
                "source_url": "https://energy-efficient-products.ec.europa.eu/product-list/tyres_en",
                "data_type": "official_real_market_reference",
                "used_for": "Explains EU tyre label dimensions: rolling resistance, wet grip, noise, snow and ice suitability.",
                "limitation": "Bulk API access requires API terms/key; public UI and PDFs can be integrated in a later connector.",
            },
        ]
    )
    sources.to_csv(REFERENCE_DIR / "real_tyre_data_sources.csv", index=False)
    print(f"Real reference data written to {REFERENCE_DIR}")


if __name__ == "__main__":
    main()
